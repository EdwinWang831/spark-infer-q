"""Five-mode GPU experiment and same-prefix counterfactual width replay."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

from e3_followup_work import TASKS, selected_check
from spark_infer_q.budget_audit import MODES, candidate_scores, hindsight_paths, mode_order

# Frozen before this experiment. Disjoint from the eight reused regression tasks.
CALIBRATION = [
    ('Define gravity in one short sentence.', False),
    ('将 Thank you 翻译成中文，只输出译文。', False),
    ('Return only a JSON array containing 2, 4, and 6.', False),
    ('Write a short Python function is_even(n). Output only the function.', False),
    ('A basket has 18 apples. Four are removed and nine added. How many remain?', True),
    ('用两句话解释缓存为什么能减少重复计算。', False),
]
# Every decode position of these cases is replayed, not just visited fixed3 nodes.
# Prior results motivated this diagnostic selection: short EOS, mixed and high acceptance.
REPLAY_TASKS = {'translation', 'explanation', 'reasoning_code'}


def run(root, checkpoint, w, target, report, save, repetitions):
    import torch
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import Draft, upstream_helpers
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    from e3_mechanism_checks import entry, params
    import spark_q_e3_varlen

    budget_sha = hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()
    assert budget_sha == '722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    report['reference'] = {'budget_binary_sha256':budget_sha,
        'deepinfer_commit':'774ce244b87cc0eb8d4b94ab9f5de2ab22049f01',
        'draft_checkpoint':'incoai/Qwen3.8-Flash-Next-DFlash2',
        'draft_revision':'a22a1744e85c17364dcc54a6a420af7135d4c7f6',
        'previous_numerical_report':'2026-10-20-followup.json; official BF16 fidelity and target prefill unresolved'}
    h = hashlib.sha256()
    with (root/'checkpoint/model.safetensors').open('rb') as f:
        while chunk := f.read(8<<20): h.update(chunk)
    assert h.hexdigest() == 'd6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    helpers = upstream_helpers(root/'reference/dflash.py', 'b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft = Draft(root/'checkpoint', w.embed[0], w.head.b.weight, helpers, max_context=1536)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    generation = json.loads((checkpoint/'generation_config.json').read_text())
    ids = generation.get('eos_token_id', [])
    eos = set(ids if isinstance(ids, list) else [ids]); eos.add(tokenizer.eos_token_id); eos.discard(None)
    report['environment'] = {'torch': torch.__version__, 'capacity': 1536, 'eos_ids': sorted(eos),
        'target_dtype': 'original NVFP4/MXFP8 weights, BF16 activations; same serial prompt arithmetic',
        'drafter': 'unchanged E3 BF16 DFlash2 prototype; known official BF16 discrepancy remains open'}
    report['measurement_contract'] = {
        'modes': MODES, 'repetitions': repetitions,
        'order': 'five-mode cyclic Latin square, each mode occupies every position once per five repetitions',
        'scope': 'single request, bounded context, greedy, eight previously observed regression tasks',
        'decode_excludes': ['load', 'prompt', 'first pending output', 'state setup', 'calibration', 'oracle comparisons'],
        'correctness': 'separate per-kept-logit oracle runs; every timed final token/live state checked outside timer',
        'fixed_widths': 'fixed1/2/3 means 1/2/3 draft proposals; four-row drafter unchanged, verification cropped',
        'zero_draft': 'skip drafter, but retain target features to allow subsequent drafting; pure serial captures none',
        'prior': 'six independent calibration prompts capped at 192 output tokens, censored depth counts with Beta(1,1), online prior strength four',
        'runtime': 'original pinned DeepInfer MTPVarlen; full verify/commit/draft cost once, sampler zero',
        'replay': 'all serial positions of three diagnostic cases; three action repeats, rotated width order',
        'replay_limit': 'hindsight acceptance/EOS and independent action medians; excludes policy overhead; not online speed',
    }
    def emit(value): print(json.dumps(value, ensure_ascii=False), flush=True)
    def prompt_state(text, thinking):
        ids = tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
        assert 0 < len(ids) < 256
        st = target.state(); features = []
        for token in ids:
            logits, ft = target.step(st, [token]); features.append(ft)
        return ids, st, torch.cat(features), int(logits[-1].argmax())

    def generate(mode, start, context, first, limit, curve, prior, *, oracle=None, capture_oracle=False):
        st = start.clone(); ctx = context.clone() if mode != 'serial' else None
        output = [first]; steps = []; gold_logits = []; stopped = first in eos
        policy = LaggedAcceptance(prior)
        budget = spark_q_e3_varlen.BudgetProbe(); budget.curve(curve)
        for _ in range(5): budget.runtime(1, 1, curve[0][1], 0.)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(output) < limit and not stopped:
            before = len(output)
            if mode == 'serial':
                logits, _ = target.step(st, [output[-1]], capture=False)
                token = int(logits[-1].argmax()); output.append(token); stopped = token in eos
                if capture_oracle: gold_logits.append(logits.cpu())
                continue
            rates = policy.previous.copy()
            t = time.perf_counter()
            width = int(mode[-1])+1 if mode.startswith('fixed') else budget.schedule(params(),
                [entry(policy.current, policy.previous, process=len(steps)+2, schedule=len(steps)+2)])[0]
            decision_ms = (time.perf_counter()-t)*1000
            k = min(width-1, limit-len(output)-1)
            t = time.perf_counter()
            proposals = draft.propose(ctx, output[-1])[0][:k] if k else []
            torch.cuda.synchronize(); dm = (time.perf_counter()-t)*1000
            t = time.perf_counter(); logits, features, buffers = target.run(st, [output[-1]]+proposals)
            vm = (time.perf_counter()-t)*1000
            pred = logits.argmax(-1).tolist()
            out, keep, _, stopped = greedy_result(output[-1], proposals, pred, eos_ids=eos, max_new=limit-len(output))
            raw_accept = 0
            for proposed, golden in zip(proposals, pred):
                if proposed != golden: break
                raw_accept += 1
            observed = min(k, keep) if stopped else k
            accepted = min(raw_accept, observed)
            t = time.perf_counter(); target.commit(st, buffers, k+1, keep); cm = (time.perf_counter()-t)*1000
            ctx = torch.cat([ctx, features[:keep]]); output.extend(out)
            assert len(ctx) == st.pos == start.pos+len(output)-1
            if oracle is not None:
                assert out == oracle['tokens'][before:before+len(out)], (mode, before, 'tokens')
                for j in range(keep):
                    assert torch.equal(logits[j:j+1].cpu(), oracle['logits'][before-1+j]), (mode, before, j, 'logits')
            policy.observe(observed, accepted)
            budget.runtime(1, k+1, vm+cm+dm, 0.)
            steps.append({'width':width, 'verified_rows':k+1, 'observed_drafts':observed, 'accepted_drafts':accepted,
                'emitted_tokens':len(out), 'draft_ms':dm, 'verify_ms':vm, 'commit_ms':cm, 'decision_ms':decision_ms,
                'lagged_rates':rates, 'output_offset':before-1})
        torch.cuda.synchronize(); elapsed = (time.perf_counter()-t0)*1000
        return {'tokens':output, 'state':st, 'logits':gold_logits, 'steps':steps, 'decode_ms':elapsed,
            'stopped_by_eos':stopped, 'reached_limit':len(output)>=limit and not stopped}

    # Warm and measure all widths on identical states, including feature capture.
    report['stage'] = 'calibration'; report['calibration'] = {'prompts':[], 'cost_samples':[]}; save()
    success = [1.,1.,1.]; trials = [2.,2.,2.]
    samples = {width:[] for width in range(1,5)}; draft_samples = []
    for ci, (text, thinking) in enumerate(CALIBRATION):
        ids, st, ctx, first = prompt_state(text, thinking)
        proposals = draft.propose(ctx, first)[0]
        for width in range(1,5):
            scratch = st.clone(); lg, ft, b = target.run(scratch, [first]+proposals[:width-1])
            target.commit(scratch, b, width, width); del scratch
        for rep in range(3):
            t = time.perf_counter(); got = draft.propose(ctx, first)[0]; torch.cuda.synchronize()
            draft_samples.append((time.perf_counter()-t)*1000); assert got == proposals
            for wi in range(4):
                width = (wi+rep+ci)%4+1; scratch = st.clone(); torch.cuda.synchronize()
                t = time.perf_counter(); lg, ft, b = target.run(scratch, [first]+proposals[:width-1])
                target.commit(scratch, b, width, width)
                ms = (time.perf_counter()-t)*1000; samples[width].append(ms)
                report['calibration']['cost_samples'].append({'prompt_index':ci,'prefix_rows':len(ids),'width':width,'verify_commit_ms':ms})
                del scratch
        curve = [(width, statistics.median(samples[width])+(statistics.median(draft_samples) if width>1 else 0.)) for width in range(1,5)]
        calibration = generate('fixed3', st, ctx, first, 192, curve, (.5,.5,.5))
        for row in calibration['steps']:
            for depth in range(row['observed_drafts']):
                if depth > row['accepted_drafts']: break
                trials[depth] += 1; success[depth] += int(depth < row['accepted_drafts'])
        report['calibration']['prompts'].append({'prompt':text,'thinking':thinking,'prompt_tokens':len(ids),
            'tokens':calibration['tokens'],'stopped_by_eos':calibration['stopped_by_eos'],
            'reached_limit':calibration['reached_limit']})
        emit({'calibration_complete':ci,'output_tokens':len(calibration['tokens'])}); save()
        del st,ctx,calibration
    prior = [s/t for s,t in zip(success,trials)]
    curve = [(width, statistics.median(samples[width])+(statistics.median(draft_samples) if width>1 else 0.)) for width in range(1,5)]
    report['calibration'].update(conditional_success=success, conditional_trials=trials, conditional_prior=prior,
        draft_samples_ms=draft_samples, effective_cost_curve_ms=curve,
        startup_scores=candidate_scores([prior[0],prior[0]*prior[1],prior[0]*prior[1]*prior[2]], [ms for _,ms in curve]))
    save(); emit({'calibration_result':report['calibration']['startup_scores']})
    report['stage'] = 'balanced_generation'; report['tasks'] = []

    def replay(task, start, context, golden):
        st = start.clone(); ctx = context.clone(); positions = []
        tokens = golden['tokens']
        for offset, anchor in enumerate(tokens[:-1]):
            torch.cuda.synchronize(); t = time.perf_counter()
            proposals = draft.propose(ctx, anchor)[0]; torch.cuda.synchronize()
            draft_costs = [(time.perf_counter()-t)*1000]
            for _ in range(2):
                t = time.perf_counter(); got = draft.propose(ctx, anchor)[0]; torch.cuda.synchronize()
                draft_costs.append((time.perf_counter()-t)*1000); assert got == proposals
            records = {width:[] for width in range(1,5)}
            outcomes = {}
            for rep in range(3):
                for wi in range(4):
                    width = (wi+rep+offset)%4+1
                    k = min(width-1, task['limit']-offset-2)
                    scratch = st.clone(); torch.cuda.synchronize(); t = time.perf_counter()
                    logits, features, b = target.run(scratch, [anchor]+proposals[:k])
                    pred = logits.argmax(-1).tolist()
                    out, keep, _, stopped = greedy_result(anchor, proposals[:k], pred, eos_ids=eos,
                        max_new=task['limit']-offset-1)
                    target.commit(scratch, b, k+1, keep)
                    temporary_ctx = torch.cat([ctx, features[:keep]])
                    torch.cuda.synchronize(); action_ms = (time.perf_counter()-t)*1000
                    assert out == tokens[offset+1:offset+1+len(out)], (task['id'], offset, width, 'replay tokens')
                    for j in range(keep):
                        assert torch.equal(logits[j:j+1].cpu(), golden['logits'][offset+j]), (task['id'], offset, width, 'replay logits')
                    # Check committed state against serial replay outside the action timer.
                    if rep == 0:
                        expected = st.clone()
                        for token in [anchor]+out[:-1]: target.step(expected, [token], capture=False)
                        differences = target.state_difference(scratch, expected)
                        assert not differences, (task['id'], offset, width, differences)
                        del expected
                    outcomes[width] = len(out)
                    records[width].append(action_ms)
                    del scratch, temporary_ctx
            actions = [{'width':width, 'emitted_tokens':outcomes[width], 'draft_ms':statistics.median(draft_costs) if width>1 else 0.,
                'action_samples_ms':records[width], 'action_median_ms':statistics.median(records[width]),
                'total_ms':statistics.median(records[width])+(statistics.median(draft_costs) if width>1 else 0.)}
                for width in range(1,5)]
            positions.append(actions)
            lg, ft = target.step(st, [anchor]); ctx = torch.cat([ctx,ft])
            assert torch.equal(lg.cpu(), golden['logits'][offset])
            if offset%20 == 0: emit({'replay_task':task['id'],'position':offset,'of':len(tokens)-1})
        assert not target.state_difference(st, golden['state'])
        return {'positions':positions, 'paths':hindsight_paths(positions), 'all_kept_logits_and_live_states_exact':True}

    for ti, task in enumerate(TASKS):
        ids, start, ctx, first = prompt_state(task['prompt'], task['thinking'])
        golden = generate('serial', start, ctx, first, task['limit'], curve, prior, capture_oracle=True)
        assert golden['stopped_by_eos'] and not golden['reached_limit'], (task['id'], 'truncated')
        record = {**task, 'prompt_tokens':len(ids), 'token_ids':golden['tokens'],
            'text':tokenizer.decode(golden['tokens'], skip_special_tokens=False),
            'stopped_by_eos':True, 'correctness':[], 'timings':[]}
        record['selected_property_check'] = selected_check(task, record['text'])
        report['tasks'].append(record); save()
        for mode in MODES[1:]:
            check = generate(mode, start, ctx, first, task['limit'], curve, prior, oracle=golden)
            assert check['tokens'] == golden['tokens']
            assert not target.state_difference(check['state'], golden['state'])
            record['correctness'].append({'mode':mode,'tokens_equal':True,'kept_logits_bit_exact':True,
                'live_state_bit_exact':True,'steps':check['steps']})
            del check
        for rep in range(repetitions):
            for order,mode in enumerate(mode_order(rep, ti)):
                result = generate(mode, start, ctx, first, task['limit'], curve, prior)
                assert result['tokens'] == golden['tokens'], (task['id'], mode, 'timed tokens')
                assert not target.state_difference(result['state'], golden['state']), (task['id'], mode, 'timed state')
                row = {'mode':mode,'repetition':rep,'order':order,'decode_ms':result['decode_ms'],
                    'tokens_equal':True,'live_state_bit_exact':True,'steps':result['steps']}
                record['timings'].append(row); save()
                emit({'task':task['id'],'mode':mode,'repetition':rep,'decode_ms':row['decode_ms'],
                    'widths':sorted(set(s['width'] for s in result['steps']))})
                del result
        baseline = statistics.median(x['decode_ms'] for x in record['timings'] if x['mode']=='serial')
        record['summary'] = {}
        for mode in MODES:
            values = [x['decode_ms'] for x in record['timings'] if x['mode']==mode]
            median = statistics.median(values)
            record['summary'][mode] = {'median_ms':median,'min_ms':min(values),'max_ms':max(values),'ratio_of_medians':baseline/median}
        if task['id'] in REPLAY_TASKS:
            report['stage'] = 'position_replay'; save()
            record['counterfactual_replay'] = replay(task, start, ctx, golden)
            emit({'replay_complete':task['id'],'paths':record['counterfactual_replay']['paths']})
            report['stage'] = 'balanced_generation'
        save(); emit({'task_complete':task['id'],'summary':record['summary']})
        del start, ctx, golden
        gc.collect()
    base = sum(t['summary']['serial']['median_ms'] for t in report['tasks'])
    report['summary'] = {'tasks':len(report['tasks']),'completed_by_eos':len(report['tasks']),
        'timing_runs':sum(len(t['timings']) for t in report['tasks']),
        'sum_serial_medians_ms':base,'all_timed_tokens_and_live_states_equal':True,
        'observed_dynamic_widths':sorted({s['width'] for t in report['tasks'] for row in t['timings'] if row['mode']=='dynamic' for s in row['steps']})}
    for mode in MODES:
        ms = sum(t['summary'][mode]['median_ms'] for t in report['tasks'])
        report['summary'][mode] = {'sum_medians_ms':ms,'ratio':base/ms}
    report['stage'] = 'complete'; save(); emit({'summary':report['summary']})
