"""Validate a post-hoc control against the immutable main study, then export it."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from e4_tasks import TASKS


def summarize(control, main):
    assert control['status'] == 'PASS_WARMUP_CONTROL' and control['post_hoc'] is True
    assert len(control['tasks']) == len(TASKS)
    by_id = {t['id']: t for t in main['tasks']}
    values = []
    for ti, (task, rec) in enumerate(zip(TASKS, control['tasks'])):
        assert all(rec[k] == task[k] for k in task)
        original = by_id[task['id']]
        assert rec['prompt_ids'] == original['native_prompt_ids'], 'prompt mismatch'
        assert len(rec['warmup']) == 2 and len(rec['timings']) == 8
        assert [r['mode'] for r in rec['warmup']] == ['serial','native_mtp']
        for rep in range(4):
            rows = sorted((r for r in rec['timings'] if r['repetition'] == rep), key=lambda r:r['order'])
            modes = ['serial','native_mtp'] if (rep+ti)%2 == 0 else ['native_mtp','serial']
            assert [r['mode'] for r in rows] == modes
        out = dict(id=task['id'], arm=task['arm'], modes={})
        for mode in ('serial','native_mtp'):
            expected = original['modes']['native_'+mode]
            rows = [r for r in rec['timings'] if r['mode'] == mode]
            all_rows = rows + [r for r in rec['warmup'] if r['mode'] == mode]
            ref = expected['timings'][0]
            for row in all_rows:
                assert row['runtime']['token_sha'] in expected['token_hashes'], 'output hash mismatch'
                assert row['usage']['completion_tokens'] == expected['completion_token_counts'][0]
                assert row['content'] == ref['content'] and row['reasoning'] == ref['reasoning']
                assert row['runtime']['enable_thinking'] == task['thinking']
                assert row['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) == 0
                assert row['answer_check'] == ref['answer_check']
                assert row['finish_reason'] == ref['finish_reason']
                row['decode_ms'] = (row['usage']['completion_tokens']-1)*1000/row['runtime']['tokens_per_second']
                assert row['decode_ms'] > 0
            out['modes'][mode] = dict(decode_tokens=expected['completion_token_counts'][0]-1,
                **{k:statistics.median(r[k] for r in rows) for k in ('decode_ms','wall_ms','wire_ttft_ms')},
                max_min_decode_ratio=max(r['decode_ms'] for r in rows)/min(r['decode_ms'] for r in rows))
        values.append(out)
    agg = {}
    for arm in ('natural','capped','all'):
        chosen = [v for v in values if arm=='all' or v['arm']==arm]
        agg[arm] = {}
        for mode in ('serial','native_mtp'):
            rows = [v['modes'][mode] for v in chosen]
            n = sum(r['decode_tokens'] for r in rows); ms = sum(r['decode_ms'] for r in rows)
            agg[arm][mode] = dict(tasks=len(rows), decode_tokens=n, sum_median_decode_ms=ms,
                decode_tokens_per_s=n*1000/ms,
                sum_median_wall_ms=sum(r['wall_ms'] for r in rows),
                sum_median_ttft_ms=sum(r['wire_ttft_ms'] for r in rows))
    return dict(status='VALIDATED_POST_HOC_NATIVE_WARMUP_CONTROL', control=control,
        summary=dict(timed_runs=sum(len(t['timings']) for t in control['tasks']),
            full_case_warmups=sum(len(t['warmup']) for t in control['tasks']),
            output_hash_count_text_checks=sum(len(t['timings'])+len(t['warmup']) for t in control['tasks']),
            aggregates=agg, per_case=values),
        limits=['post hoc native-only sequential session, not ABBA or new held-out prompts',
            'per-case warmup reduces but cannot identify all initialization/order costs',
            'original cross-engine numerical and output differences remain'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('control',type=Path)
    p.add_argument('main_result',type=Path); p.add_argument('output',type=Path); a=p.parse_args()
    c=json.loads(a.control.read_text()); main_result=json.loads(a.main_result.read_text())
    repo=Path(__file__).resolve().parents[1]
    for name, sha in c['harness_sha256'].items():
        assert hashlib.sha256((repo/'scripts'/name).read_bytes()).hexdigest()==sha, name
    result=summarize(c,main_result)
    out=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):
        assert private not in out,private
    a.output.write_text(out); print(json.dumps(result['summary']['aggregates'],indent=2))


if __name__=='__main__':main()
