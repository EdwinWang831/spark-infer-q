"""Evidence arithmetic and real-kernel classification; fixtures are not GPU runs."""
import copy
import unittest
from scripts.e50_export import check_seeded,check_native,tokens_sha
from scripts.e50_analyze import analyze


def native_fixture():
    ids=[11,12,13,14]
    metrics=[dict(prompt=True,context_rows=0,verified_rows=3,emitted=1,accepted=0,observed=0,rates=[])]
    metrics.extend(dict(prompt=False,context_rows=p,verified_rows=1,emitted=1,accepted=0,observed=0,rates=[],draft_ms=.1,verify_ms=.2,commit_ms=.1) for p in (3,4,5))
    graph=dict(draft_graph_captures=10,draft_graph_failed=False,draft_graph_replays=10)
    return dict(mode='optimized',warmup=True,output_tokens=4,output_ids=ids,output_sha256=tokens_sha(ids),output_cap=4,
        prompt_tokens=3,finish_reason='length',events=[dict(emitted=1,position=3+i,finish='length' if i==3 else None,
            elapsed_seconds=.003*(i+1),ttft_seconds=.003) for i in range(4)],
        ttft_ms=3.,completion_ms=12.,request_ms=13.,decode_ms=9.,decode_tokens=3,decode_tok_s=1000/3,
        metrics=metrics,final_position=6,graph_captures=16,graph_replays=3,verify_rounds=3,graph_fallbacks=[{}],head_fallbacks=0,
        accepted=0,observed=0,phase_ms=dict(draft_ms=.3,verify_ms=.6,commit_ms=.3),
        draft_graph_before=graph.copy(),draft_graph_after=graph.copy(),native_output_reads=dict(full=0,cached=3),
        native_terminal_history_checked=True,peak_allocated_bytes=1024)


class EvidenceTests(unittest.TestCase):
    def test_kernel_parent_excluded_and_overlap_not_doubled(self):
        events=[dict(cat='user_annotation',ph='X',name='e50.verify',ts=0,dur=100),
            dict(cat='gpu_user_annotation',ph='X',name='parent',ts=0,dur=100),
            dict(cat='kernel',ph='X',name='nvfp4_expert_kernel',ts=10,dur=10),
            dict(cat='kernel',ph='X',name='_select_tiles',ts=15,dur=15),
            dict(cat='gpu_memcpy',ph='X',name='copy',ts=35,dur=5),
            dict(cat='kernel',ph='X',name='outside',ts=99,dur=10)]
        r=analyze(dict(traceEvents=events));self.assertEqual((r['kernel_calls'],r['kernel_sum_us'],r['kernel_union_us'],r['activity_union_us']),(2,25,20,25))
        self.assertEqual(r['families']['qsa_selection']['fraction'],.6)

    def test_native_cache_timing_work(self):
        self.assertEqual(check_native(native_fixture())['native_output_reads'],dict(full=0,cached=3))
        for case in ('clock','reads','replays','rate'):
            r=native_fixture()
            if case=='clock':r['events'][1]['elapsed_seconds']=0
            elif case=='reads':r['native_output_reads']['full']=1
            elif case=='replays':r['graph_replays']=0
            else:r['decode_tok_s']*=2
            with self.subTest(case=case),self.assertRaises(ValueError):check_native(r)

    def test_seeded_commit_and_hash_corruption_rejected(self):
        r=dict(mode='baseline',output_tokens=4,output_ids=[11,12,13,14],output_sha256=tokens_sha([11,12,13,14]),
            steps=[dict(position=p,keep=1,width=1,outputs=[t],accepted=0,observed=0) for p,t in zip((3,4,5),(12,13,14))],
            final_position=6,graph_replays=3,head_kernel_hashes_unchanged=True,state_sha256='d'*64,decode_ms=10.)
        check_seeded(r,3)
        for case in ('hash','position','keep'):
            cur=copy.deepcopy(r)
            if case=='hash':cur['output_sha256']='0'*64
            elif case=='position':cur['steps'][1]['position']=0
            else:cur['steps'][0]['keep']=2
            with self.subTest(case=case),self.assertRaises(ValueError):check_seeded(cur,3)

    def test_fixed_arm_rejected(self):
        r=native_fixture();r['mode']='fixed'
        with self.assertRaises(ValueError):check_native(r)


if __name__=='__main__':unittest.main()
