"""Capacity and timing gates reject shorter work or overstated quality claims."""
import unittest
from scripts.e46_export import coverage


def fixture():
    return dict(physical_capacity=262160,logical_capacity=262144,prompt_tokens=262080,output_cap=64,
        input=dict(prompt_tokens=262080),checkpoint_position_limit=262144,
        chunks=[dict(index=i,position=min((i+1)*1024,262080),rows=1024 if i<255 else 960,
                     wall_ms=1000.,MemAvailable=16*1024**3,allocated_bytes=80*1024**3) for i in range(256)],
        windows=[dict(width=w,parity=p,keep=w if w%2 else w//2,logits_exact=True,features_exact=True,state_exact=True) for p in (0,1) for w in range(1,9)],
        selector=[dict(position=p,complete_blocks=p//4,selected_keys=2048+p%4,exact=True) for p in (261889,261985,262080)],
        oracle_tokens_exact=True,oracle_state_exact=True,prefix_cache_unchanged=True,
        prefix_sha256='a'*64,prefix_after_sha256='a'*64,output_tokens=4,oracle_output_tokens=4,
        final_position=262083,oracle_position=262083,context_start=260043,retained_rows=2040,
        prefill_feature_rows=2040,feature_storage_bytes=104448000,
        trace=[dict(prompt=False,verified_rows=3,emitted=3,accepted=2,observed=2)],
        graph=dict(graph_fallbacks={},head_fallbacks=0,graph_captures=16,graph_replays=17),
        kv_dtype='bf16',paged_kv=False,multi_request=False,context_1m_validated=False,
        prefill_active_ms=256001.,first_token_ttft_ms=256001.,diagnostic_ms=10000.,decode_active_ms=3.,
        decode_active_tps=1000.,retrieval_hits=[True,False,False],generated_text='ALPHA',
        retrieval_keys=['ALPHA','BETA','GAMMA'],peak_allocated_bytes=81*1024**3)


class Direct256KEvidence(unittest.TestCase):
    def test_capacity_success_does_not_imply_retrieval_or_serving_quality(self):
        v=coverage(fixture())
        self.assertEqual((v['prompt_tokens'],v['retrieval_hits'],v['retrieval_total']),(262080,1,3))
        self.assertEqual((v['graph_numeric_buckets'],v['numeric_full_logits_rows']),(16,72))
        self.assertEqual(v['decode_active_tps'],1000.)

    def test_incomplete_prompt_or_missing_long_selector_cannot_publish(self):
        for case in ('prompt','chunk','selector','bucket'):
            v=fixture()
            if case=='prompt':v['prompt_tokens']=4096
            elif case=='chunk':v['chunks'].pop()
            elif case=='selector':v['selector'][0]['complete_blocks']=32768
            else:v['windows'].pop()
            with self.subTest(case=case),self.assertRaises(ValueError):coverage(v)

    def test_prefix_mutation_fallback_or_oracle_mismatch_rejected(self):
        for case in ('prefix','fallback','oracle','state'):
            v=fixture()
            if case=='prefix':v['prefix_after_sha256']='b'*64
            elif case=='fallback':v['graph']['head_fallbacks']=1
            elif case=='oracle':v['oracle_tokens_exact']=False
            else:v['windows'][-1]['state_exact']=False
            with self.subTest(case=case),self.assertRaises(ValueError):coverage(v)

    def test_decode_count_and_diagnostic_time_cannot_be_laundered(self):
        for case in ('tokens','diagnostic','memory','retrieval'):
            v=fixture()
            if case=='tokens':v['trace'][0]['emitted']=4
            elif case=='diagnostic':v['decode_active_ms']+=v['diagnostic_ms']
            elif case=='memory':v['chunks'][-1]['MemAvailable']=7*1024**3
            else:v['retrieval_hits']=[True]*3
            with self.subTest(case=case),self.assertRaises(ValueError):coverage(v)


if __name__=='__main__':unittest.main()
