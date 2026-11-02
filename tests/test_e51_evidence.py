"""Offline evidence validation; these fixtures are not model/GPU measurements."""
import copy
import unittest
from scripts.e51_export import check_run,token_sha
from scripts.e50_analyze import family


def fixture():
    ids=[11,12,13,14]
    metrics=[dict(prompt=True,context_rows=0,verified_rows=3,emitted=1,accepted=0,observed=0)]
    metrics += [dict(prompt=False,context_rows=p,verified_rows=1,emitted=1,accepted=0,observed=0,
                     draft_ms=.1,verify_ms=.2,commit_ms=.1) for p in (3,4,5)]
    return dict(mode='ring',warmup=True,output_tokens=4,output_ids=ids,output_sha256=token_sha(ids),
        events=[dict(emitted=1,elapsed=.003*(i+1),ttft=.003,finish='length' if i==3 else None) for i in range(4)],
        prompt_tokens=3,decode_ms=9.,decode_tokens=3,decode_tok_s=1000/3,ttft_ms=3.,request_ms=13.,
        metrics=metrics,live_state=dict(pos=6),graph_captures=16,graph_replays=3,verify_rounds=3,
        graph_fallbacks=[{}],head_fallbacks=0,draft_captures_before=10,draft_captures_after=10,
        kernel_hashes_unchanged=True,accepted=0,observed=0,
        phase_ms=dict(draft_ms=.3,verify_ms=.6,commit_ms=.3),peak_allocated_bytes=1024,state_storage_bytes=512)


class Evidence(unittest.TestCase):
    def test_fp8_attention_is_not_lost_in_other(self):
        self.assertEqual(family('_chunks8'),'attention')

    def test_event_work_and_timing(self):
        self.assertAlmostEqual(check_run(fixture(),formal=False)['decode_tok_s'],1000/3)

    def test_corrupted_observations_are_rejected(self):
        for case in ('rate','clock','outputs','position','replays','capture','phase'):
            r=copy.deepcopy(fixture())
            if case=='rate':r['decode_tok_s']*=2
            elif case=='clock':r['events'][1]['elapsed']=0
            elif case=='outputs':r['output_ids'][1]+=1
            elif case=='position':r['live_state']['pos']+=1
            elif case=='replays':r['graph_replays']=0
            elif case=='capture':r['draft_captures_after']=11
            else:r['phase_ms']['verify_ms']*=2
            with self.subTest(case=case),self.assertRaises(ValueError):check_run(r,formal=False)

    def test_small_fixture_is_not_full256k_evidence(self):
        with self.assertRaises(ValueError):check_run(fixture())


if __name__=='__main__':unittest.main()
