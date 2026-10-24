"""Serve two-request E33 batch execution through an asynchronous owner worker."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--mode',choices=('none','adaptive'),default='adaptive')
    parser.add_argument('--capacity',type=int,default=1536);parser.add_argument('--port',type=int,default=18084)
    parser.add_argument('--execution',choices=('graph','eager'),default='graph')
    parser.add_argument('--draft-execution',choices=('sequential','batch'),default='sequential')
    parser.add_argument('--draft-w8-batch-library',type=Path)
    parser.add_argument('--draft-w8-batch-sha256')
    parser.add_argument('--batch-costs',type=Path)
    parser.add_argument('--batch-costs-sha256')
    args=parser.parse_args()
    if not 16<=args.capacity<=1536:parser.error('capacity must be 16..1536')
    if bool(args.batch_costs)!=bool(args.batch_costs_sha256):parser.error('batch costs and their SHA256 must be supplied together')
    if args.batch_costs and (args.mode!='adaptive' or args.execution!='graph'):
        parser.error('joint budgets require adaptive Graph execution')
    if args.draft_execution=='batch' and (args.mode!='adaptive' or args.batch_costs):
        parser.error('batch draft requires adaptive mode and recalibrated batch costs; old joint costs are unsupported')
    if args.draft_execution=='batch' and not (args.draft_w8_batch_library and args.draft_w8_batch_sha256):
        parser.error('batch draft requires an explicitly fingerprinted W8 batch library')
    if args.draft_execution!='batch' and (args.draft_w8_batch_library or args.draft_w8_batch_sha256):
        parser.error('W8 batch library requires --draft-execution batch')
    root=args.root.resolve();checkout=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(checkout),str(checkout/'scripts'),str(root/'followup-deps'),str(root/'python-deps')]
    from spark_infer_q.native_scheduler import NativeScheduler
    from spark_infer_q.kaifra_optimized import optimized_provider
    from spark_infer_q.kaifra_batch import batch_engine,BatchedRequestDraftProvider
    from spark_infer_q.kaifra_batch_target import BatchTarget
    from spark_infer_q.kaifra_worker import AsyncWorker
    from spark_infer_q.kaifra_service import make_server
    from spark_infer_q.screened_head_fused import FusedScreenedHead
    from e3_wide_target import WideTarget
    from e12_session import load_session
    from spark_infer_q.joint_budget import JointBudget
    joint_budget=None if args.batch_costs is None else JointBudget.load(args.batch_costs,sha256=args.batch_costs_sha256)
    with NativeScheduler(root.parent/'DeepInfer',root.parent/'build-release/lib/deepinfer',max_drafts=0 if args.mode=='none' else 7) as native:
        torch=native.torch
        with torch.inference_mode():
            session=load_session(root,args.checkpoint,lambda x:print(json.dumps(x),flush=True))
            provider,metadata=optimized_provider(session) if args.mode=='adaptive' else (None,{})
            if args.draft_execution=='batch':
                from spark_infer_q.draft_batch_w8 import load_library
                matmul,identity=load_library(args.draft_w8_batch_library,args.draft_w8_batch_sha256)
                provider=BatchedRequestDraftProvider(provider,matmul,kernel_identity=identity)
                metadata=provider.profile_metadata
            screen=FusedScreenedHead(session.weights.head)
            if not screen.static_ok:raise ValueError('E33 head range gate failed')
            target=BatchTarget(WideTarget(session.weights,capacity=2048),screen,graphs=args.execution=='graph')
            target.warmup();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        @contextmanager
        def context():
            with torch.inference_mode(),torch.cuda.stream(stream):yield
        worker=AsyncWorker(lambda:batch_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
            capacity=args.capacity,tokenizer=session.tokenizer,target_only=args.mode=='none',joint_budget=joint_budget),context_factory=context)
        try:
            worker.start();native.bind_worker(worker.thread);server=make_server(worker,('127.0.0.1',args.port))
            print(json.dumps(dict(status='READY',mode=args.mode,metadata=metadata,**{'worker':worker.snapshot()})),flush=True)
            try:server.serve_forever()
            except KeyboardInterrupt:pass
            finally:
                worker.close();server.server_close()
        finally:
            # A failed/unfinished fence is explicit: target.close refuses live
            # leases rather than freeing device state beneath the owner thread.
            if worker.thread is not None and worker.thread.is_alive():worker.close()
            target.close()


if __name__=='__main__':main()
