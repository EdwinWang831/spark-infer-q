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
    parser.add_argument('--prefill-execution',choices=('rows8','owned'),default='rows8')
    parser.add_argument('--prompt-token-budget',type=int,default=32)
    parser.add_argument('--prefill-chunk-rows',type=int,default=128)
    parser.add_argument('--interleave-prefill',action='store_true')
    parser.add_argument('--worker-cpu',choices=('default','performance'),default='default')
    parser.add_argument('--target-backend',choices=('legacy','compact'),default='legacy')
    parser.add_argument('--target-state-library',type=Path)
    parser.add_argument('--target-state-sha256')
    parser.add_argument('--draft-execution',choices=('sequential','batch'),default='sequential')
    parser.add_argument('--draft-buffers',action='store_true')
    parser.add_argument('--draft-graphs',action='store_true')
    parser.add_argument('--draft-w8-batch-library',type=Path)
    parser.add_argument('--draft-w8-batch-sha256')
    parser.add_argument('--batch-costs',type=Path)
    parser.add_argument('--batch-costs-sha256')
    args=parser.parse_args()
    if args.draft_graphs and not args.draft_buffers:parser.error('draft graphs require --draft-buffers')
    if (args.draft_buffers or args.draft_graphs) and args.draft_execution!='batch':
        parser.error('draft buffers/graphs require --draft-execution batch')
    if not 16<=args.prompt_token_budget<=512:parser.error('prompt token budget must be 16..512')
    if not 1<=args.prefill_chunk_rows<=512:parser.error('prefill chunk rows must be 1..512')
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
    if args.target_backend=='compact' and not (args.target_state_library and args.target_state_sha256):
        parser.error('compact Target requires an explicitly fingerprinted state library')
    if args.target_backend!='compact' and (args.target_state_library or args.target_state_sha256):
        parser.error('Target state library requires --target-backend compact')
    if args.target_backend=='compact' and args.batch_costs:
        parser.error('compact Target requires recalibrated costs; old joint costs are unsupported')
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
                provider=BatchedRequestDraftProvider(provider,matmul,kernel_identity=identity,
                    max_requests=2,buffers=args.draft_buffers,graphs=args.draft_graphs)
                metadata=provider.profile_metadata
            screen=FusedScreenedHead(session.weights.head)
            if not screen.static_ok:raise ValueError('E33 head range gate failed')
            if args.target_backend=='compact':
                from spark_infer_q.compact_target import CompactTarget
                from spark_infer_q.qwen_state_kernel import load_state_kernel
                state_kernel=load_state_kernel(args.target_state_library,args.target_state_sha256)
                target=CompactTarget(WideTarget(session.weights,capacity=2048),screen,state_kernel,graphs=args.execution=='graph')
            else:target=BatchTarget(WideTarget(session.weights,capacity=2048),screen,graphs=args.execution=='graph')
            target.warmup();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            prefill=None
            if args.prefill_execution=='owned':
                from spark_infer_q.prefill import PrefillExecutor
                from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider
                with torch.cuda.stream(stream):
                    prefill=PrefillExecutor(TensorFoldPrefillProvider(session.weights,torch,capacity=2048),chunk_rows=args.prefill_chunk_rows)
        placement={}
        @contextmanager
        def context():
            from spark_infer_q.worker_cpu import worker_cpu
            with worker_cpu(args.worker_cpu) as metadata,torch.inference_mode(),torch.cuda.stream(stream):
                placement.update(metadata);yield
        def make():
            if args.draft_graphs:provider.backend.warmup()
            engine=batch_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                capacity=args.capacity,tokenizer=session.tokenizer,target_only=args.mode=='none',joint_budget=joint_budget,
                prefill_executor=prefill,prompt_token_budget=args.prompt_token_budget,interleave_prefill=args.interleave_prefill)
            engine.capabilities.update(placement);return engine
        worker=AsyncWorker(make,context_factory=context)
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
            if prefill is not None:prefill.close()
            if args.draft_execution=='batch':provider.close()
            target.close()


if __name__=='__main__':main()
