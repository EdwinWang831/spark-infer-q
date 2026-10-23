"""Serve one K2 mode using pinned Target/Q4 DFlash2 loaded once."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mode',choices=('none','lagged','confidence','adaptive'),default='adaptive')
    parser.add_argument('--profile',choices=('baseline','e33'),default='baseline')
    parser.add_argument('--execution',choices=('eager','graph'),default='eager')
    parser.add_argument('--capacity',type=int,default=256)
    parser.add_argument('--port',type=int,default=18083)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--cost-bucket',choices=('short','long'),default='short',
                        help='confidence cost bucket; adaptive selects by context length')
    args=parser.parse_args();root=args.root.resolve()
    if args.execution=='graph' and args.mode=='none':
        parser.error('graph execution currently requires speculative dynamic mode')
    if args.profile=='e33' and (args.mode!='adaptive' or args.execution!='graph' or args.model is not None):
        parser.error('e33 profile requires adaptive + graph and its frozen model')
    checkout=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(checkout),str(checkout/'scripts'),str(root/'followup-deps'),str(root/'python-deps')]
    from spark_infer_q.native_scheduler import NativeScheduler
    from spark_infer_q.kaifra_runtime import KaifraEngine
    from spark_infer_q.kaifra_speculative import session_provider,speculative_engine
    from spark_infer_q.kaifra_http import make_server
    with NativeScheduler(root.parent/'DeepInfer',root.parent/'build-release/lib/deepinfer',
                         max_drafts=0 if args.mode=='none' else 7) as native:
        from e12_session import load_session
        with native.torch.inference_mode():
            session=load_session(root,args.checkpoint,lambda x:print(json.dumps(x),flush=True))
        with ExitStack() as resources:
            metadata={};target=session.target
            if args.profile=='e33':
                from spark_infer_q.kaifra_optimized import optimized_provider,optimized_target
                with native.torch.inference_mode():
                    provider,metadata=optimized_provider(session)
                    target=optimized_target(session)
                resources.callback(target.close)
                target.warmup();curve=None
            else:
                if args.execution=='graph':
                    from spark_infer_q.kaifra_graph import GraphTarget
                    target=GraphTarget(target)
                    resources.callback(target.close)
                    target.warmup()
                if args.mode!='none':
                    provider,curve,metadata=session_provider(session,args.mode,model_path=args.model,cost_bucket=args.cost_bucket)
            if args.mode=='none':
                engine=KaifraEngine.from_native(native,session.target,eos_ids=session.eos,
                    tokenizer=session.tokenizer,capacity=args.capacity,max_requests=1)
            else:
                engine=speculative_engine(native,target,provider,eos_ids=session.eos,
                    prior=session.calibration['conditional_prior'],mode=args.mode,capacity=args.capacity,
                    tokenizer=session.tokenizer,curve=curve)
            with engine:
                server=make_server(engine,('127.0.0.1',args.port))
                print(json.dumps(dict(status='READY',mode=args.mode,profile=args.profile,
                    capabilities=engine.capabilities,metadata=metadata)),flush=True)
                try:server.serve_forever()
                except KeyboardInterrupt:pass
                finally:server.server_close()



if __name__=='__main__':main()
