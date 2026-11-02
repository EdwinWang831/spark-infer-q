"""Load the pinned target once and serve the K1 native-scheduler runtime."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='prepared E3/E8 dependency root')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18083)
    parser.add_argument('--capacity', type=int, default=1024)
    parser.add_argument('--prefill-rows', type=int, choices=(1,2,4,8), default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    # The current checkout owns the runtime; prepared roots supply pinned loaders
    # and dependencies. No upstream source or research worker is modified.
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(checkout), str(checkout/'scripts'),
                   str(root/'followup-deps'), str(root/'python-deps')]
    from spark_infer_q.native_scheduler import NativeScheduler
    from spark_infer_q.kaifra_runtime import KaifraEngine
    from spark_infer_q.kaifra_http import make_server
    with NativeScheduler(root.parent/'DeepInfer', root.parent/'build-release/lib/deepinfer') as native:
        from e3_target import load_target
        from e3_wide_target import WideTarget
        from transformers import AutoTokenizer
        weights, _ = load_target(root, args.checkpoint, lambda x: print(json.dumps(x), flush=True))
        target = WideTarget(weights, capacity=args.capacity)
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                                  trust_remote_code=False)
        raw = json.loads((args.checkpoint/'generation_config.json').read_text())['eos_token_id']
        eos = set(raw if isinstance(raw, list) else [raw]); eos.add(tokenizer.eos_token_id)
        with KaifraEngine.from_native(native, target, eos_ids=eos, capacity=args.capacity,
                max_requests=1, prefill_rows=args.prefill_rows, tokenizer=tokenizer) as engine:
            server = make_server(engine, ('127.0.0.1', args.port))
            print(json.dumps(dict(status='READY', port=server.server_port,
                                  capabilities=engine.capabilities)), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()


if __name__ == '__main__': main()
