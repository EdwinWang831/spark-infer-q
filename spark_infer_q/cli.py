"""Public command line; help and environment checks never import GPU libraries."""
import argparse
import json
import os
from pathlib import Path


VERSION = '0.3.0'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='kaifra', description='Local adaptive inference engine')
    parser.add_argument('--version', action='version', version=VERSION)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('doctor', 'serve'):
        child = commands.add_parser(command)
        child.add_argument('--root', required=True, type=Path, help='prepared runtime directory')
        child.add_argument('--checkpoint', required=True, type=Path, help='existing local target checkpoint')
        if command == 'serve':
            child.add_argument('--port', type=int, default=8000)
            child.add_argument('--model', default='spark-infer-q')
            child.add_argument('--capacity', type=int, default=65536,
                               help='per-request input plus output tokens (maximum 65536)')
            child.add_argument('--max-requests', type=int, default=4,
                               help='running request slots, separate from the waiting queue')
            child.add_argument('--mode', choices=('adaptive', 'none'), default='adaptive')
            child.add_argument('--execution', choices=('graph', 'eager'), default='graph')
            child.add_argument('--queue-size', type=int, default=4)
            child.add_argument('--admission-timeout', type=float, default=30.)
            child.add_argument('--prefill-chunk-rows', type=int, choices=(128, 256, 512, 1024))
            child.add_argument('--prefill-experts', type=Path,
                               help='optional pinned E44 expert extension; selects 1024-row prefill by default')
    args = parser.parse_args(argv)
    from .engine_server import ServiceConfig, inspect_environment
    settings = {k: v for k, v in vars(args).items() if k != 'command'}
    try:
        config = ServiceConfig(**settings)
    except ValueError as exc:
        parser.error(str(exc))
    report = inspect_environment(config)
    if args.command == 'doctor' or not report['ready']:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['ready']:
        return 1
    if args.command == 'doctor':
        return 0
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '0') != '0':
        parser.error('the validated profile requires TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0')
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    import torch
    from .model_loader import load_session
    from .engine_server import open_service, run_service
    with torch.inference_mode():
        session = load_session(config.root, config.checkpoint,
                               lambda item: print(json.dumps(item), flush=True))
    with open_service(session, config) as service:
        run_service(service)
    return 0
