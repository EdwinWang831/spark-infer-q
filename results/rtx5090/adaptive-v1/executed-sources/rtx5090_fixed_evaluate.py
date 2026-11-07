"""Grade frozen outputs, executing HumanEval only inside the macOS OS sandbox."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import platform
import re
import string
import subprocess
import sys
import tempfile


SANDBOX_PROFILE = '''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(deny file-read-data (subpath "/Users") (subpath "/Volumes") (subpath "/private") (subpath "/tmp"))
(deny mach-lookup)
(deny process-fork)
'''
CODE_WORKER = '''import json, random, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
item=json.load(sys.stdin)
namespace={}
random.seed(0)
exec(compile(item['code'], '<candidate>', 'exec'), namespace)
random.seed(0)
exec(compile(item['test'], '<official-test>', 'exec'), namespace)
namespace['check'](namespace[item['entry_point']])
print('__FIXED_V1_TESTS_PASSED__')
'''


def isolated_python(code, payload=''):
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('code evaluation requires the verified macOS sandbox; no unsandboxed fallback')
    return subprocess.run(
        ['/usr/bin/sandbox-exec', '-p', SANDBOX_PROFILE, sys.executable, '-I', '-S', '-c', code],
        input=payload, text=True, capture_output=True, timeout=5, cwd='/', env={'PATH': '/usr/bin:/bin'},
    )


def verify_sandbox():
    with tempfile.TemporaryDirectory() as directory:
        canary = Path(directory) / 'private-canary'
        canary.write_text('not-for-candidate')
        attempts = {
            'private_file_read': f'open({str(canary)!r}).read()',
            'file_write': f'open({str(canary.with_name("write-canary"))!r}, "w")',
            'network_connect': 'socket.socket().connect(("127.0.0.1", 9))',
            'child_process': 'subprocess.run(["/usr/bin/true"], check=True)',
        }
        results = {}
        for name, attempt in attempts.items():
            code = 'import socket,subprocess\ntry:\n    ' + attempt + '\nexcept PermissionError:\n    print("DENIED")\n'
            result = isolated_python(code)
            results[name] = result.returncode == 0 and result.stdout.strip() == 'DENIED'
        if not all(results.values()):
            raise RuntimeError(f'sandbox boundary check failed: {results}')
    return results


def extract_code(text):
    fenced = re.search(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    return fenced.group(1).strip() if fenced else text.strip()


def grade_code(task, text, canonical=False):
    code = task['original_prompt'] + task['canonical_solution'] if canonical else extract_code(text)
    item = {'code': code, 'test': task['test'], 'entry_point': task['entry_point']}
    try:
        result = isolated_python(CODE_WORKER, json.dumps(item))
        passed = result.returncode == 0 and result.stdout.rstrip().endswith('__FIXED_V1_TESTS_PASSED__')
        return {'pass': passed, 'exit_code': result.returncode,
                'error': result.stderr[-1600:] if not passed else None,
                'code_sha256': hashlib.sha256(code.encode()).hexdigest()}
    except subprocess.TimeoutExpired:
        return {'pass': False, 'error': 'execution timeout'}


def grade_math(expected, text):
    number = r'([-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
    strict = re.findall(r'####\s*\$?\s*' + number, text)
    # The frozen prompt literally says "#### <number>". Preserve the strict
    # score, but do not call a correct numeric answer wrong for retaining < >.
    values = re.findall(r'####\s*(?:<\s*)?\$?\s*' + number, text)
    extracted = values[-1].replace(',', '') if values else None
    try:
        passed = extracted is not None and Decimal(extracted) == Decimal(expected.replace(',', ''))
        strict_pass = bool(strict) and Decimal(strict[-1].replace(',', '')) == Decimal(expected.replace(',', ''))
    except InvalidOperation:
        passed = strict_pass = False
    return {'pass': passed, 'strict_pass': strict_pass, 'extracted': extracted,
            'literal_placeholder_brackets': bool(values) and not strict, 'missing_answer_marker': not values}


def normalize_answer(text):
    text = ''.join(character for character in text.lower() if character not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def grade_qa(answers, text):
    prediction = normalize_answer(text)
    scores = []
    for answer in answers:
        gold = normalize_answer(answer)
        left, right = prediction.split(), gold.split()
        common = sum((Counter(left) & Counter(right)).values())
        f1 = 2 * common / (len(left)+len(right)) if left or right else 1.0
        scores.append((prediction == gold, f1))
    return {'pass': any(exact for exact, _ in scores), 'f1': max(f1 for _, f1 in scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    manifest = json.loads(args.manifest.read_text())
    report = {'python': sys.version, 'platform': platform.platform(),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'code_test_random_seed': 0,
              'math_format_note': 'literal placeholder angle brackets normalized; strict score retained; correction made after first-run format audit',
              'sandbox_profile': SANDBOX_PROFILE, 'sandbox_checks': verify_sandbox(),
              'canonical_checks': [], 'tasks': []}
    for task in manifest['tasks']:
        if task['dataset'] == 'humaneval':
            result = grade_code(task, '', canonical=True)
            report['canonical_checks'].append({'id': task['id'], **result})
            if not result['pass']:
                raise RuntimeError(f'canonical solution failed evaluation: {task["id"]}: {result}')
    if args.run:
        measured = json.loads(args.run.read_text())
        if measured['status'] != 'COMPLETE_MEASURED' or measured['manifest_sha256'] != report['manifest_sha256']:
            raise ValueError('run is incomplete or uses a different task manifest')
        report['run_sha256'] = hashlib.sha256(args.run.read_bytes()).hexdigest()
        by_id = {task['id']: task for task in manifest['tasks']}
        for row in measured['tasks']:
            task = by_id[row['id']]
            grades = {}
            for width, candidate in row['candidates'].items():
                text = candidate['text']
                if task['dataset'] == 'humaneval':
                    grade = grade_code(task, text)
                elif task['dataset'] == 'gsm8k':
                    grade = grade_math(task['expected'], text)
                else:
                    grade = grade_qa(task['answers'], text)
                grades[width] = grade
            report['tasks'].append({'id': task['id'], 'dataset': task['dataset'],
                                    'split': task['split'], 'grades': grades})
    report['status'] = 'COMPLETE'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sandbox': report['sandbox_checks'], 'canonical_pass': len(report['canonical_checks']),
                      'graded_tasks': len(report['tasks'])}))


if __name__ == '__main__':
    main()
