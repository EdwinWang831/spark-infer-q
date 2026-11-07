"""Freeze 36 public tasks and the protocol before any target inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request


SEED = 'rtx5090-fixed-v1-20261106'
SOURCES = {
    'gsm8k': ('openai/grade-school-math', 'b0bb162abedc65e1fdd8e93ed090fd7598ee68bc',
              'grade_school_math/data/test.jsonl'),
    'humaneval': ('openai/human-eval', '463c980b59e818ace59f6f9803cd92c749ceae61',
                 'data/HumanEval.jsonl.gz'),
    'squad': ('rajpurkar/SQuAD-explorer', '240e165ab706d95bd4323653bb92421f446208cf',
              'dataset/dev-v1.1.json'),
}
PROTOCOL = {
    'name': 'rtx5090-fixed-v1', 'widths': [1, 4, 8, 16], 'repetitions': 3,
    'max_new_tokens': 256, 'dtype': 'bfloat16', 'enable_thinking': False, 'temperature': 0,
    'selection_seed': SEED, 'per_dataset': {'calibration': 4, 'test': 8},
    'upstream_commit': '07ebd93db9f472af339b644bb70221ad8428328a',
    'target_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
    'draft_revision': 'b74e3a329c4d963783143b1e970d95b002be72bd',
    'width1': 'upstream dflash_generate block_size=1; no draft forward; independent HF output check',
    'parity': 'width1 must equal independent HF; wider BF16 differences reported, not silently accepted as exact',
    'repeats': 'all repeats of a given task/width must produce identical token IDs',
    'selection_metric': 'minimum sum of per-task median request wall on calibration, ties use smaller width',
    'primary_metric': 'paired request wall reduction on test; output length and correctness reported jointly',
    'oracle': 'per-request best fixed width is descriptive optimistic opportunity, not a deployable policy or within-request upper bound',
    'cross_repeat_oracle': 'select width using other two repeats; evaluate held-out repeat; no policy claim',
    'quality': {'gsm8k': 'numeric final answer after ####', 'humaneval': 'official hidden tests',
                'squad': 'standard normalized EM and token F1'},
    'limitations': ['36 public benchmark tasks, not representative traffic or paper-sized evaluation',
                    'SQuAD is context-grounded QA, not open-ended conversation',
                    'no serving engine, CUDA Graph, learned controller, or DFlash2 selector'],
}


def rank(value):
    return hashlib.sha256((SEED + '|' + value).encode()).hexdigest()


def build_pools(raw):
    pools = {}
    pools['gsm8k'] = [
        {'id': f'gsm8k/test/{index}', 'group': hashlib.sha256(row['question'].encode()).hexdigest(),
         'prompt': row['question'] + '\nExplain briefly and finish with "#### <number>".',
         'expected': row['answer'].split('####')[-1].strip()}
        for index, row in enumerate(raw['gsm8k'])]
    pools['humaneval'] = [
        {'id': row['task_id'], 'group': row['task_id'],
         'prompt': 'Complete the Python function below. Return the complete function and any required imports, '
                   'with no explanation.\n\n' + row['prompt'],
         'entry_point': row['entry_point'], 'test': row['test'],
         'original_prompt': row['prompt'], 'canonical_solution': row['canonical_solution']}
        for row in raw['humaneval']]
    # One question per paragraph prevents calibration/test context overlap.
    paragraphs = {}
    for article in raw['squad']['data']:
        for paragraph in article['paragraphs']:
            group = hashlib.sha256(paragraph['context'].encode()).hexdigest()
            question = min(paragraph['qas'], key=lambda row: rank(row['id']))
            paragraphs[group] = {
                'id': 'squad/' + question['id'], 'group': group,
                'prompt': 'Answer the question using only the provided passage. '
                          'Return only the shortest answer phrase, without explanation.\n\nPassage:\n'
                          + paragraph['context'] + '\n\nQuestion: ' + question['question'],
                'answers': [answer['text'] for answer in question['answers']],
            }
    pools['squad'] = list(paragraphs.values())
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.output / 'tasks.json'
    if manifest.exists():
        parser.error('manifest exists; preserve the frozen task selection')
    sources_dir = args.output / 'sources'
    sources_dir.mkdir(parents=True, exist_ok=True)
    raw, provenance = {}, {}
    for name, (repo, commit, relative) in SOURCES.items():
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{relative}'
        request = urllib.request.Request(url, headers={'User-Agent': 'deepinfer-research'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        (sources_dir / Path(relative).name).write_bytes(payload)
        provenance[name] = {'url': url, 'commit': commit, 'sha256': hashlib.sha256(payload).hexdigest()}
        if relative.endswith('.gz'):
            payload = gzip.decompress(payload)
        raw[name] = json.loads(payload) if name == 'squad' else [json.loads(line) for line in payload.splitlines()]
    pools = build_pools(raw)
    tasks = []
    for name, pool in pools.items():
        ordered = sorted(pool, key=lambda row: rank(name + '|' + row['group']))[:12]
        for index, row in enumerate(ordered):
            tasks.append({**row, 'dataset': name, 'split': 'calibration' if index < 4 else 'test'})
    tasks.sort(key=lambda row: rank(row['id'] + '|run-order'))
    record = {'protocol': PROTOCOL, 'sources': provenance, 'tasks': tasks}
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tasks': len(tasks), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      'sources': provenance}, indent=2))


if __name__ == '__main__':
    main()
