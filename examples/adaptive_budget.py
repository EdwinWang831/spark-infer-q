"""Run the released E17 predictor and budget rule on synthetic CPU inputs.

From the repository root: python3 -m examples.adaptive_budget
This is a decision walkthrough, not model generation or a benchmark.
"""

import json
from pathlib import Path

from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.local_policy import ProgressModel


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'results/e17/completed/e17-selected-model.json'
    artifact = json.loads(path.read_text())
    metadata = artifact['metadata']
    model = ProgressModel(metadata)

    # Historical short-context costs, not timings from the machine running this example.
    # All widths pay the complete draft and predictor cost, including one verification row.
    bucket = artifact['costs']['short']
    costs = [value + bucket['feature_draft_ms'] + artifact['cpu_predict_ms']
             for value in bucket['verify_commit_ms']]

    low = list(metadata['mean'])
    low[:21] = [0.] * 21
    higher = list(metadata['mean'])
    # F1 starts with seven triples: edge margin, unary margin, selected-token deficit.
    # Change only the first two margins; keep request/hidden features at their training mean.
    for index in range(0, 21, 3):
        higher[index] += metadata['scale'][index]
        higher[index + 1] += metadata['scale'][index + 1]

    print('E17 predictor + existing budget rule; standard-library CPU walkthrough')
    print('Synthetic features, NOT real DFlash traces or an accuracy/throughput test.')
    print('Costs: frozen E17 short-context calibration, not current-machine timings.')
    print('Rows include one pending anchor plus a draft prefix; drafts were already computed.')
    print()
    print(f'{"scenario":<19} {"rows":>4} {"drafts verified":>15} '
          f'{"expected tokens":>16} {"estimated ms":>13}')
    for label, features in [('zero_margins', low), ('mean_features', metadata['mean']),
                            ('larger_margins', higher)]:
        rates = model.predict(features)
        rows = choose_rows(rates, costs)
        expected = 1. + sum(rates[:rows - 1])
        print(f'{label:<19} {rows:>4} {rows - 1:>15} '
              f'{expected:>16.3f} {costs[rows - 1]:>13.3f}')


if __name__ == '__main__':
    main()
