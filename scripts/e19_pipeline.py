"""Finite continuation after E17, using the same serialized resident worker."""
import argparse
import json
from pathlib import Path
import time

from e19_run_contract import CONTRACT, SOURCES, DEPENDENCIES, sha, verify_seal, write_new


def main(root):
    output = root/'artifacts/e19-pipeline.json'
    assert not output.exists()
    worker = json.loads((root/'worker-status.json').read_text())
    assert worker['status'] in ('READY', 'RUNNING') and worker['load_count'] == 1
    seal = dict(contract=CONTRACT, created_unix=time.time(), resident_pid=worker['pid'], load_count=1,
                source_sha256={n: sha(root/n) for n in SOURCES+DEPENDENCIES},
                source_test_collection_files_present=len(list((root/'artifacts').glob('e17-test-??.json'))),
                source_e17_evaluation_present=(root/'artifacts/e17-evaluation.json').exists(),
                selection_test_policy='fit stage opens only E17 train/calibration; tests may already exist but are not read until MLP selection is sealed')
    write_new(root/'artifacts/e19-source-seal.json', seal)
    report = dict(status='WAITING_FOR_E17', started_unix=time.time(), source_seal_sha256=sha(root/'artifacts/e19-source-seal.json'),
                  resident_pid=worker['pid'], load_count=1, jobs=[], contract=CONTRACT)

    def save():
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(output)

    def check():
        state = json.loads((root/'worker-status.json').read_text())
        assert state['pid'] == worker['pid'] and state['load_count'] == 1
        assert state['status'] in ('READY', 'RUNNING'), state
        assert sha(root/'artifacts/e19-source-seal.json') == report['source_seal_sha256']
        verify_seal(root)
        return state

    def submit(name, module, arguments, expected):
        state = check()
        # Never compete for the queue with the E17 driver or another experiment.
        while state['status'] != 'READY':
            time.sleep(5)
            state = check()
        assert not list((root/'jobs/pending').glob('*.json'))
        assert not list((root/'jobs/running').glob('*.json'))
        assert not (root/'jobs/done'/name).exists()
        job = dict(module=module, sha256=seal['source_sha256'][module+'.py'], arguments=arguments)
        path = root/'jobs/pending'/name
        temp = path.with_suffix('.tmp')
        write_new(temp, job)
        temp.replace(path)
        report.update(status='RUNNING', active_job=name)
        save()
        done = root/'jobs/done'/name
        while not done.exists():
            check()
            time.sleep(5)
        result = json.loads(done.read_text())
        assert result['status'] == 'PASS', result
        assert result['worker_pid'] == worker['pid'] and result['load_count'] == 1
        assert result['result']['status'] == expected, result
        report['jobs'].append(dict(name=name, request=job, result=result))
        save()
        return result

    save()
    try:
        while True:
            check()
            path = root/'artifacts/e17-pipeline.json'
            prior = json.loads(path.read_text())
            if prior['status'] in ('PASS_E17_CHAIN', 'COMPLETE_OFFLINE_GATE_FAILED'):
                report.update(e17_pipeline_sha256=sha(path), e17_terminal_status=prior['status'])
                break
            assert prior['status'] == 'RUNNING', prior.get('error', prior['status'])
            time.sleep(15)
        submit('061-e19-smoke.json', 'e19_train', dict(stage='smoke'), 'PASS_E19_SMOKE')
        submit('062-e19-fit.json', 'e19_train', dict(stage='fit'), 'PASS_E19_FIT')
        result = submit('063-e19-evaluation.json', 'e19_train', dict(stage='evaluate'), 'PASS_E19_EVALUATION')
        if result['result']['gate_pass']:
            submit('064-e19-online.json', 'e19_online', {}, 'PASS_E19_ONLINE')
            report['status'] = 'PASS_E19_CHAIN'
        else:
            report.update(status='COMPLETE_OFFLINE_GATE_FAILED', online='NOT_RUN')
        state = check()
        for _ in range(6):
            if state['status'] == 'READY':
                break
            time.sleep(1)
            state = check()
        report.update(worker_after=state, finished_unix=time.time())
        report.pop('active_job', None)
        save()
    except Exception as exc:
        report.update(status='STOPPED_ON_FAILURE', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
