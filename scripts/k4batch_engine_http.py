"""External stdlib-only checks against an uninstrumented unified HTTP service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import socket
from threading import Barrier, Event, Thread
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-client',type=Path)
    parser.add_argument('--expected-prefill-rows',type=int)
    parser.add_argument('--expected-capacity',type=int,default=32768)
    args=parser.parse_args()
    inputs=json.loads(args.inputs.read_text());reference=json.loads(args.reference.read_text())
    assert reference['status'] in ('PASS_ENGINE_V02_LONG','PASS_ENGINE_CONTEXT')
    report=dict(status='RUNNING',client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),runs=[])
    if args.resume_client is not None:
        previous=json.loads(args.resume_client.read_text())
        assert previous['status']=='FAIL_ENGINE_V02_HTTP'
        assert (previous['error']=="AssertionError: {'active': 4, 'streams': 8, 'pending': 7}" or
            previous['error'].startswith('AssertionError: (429,') and 'request admission timed out' in previous['error'])
        assert len(previous['runs'])==1 and previous['runs'][0]['name']=='four-long-uninstrumented'
        assert previous['input_sha256']==report['input_sha256']
        assert previous['reference_sha256']==report['reference_sha256']
        failures=list(previous.get('retained_client_failures',[]))
        if previous.get('retained_client_failure'):failures.append(previous['retained_client_failure'])
        failures.append(dict(status=previous['status'],error=previous['error'],
                client_sha256=previous['client_sha256'],
                raw_sha256=hashlib.sha256(args.resume_client.read_bytes()).hexdigest()))
        report.update(runs=previous['runs'],before=previous['before'],retained_client_failures=failures)

    def save(phase):
        report['phase']=phase
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)

    def get(path='/health',body=None):
        connection=http.client.HTTPConnection('127.0.0.1',18085,timeout=300)
        try:
            connection.request('GET' if body is None else 'POST',path,
                None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
            response=connection.getresponse();return response.status,json.loads(response.read())
        finally:connection.close()

    def open_request(item,stream=True,limit=64):
        connection=http.client.HTTPConnection('127.0.0.1',18085,timeout=300)
        body=dict(messages=item['messages'],max_tokens=limit,stream=stream)
        if stream:body['stream_options']=dict(include_usage=True)
        started=time.perf_counter()
        connection.request('POST','/v1/chat/completions',json.dumps(body,ensure_ascii=False).encode(),
                           {'Content-Type':'application/json'})
        response=connection.getresponse()
        assert response.status==200,(response.status,response.read())
        return connection,response,started

    def consume(opened,stream=True):
        connection,response,started=opened
        try:
            key=response.getheader('X-Request-Id');text='';usage=None;first=None;last=None;reason=None
            if stream:
                done=False
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':done=True;break
                    value=json.loads(data);assert 'error' not in value,value
                    for choice in value['choices']:
                        delta=choice['delta'].get('content','')
                        if delta:
                            last=time.perf_counter();first=last if first is None else first;text+=delta
                        if choice['finish_reason'] is not None:reason=choice['finish_reason']
                    if value.get('usage') is not None:usage=value['usage']
                assert done and usage is not None
            else:
                value=json.loads(response.read());key=value['id'].removeprefix('chatcmpl-')
                text=value['choices'][0]['message']['content'];usage=value['usage']
                reason=value['choices'][0]['finish_reason']
            return dict(request_id=key,text=text,usage=usage,finish=reason,
                client_seconds=time.perf_counter()-started,stream=stream,
                first_content_seconds=None if first is None else first-started,
                content_interval_seconds=None if first is None else last-first)
        finally:connection.close()

    def generate(item,stream=True,barrier=None,limit=64):
        if barrier is not None:barrier.wait(10)
        return consume(open_request(item,stream,limit),stream)

    def idle():
        end=time.monotonic()+60
        while time.monotonic()<end:
            status,health=get()
            assert status==200 and health['status']=='ready',health
            if health['active_requests']==health['open_streams']==health['pending_submissions']==0:return health
            time.sleep(.02)
        raise AssertionError('service did not return idle')

    def cohort(name,items,*,overflow=False):
        save(name);start=time.perf_counter();barrier=Barrier(len(items));stop=Event();saturated=Event()
        peaks=dict(active=0,streams=0,pending=0,full_running_with_four_waiting=False)
        def monitor():
            while not stop.is_set():
                _,health=get()
                peaks['active']=max(peaks['active'],health['active_requests'])
                peaks['streams']=max(peaks['streams'],health['open_streams'])
                peaks['pending']=max(peaks['pending'],health['pending_submissions'])
                if (health['active_requests']==4 and health['open_streams']==8
                        and health['pending_submissions']==4):
                    peaks['full_running_with_four_waiting']=True
                if health['open_streams']==8:saturated.set()
                stop.wait(.02)
        thread=Thread(target=monitor);thread.start()
        try:
            with ThreadPoolExecutor(len(items)) as pool:
                futures=[pool.submit(generate,item,index%2==0,barrier) for index,item in enumerate(items)]
                if overflow:
                    assert saturated.wait(20),'did not observe eight open streams'
                    status,value=get('/v1/chat/completions',dict(messages=items[0]['messages'],max_tokens=1))
                    assert status==429,(status,value)
                values=[future.result(600) for future in futures]
        finally:stop.set();thread.join(10)
        health=idle();assert peaks['active']==4,peaks
        result=dict(name=name,requests=values,wall_seconds=time.perf_counter()-start,observed_peaks=peaks,health=health)
        if overflow:
            assert peaks['streams']==8 and peaks['full_running_with_four_waiting'],peaks
            for index in range(4):
                assert values[index]['text']==values[index+4]['text'],'same prompt changed after queue/slot reuse'
                assert values[index]['usage']==values[index+4]['usage']
            result['overflow_429']=True
        report['runs'].append(result);save(name+':done');return result

    try:
        if args.resume_client is None:report['before']=idle()
        caps=report['before']['capabilities']
        assert caps['max_running_requests']==4 and caps['context_capacity']==args.expected_capacity
        assert caps['gpu_batch'] is False and caps['target_kv_dtype']=='bf16'
        if args.expected_prefill_rows is not None:
            assert caps['prefill_chunk_rows']==args.expected_prefill_rows
            report['expected_prefill_rows']=args.expected_prefill_rows
        long=report['runs'][0] if args.resume_client is not None else cohort('four-long-uninstrumented',inputs['long'])
        for index,value in enumerate(long['requests']):
            gold=reference['target_only_controls'][str(index)]
            assert value['text']==gold['text'],'live output changed'
            assert value['usage']['completion_tokens']==len(gold['tokens'])
            assert value['usage']['prompt_tokens']==inputs['long'][index]['prompt_tokens']
        # Keep queue service time below the unchanged 30s admission timeout.
        # The separate 32K cohort already covers large HTTP payloads.
        queue_items=[dict(messages=[dict(role='user',content=item['messages'][0]['content'][:1024]+
            '\nExplain in detail with examples, in at least 500 words.')]) for item in inputs['short']]
        cohort('four-running-four-queued',queue_items*2,overflow=True)
        save('cancellation-survivors')
        opened=[open_request(item,limit=128) for item in inputs['short']]
        status,health=get();assert health['active_requests']==4,health
        first=opened[0][1].getheader('X-Request-Id')
        before_cancel=health['metrics']['requests_cancelled_total']
        status,value=get('/cancel',dict(request_id=first));assert status==200 and value['cancelled']
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(consume,opened))
        health=idle();assert health['metrics']['requests_cancelled_total']==before_cancel+1
        assert results[0]['usage']['completion_tokens']<128
        for index in range(1,4):
            replay=generate(inputs['short'][index],stream=False,limit=128)
            assert replay['text']==results[index]['text'],'cancellation changed survivor'
            assert replay['usage']==results[index]['usage']
        report['cancellation']=dict(actual_active=4,survivors_exact=3,cancelled_tokens=results[0]['usage']['completion_tokens'])
        # Drop a real SSE socket; metrics and an exact replay verify recovery.
        opened=open_request(queue_items[0],limit=128)
        connection,response,_=opened
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR);response.close();connection.close()
        health=idle()
        replay=generate(queue_items[0],stream=False,limit=64)
        assert replay['text']==report['runs'][1]['requests'][0]['text']
        report['disconnect_recovered']=True
        report['after']=idle();assert report['after']['metrics']['worker_failures_total']==0
        report['status']='PASS_ENGINE_V03_HTTP' if args.expected_capacity==65536 else 'PASS_ENGINE_V02_HTTP'
        save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_V02_HTTP',error=type(exc).__name__+': '+str(exc)[:3000]);save('failed');raise


if __name__=='__main__':main()
