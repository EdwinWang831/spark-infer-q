"""Check overlapping CPU/GPU intervals before publishing a time distribution."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e9_analyze import analyze,duration,intersection_duration,RangeIndex


class ProfileAccountingTests(unittest.TestCase):
    def test_union_clipping_and_overlap(self):
        self.assertEqual(duration([(0,10),(5,20),(20,30)],3,25),22)
        self.assertEqual(intersection_duration([(0,10),(5,20)],[(8,12),(15,30)]),9)
        with self.assertRaises(AssertionError):RangeIndex([dict(start=0,end=10,name='a'),dict(start=9,end=11,name='b')])

    def test_nested_apis_and_overlapping_kernels_are_not_added_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite';c=sqlite3.connect(path)
            c.executescript('''
CREATE TABLE StringIds(id INTEGER,value TEXT);
INSERT INTO StringIds VALUES(1,'cudaLaunchKernel'),(2,'cuLaunchKernel'),(3,'cudaDeviceSynchronize'),(4,'kernelA'),(5,'kernelB');
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,textId INTEGER);
INSERT INTO NVTX_EVENTS VALUES(100,300,'e9.decode.fixture',99,NULL),(110,160,'e9.phase.draft',99,NULL),(160,280,'e9.phase.verify',99,NULL);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(110,130,99,1,1),(175,185,99,2,1),(190,260,99,3,3);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(115,125,99,1,2);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,correlationId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(120,200,4,1),(180,230,5,2);
''');c.commit();c.close()
            r=analyze(path)
        self.assertEqual(r['gpu']['kernel_busy_percent'],55)
        self.assertAlmostEqual(r['gpu']['kernel_duration_sum_ms'],130/1e6)
        self.assertAlmostEqual(r['cpu_cuda']['main_thread_api_union_ms'],100/1e6)
        self.assertAlmostEqual(r['cpu_cuda']['sync_overlap_gpu_activity_ms'],40/1e6)
        self.assertEqual(r['gpu']['correlated_kernel_calls'],2)
        self.assertAlmostEqual(r['unlabeled_host_ms'],30/1e6)
        self.assertAlmostEqual(r['gpu']['kernel_by_launch_phase']['phase']['e9.phase.draft'],80/1e6)
        self.assertNotIn('/tmp/',json.dumps(r))


if __name__=='__main__':unittest.main()
