from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from spark_infer_q.worker_cpu import worker_cpu


class WorkerCPU(unittest.TestCase):
    def fixture(self,root):
        for cpu,freq in ((1,2808000),(5,3900000),(11,2808000),(15,3900000)):
            p=Path(root)/('cpu'+str(cpu))/'cpufreq/cpuinfo_max_freq';p.parent.mkdir(parents=True);p.write_text(str(freq))
        calls=[];system=NS(sched_getaffinity=lambda pid:{1,5,11,15},sched_setaffinity=lambda pid,cpus:calls.append((pid,set(cpus))))
        return system,calls

    def test_only_allowed_fastest_cpus_and_calling_thread_are_selected_and_restored(self):
        with tempfile.TemporaryDirectory() as root:
            system,calls=self.fixture(root)
            with worker_cpu('performance',system=system,frequency_root=root) as metadata:
                self.assertEqual(metadata['worker_cpu_affinity'],[5,15]);self.assertEqual(calls,[(0,{5,15})])
            self.assertEqual(calls[-1],(0,{1,5,11,15}))

    def test_failure_in_owner_body_restores_previous_mask(self):
        with tempfile.TemporaryDirectory() as root:
            system,calls=self.fixture(root)
            with self.assertRaisesRegex(RuntimeError,'owner failed'):
                with worker_cpu('performance',system=system,frequency_root=root):raise RuntimeError('owner failed')
            self.assertEqual(calls,[(0,{5,15}),(0,{1,5,11,15})])

    def test_default_needs_no_linux_api_and_invalid_topology_never_changes_affinity(self):
        with worker_cpu(system=NS()) as metadata:self.assertEqual(metadata,{'worker_cpu_policy':'default'})
        with self.assertRaises(RuntimeError):
            with worker_cpu('performance',system=NS()):pass
        with tempfile.TemporaryDirectory() as root:
            system,calls=self.fixture(root);(Path(root)/'cpu5/cpufreq/cpuinfo_max_freq').unlink()
            with self.assertRaisesRegex(RuntimeError,'cannot read'):
                with worker_cpu('performance',system=system,frequency_root=root):pass
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
