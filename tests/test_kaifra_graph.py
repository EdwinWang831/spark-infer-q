from types import SimpleNamespace
import unittest

from spark_infer_q.kaifra_graph import GraphTarget


class GraphStateOwnership(unittest.TestCase):
    def target(self):
        class State:
            resets=0
            def reset(self,w):self.resets+=1
        self.fences=0
        def sync():self.fences+=1
        return SimpleNamespace(buffers={i:None for i in range(1,9)},w=object(),
            state=State,torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)))

    def test_static_lease_reuse_and_foreign_state_rejected(self):
        target=GraphTarget(self.target());first=target.state()
        with self.assertRaisesRegex(RuntimeError,'already leased'):target.state()
        with self.assertRaisesRegex(RuntimeError,'request is active'):target.warmup()
        with self.assertRaisesRegex(RuntimeError,'active request'):target.close()
        with self.assertRaisesRegex(RuntimeError,'foreign'):target.release_state(object())
        target.release_state(first)
        with self.assertRaisesRegex(RuntimeError,'released'):target.release_state(first)
        second=target.state();self.assertIs(first,second);self.assertEqual(first.resets,1)
        target.release_state(second);target.close()
        self.assertFalse(target.capabilities['cuda_graph'])
        with self.assertRaisesRegex(RuntimeError,'closed'):target.state()

    def test_bucket_configuration_and_unleased_execution_rejected(self):
        for widths in ((),(0,),(9,),(1,1),(True,),(1.5,)):
            with self.subTest(widths=widths),self.assertRaises(ValueError):GraphTarget(self.target(),widths=widths)
        target=GraphTarget(self.target())
        with self.assertRaisesRegex(RuntimeError,'active request'):target.run(object(),[1])


if __name__=='__main__':unittest.main()
