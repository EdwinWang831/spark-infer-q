"""Component summary checks; synthetic fixtures are not GPU evidence."""
import unittest
from scripts.e52_component import summary


def fixture(factor):
    cases=[]
    for name in ('en','zh','code','mixed'):
        timings=[]
        for regime in ('warm','evict_128MiB'):
            for rep in range(6):
                timings.extend([dict(regime=regime,repetition=rep,mode='original',ms=1.),
                                dict(regime=regime,repetition=rep,mode='pipeline',ms=factor)])
        cases.append(dict(name=name,timings=timings))
    return cases


class Summary(unittest.TestCase):
    def test_time_reduction_and_speed_are_distinct(self):
        r=summary(fixture(.9))['warm']
        self.assertAlmostEqual(r['time_reduction'],.1)
        self.assertAlmostEqual(r['speed_change'],1/.9-1)
        self.assertGreater(r['root_bootstrap_gain_95'][0],0)

    def test_slowdown_is_not_reported_as_saving(self):
        r=summary(fixture(1.01))['evict_128MiB']
        self.assertLess(r['time_reduction'],0)
        self.assertLess(r['root_bootstrap_gain_95'][1],0)


if __name__=='__main__':unittest.main()
