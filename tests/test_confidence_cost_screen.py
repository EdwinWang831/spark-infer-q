import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e13_analyze import cost_screen, ridge_fit


class ConfidenceCostScreenTests(unittest.TestCase):
    def test_normalization_preserves_cost_units(self):
        xs=[[n,n*n] for n in range(1,10)]
        ys=[3+2*x[0]-.1*x[1] for x in xs]
        model=ridge_fit(xs,ys)
        point=[5,25]
        value=model['coefficients'][0]+sum(c*(x-m)/s for c,x,m,s in
            zip(model['coefficients'][1:],point,model['means'],model['scales']))
        self.assertAlmostEqual(value,10.5,delta=.01)

    def test_cost_features_ignore_acceptance_outcomes_and_unobserved_tail(self):
        tasks=[]
        for index in range(4):
            steps=[]
            for j in range(16):
                width=2+j%4;ctx=40+20*index+j;union=600+11*j
                steps.append(dict(verified_rows=width,context_rows=ctx,previous_union=None if j==0 else union-11,
                    true_union=union,verify_ms=40+2*width+.01*ctx+.001*union,
                    labels=[dict(depth=1,matched=j%2,signals=[float(j%7),0.,.5,0.])]))
            tasks.append(dict(id=str(index),split='train' if index<2 else 'test',steps=steps))
        first=cost_screen(tasks)
        altered=copy.deepcopy(tasks)
        for task in altered:
            for step in task['steps']:
                step['labels'][0]['matched']=1-step['labels'][0]['matched']
                step['labels'].append(dict(depth=2,matched=0,signals=[1e6,0.,.5,0.]))
        self.assertEqual(first,cost_screen(altered))
        for task in altered:
            if task['split']=='test':
                for step in task['steps']:step['verify_ms']+=100
        self.assertEqual(first['models'],cost_screen(altered)['models'])


if __name__=='__main__':unittest.main()
