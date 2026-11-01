import math
import unittest
import numpy as np
from src.HMMEngine import HMM, baum_welch, left_right_model, classify, project_2d
from src.FeatureExtractor import (extract_2d, extract_3d, map_2d_to_sphere,
                                  tangent_deltas, quantize_direction, to_spherical)


class TestNativeMath(unittest.TestCase):
    def setUp(self):
        self.hmm = HMM([[.7,.3],[.4,.6]],[[.6,.4],[.3,.7]],[.8,.2])

    def test_forward_bruteforce(self):
        obs = [0,1]
        likelihood = sum(self.hmm.pi[i]*self.hmm.B[i,obs[0]]*self.hmm.A[i,j]*self.hmm.B[j,obs[1]]
                         for i in range(2) for j in range(2))
        alpha, scale, ll = self.hmm.forward(obs)
        self.assertAlmostEqual(math.exp(ll), likelihood)
        self.assertTrue(np.allclose(alpha.sum(axis=1), 1.))

    def test_backward_and_posterior(self):
        obs = [0,1,0]
        alpha,scale,_ = self.hmm.forward(obs)
        beta = self.hmm.backward(obs,scale)
        self.assertTrue(np.allclose((alpha*beta).sum(axis=1), np.ones(3)))
        self.assertTrue(np.allclose(self.hmm.posterior(obs).sum(axis=1),1.))

    def test_viterbi_matches_bruteforce(self):
        obs = [0,1,0]
        p_best, score = self.hmm.viterbi(obs)
        from itertools import product
        candidates = []
        for q in product(range(2),repeat=3):
            p = self.hmm.pi[q[0]]*self.hmm.B[q[0],obs[0]]
            for t in range(1,3):
                p *= self.hmm.A[q[t-1],q[t]]*self.hmm.B[q[t],obs[t]]
            candidates.append((p,list(q)))
        mx, best = max(candidates)
        self.assertEqual(p_best,best)
        self.assertAlmostEqual(score,math.log(mx))

    def test_em_maintains_normalization_and_topology(self):
        model = left_right_model(3,10,seed=5,label='a')
        original_mask = model.A==0
        ll = baum_welch(model, [[0,0,1,2,1,2,8], [1,0,0,3,2,8]],epochs=3)
        self.assertEqual(len(ll),3)
        self.assertTrue(np.allclose(model.A.sum(axis=1),1.))
        self.assertTrue(np.allclose(model.B.sum(axis=1),1.))
        self.assertTrue(np.all(model.A[original_mask]==0))

    def test_posteriors_sum_one(self):
        other = HMM([[.7,.3],[.4,.6]],[[.2,.8],[.8,.2]],[.8,.2])
        ranking = classify({'a':self.hmm,'b':other},[0,1,0])
        self.assertAlmostEqual(sum(p for _,p,_ in ranking),1.)

    def test_projection_is_not_pen_jump(self):
        model=left_right_model(2,10,seed=1)
        model.median_length=15
        projected=project_2d(model,[0,1],(400.,500.))
        self.assertIsInstance(projected,list)

    def test_lift_and_dot(self):
        strokes=[[[500,500,0],[520,500,6]],[[500,200,100]]]
        obs=extract_2d(strokes,step_units=5.)
        self.assertEqual(obs[-2:], [8,9])
        self.assertTrue(all(o==0 for o in obs[:-2]))

    def test_up_direction_from_inverted_canvas(self):
        self.assertEqual(extract_2d([[[100,400,0],[100,380,5]]],step_units=10.),[2,2])

    def test_3d_sphere_math(self):
        strokes=[[[400,500,0],[420,500,10],[440,500,20]]]
        s3=map_2d_to_sphere(strokes)
        symbols=extract_3d(s3,step_units=.018)
        self.assertTrue(symbols)
        self.assertTrue(all(0<=s<8 for s in symbols))
        self.assertAlmostEqual(float(np.linalg.norm(s3[0][0])),1.)
        th,ph=to_spherical(s3[0][0])
        self.assertAlmostEqual(ph,0.)

if __name__=='__main__':
    unittest.main()
