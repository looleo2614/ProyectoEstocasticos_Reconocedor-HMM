import unittest
from scripts.evaluate_realtime import truncate_sample
from src.HMMEngine import classify, left_right_model

class TestRealtime(unittest.TestCase):
    def test_prefix_preserves_strokes(self):
        sample = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]]
        self.assertEqual(truncate_sample(sample, .60), [sample[0]])
        self.assertEqual(truncate_sample(sample, 1.), sample)
    def test_empty_prior_uniform(self):
        models = {ch: left_right_model(2, 10, seed=i) for i, ch in enumerate('abc')}
        self.assertEqual([round(p, 9) for _, p, _ in classify(models, [])], [round(1/3, 9)]*3)

if __name__ == '__main__':
    unittest.main()
