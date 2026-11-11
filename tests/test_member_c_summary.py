"""Guard against missing seeds, population SD and best-seed selection."""

import unittest
from member_c.summarize_multiseed import aggregate, SEEDS


class TestMultiSeedSummary(unittest.TestCase):
    def rows(self):
        return [dict(dataset=d,hidden_sizes=w,seed=s,validation_accuracy=v,
                     validation_loss=0.3,train_accuracy=1.0,gap_percentage_points=100*(1-v),
                     parameter_count=100 if w=='128x128' else 200,
                     training_seconds=2.0,run_elapsed_seconds=3.0)
                for d in ('20','30','40') for w in ('128x128','256x256')
                for s,v in zip(SEEDS,(0.8,0.9,1.0))]

    def test_sample_standard_deviation_and_complete_seed_accounting(self):
        groups,paired,_=aggregate(self.rows())
        self.assertEqual(len(groups),6)
        self.assertAlmostEqual(groups[0]['mean_validation_accuracy'],0.9)
        self.assertAlmostEqual(groups[0]['sample_std_validation_accuracy'],0.1)
        self.assertEqual(groups[0]['total_training_seconds'],6)
        self.assertAlmostEqual(paired[0]['mean_difference_pp'],0)
        self.assertAlmostEqual(paired[0]['sample_std_difference_pp'],0)

    def test_missing_duplicate_or_unexpected_seeds_rejected(self):
        rows=self.rows()
        for invalid in (rows[:-1],rows+[rows[0]],rows[:-1]+[dict(rows[-1],seed=999)]):
            with self.assertRaises(ValueError):aggregate(invalid)

    def test_candidate_is_based_on_mean_not_best_seed(self):
        rows=self.rows()
        for r in rows:
            r['validation_accuracy']=([0.99,0.7,0.7][SEEDS.index(r['seed'])] if r['hidden_sizes']=='128x128' else 0.85)
        _,_,candidates=aggregate(rows)
        self.assertTrue(all(c['hidden_sizes']=='256x256' for c in candidates))


if __name__=='__main__':unittest.main()
