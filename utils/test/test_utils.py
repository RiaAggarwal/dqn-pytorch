import unittest

from utils.utils import get_multi_index_history_df, get_rewards_history_df, get_experiments

experiments = ['baseline-1', 'snell-4', 'snell-5']

class TestUtils(unittest.TestCase):
    @unittest.skip
    def test_get_history_dataframes(self):
        df = get_multi_index_history_df(experiments)

    def test_get_rewards_history_df_columns_are_experiments(self):
        df = get_rewards_history_df(experiments)
        self.assertEqual(experiments, list(df.columns))

    def test_get_experiments_returns_a_list_of_dicts_with_keys_label_and_value(self):
        exp = get_experiments()
        self.assertIsInstance(exp, list)

        for e in exp:
            self.assertIsInstance(e, dict)
            self.assertEqual(['label', 'value'], list(e.keys()))


if __name__ == '__main__':
    unittest.main()
