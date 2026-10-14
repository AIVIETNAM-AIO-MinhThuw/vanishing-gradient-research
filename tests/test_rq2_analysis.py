import unittest


class RQ2ClassificationTests(unittest.TestCase):
    def test_threshold_free_coexistence_requires_both_relations(self) -> None:
        cases = [
            (0.7, 0.1, 0.2, True),
            (0.1, 0.1, 0.2, False),
            (0.7, 0.1, -0.2, False),
        ]
        for accuracy, initial, gap, expected in cases:
            self.assertEqual((accuracy > initial) and (gap > 0), expected)

    def test_gap_sign_matches_first_vs_last_relation(self) -> None:
        import math

        for first, last in ((0.1, 0.2), (0.2, 0.1), (0.1, 0.1)):
            gap = math.log10(last / first)
            self.assertEqual(gap > 0, first < last)


if __name__ == "__main__":
    unittest.main()
