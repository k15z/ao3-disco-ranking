import os
import tempfile
import unittest
from unittest.mock import Mock

from ao3_disco_ranking.utils import score


class UtilsTest(unittest.TestCase):
    def test_score(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path_to_jsonl = os.path.join(tempdir, "dataset.jsonl")
            with open(path_to_jsonl, "w") as tmpfile:
                tmpfile.write('{"work": "a", "candidates": {"b": 1, "c": 2, "d": 3}}\n')

            model = Mock()
            model.rank = Mock(
                return_value=[
                    ("b", 1),
                    ("c", 2),
                    ("d", 3),
                ]
            )
            self.assertAlmostEqual(score(model, path_to_jsonl), 1.0)

            model = Mock()
            model.rank = Mock(
                return_value=[
                    ("b", 3),
                    ("c", 2),
                    ("d", 1),
                ]
            )
            self.assertAlmostEqual(score(model, path_to_jsonl), 0.789, places=2)
