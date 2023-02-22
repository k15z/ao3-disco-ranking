import os
import tempfile
import unittest

from ao3_disco_ranking.models.base import BaseModel


class BaseModelTest(unittest.TestCase):
    def test_base_model(self):
        model = BaseModel()
        with tempfile.TemporaryDirectory() as tempdir:
            path_to_jsonl = os.path.join(tempdir, "dataset.jsonl")
            with open(path_to_jsonl, "w") as tmpfile:
                tmpfile.write('{"work": "a", "candidates": {"b": 1, "c": 2, "d": 3}}\n')

            model.fit(path_to_jsonl)
            with self.assertRaises(NotImplementedError):
                model.embedding("dummy")
            candidate_score = model.rank("hello", ["a", "b", "c"], {})
            candidates, _ = zip(*candidate_score)
            assert set(candidates) == set(["a", "b", "c"])
