import gzip
import os
import unittest
from unittest.mock import patch

from ao3_disco_ranking.ao3.work import Work


class TestWork(unittest.TestCase):
    @patch("ao3_disco_ranking.ao3.work.get")
    def test_work(self, get):
        path_to_example = os.path.join(os.path.dirname(__file__), "30050436.gz")
        with gzip.open(path_to_example, "rt") as fin:
            get.return_value = fin.read()
        work = Work.load("30050436")
        assert work.title == "All lives are (love)stories"
