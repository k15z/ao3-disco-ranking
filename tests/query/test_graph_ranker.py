import unittest
from copy import deepcopy

from ao3_disco_ranking.query.graph_ranker import GraphRanker
from ao3_disco_ranking.query.tags_filter import TagsFilter


class TestGraphRanker(unittest.TestCase):
    def test_rank(self):
        collections = [("11111111", "00000000")]
        graph = GraphRanker(collections)
        graph.rank("11111111")
