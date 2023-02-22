import unittest

from ao3_disco_ranking.query.graph_ranker import GraphRanker


class TestGraphRanker(unittest.TestCase):
    def test_rank(self):
        collections = [("1", "2", "3"), (), ("1", "3", "4"), ("1"), ("2", "3")]
        graph = GraphRanker(collections)

        work_to_score = graph.rank("1")
        assert set(work_to_score) == set([("2", 1), ("3", 2), ("4", 1)])

        work_to_score = graph.rank("5")
        assert work_to_score == []

        work_to_score = graph.rank("1", candidates=["0", "1", "2", "3"])
        assert set(work_to_score) == set([("2", 1), ("3", 2)])

        work_to_score = graph.rank("1", candidates=["0", "1", "2", "3"], num_results=1)
        assert work_to_score == [("3", 2)]
