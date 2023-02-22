import json
import unittest
from unittest.mock import Mock, patch

from ao3_disco_ranking.query.graph_ranker import GraphRanker
from ao3_disco_ranking.query.query_handler import QueryHandler
from ao3_disco_ranking.query.tags_filter import TagsFilter


def get_works():
    return {
        "1": {
            "tags": {
                "fandom": ["Merlin (TV)"],
                "rating": ["Teen And Up Audiences"],
                "category": ["F/M"],
                "freeform": ["Alternate Universe - Canon Divergence"],
                "character": ["Guinevere"],
            },
            "work_id": "1",
        },
        "2": {
            "tags": {
                "fandom": ["Merlin (TV)"],
                "rating": ["Teen And Up Audiences"],
                "category": ["F/M"],
                "freeform": ["Alternate Universe - Canon Divergence"],
                "character": ["Guinevere"],
            },
            "work_id": "2",
        },
        "3": {
            "tags": {
                "fandom": ["Merlin (TV)"],
                "rating": ["Teen And Up Audiences"],
                "category": ["F/M"],
                "freeform": ["Alternate Universe - Canon Divergence"],
                "character": ["Guinevere"],
            },
            "work_id": "3",
        },
    }


class TestQueryHandler(unittest.TestCase):
    @patch("ao3_disco_ranking.query.query_handler.get_work_jsons")
    def test_basic_query(self, jsons_mock):
        collections = [("1", "2", "3"), (), ("1", "3", "4"), ("1"), ("2", "3")]
        graph = GraphRanker(collections)

        works = get_works()
        filter = TagsFilter(works)

        embedding_ranker = Mock()
        embedding_ranker.rank = Mock(return_value="result")

        jsons_mock.return_value = [get_works()["1"]]

        query = QueryHandler(filter, graph, embedding_ranker)
        assert query.basic_query("1") == "result"
