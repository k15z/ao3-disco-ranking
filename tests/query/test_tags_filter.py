import unittest
from copy import deepcopy

from ao3_disco_ranking.query.tags_filter import TagsFilter


def get_work():
    return {
        "tags": {
            "rating": ["Teen And Up Audiences", "test"],
            "character": ["Guinevere"],
            "relationship": ["Guinevere/Arthur Pendragon"],
        },
        "title": "Two Hearts",
        "authors": ["sneetchstar"],
        "summary": 'Fix-it fic, starting with episode 5x09, "With All My Heart," through the end of the series.',
        "work_id": "11111111",
        "statistics": {
            "hits": "5822",
            "kudos": "275",
            "words": "95200",
            "status": "2017-03-04",
            "chapters": "18/18",
            "comments": "26",
            "language": "English",
            "bookmarks": "92",
            "published": "2017-02-28",
        },
    }


def get_works():
    works = {
        "00000000": {
            "tags": {
                "fandom": ["Merlin (TV)"],
                "rating": ["Teen And Up Audiences"],
                "category": ["F/M"],
                "freeform": ["Alternate Universe - Canon Divergence"],
                "character": ["Guinevere"],
            },
            "title": "Two Hearts",
            "authors": ["sneetchstar"],
            "summary": 'Fix-it fic, starting with episode 5x09, "With All My Heart," through the end of the series.',
            "work_id": "10001435",
            "statistics": {
                "hits": "5822",
                "kudos": "275",
                "words": "95200",
                "status": "2017-03-04",
                "chapters": "18/18",
                "comments": "26",
                "language": "English",
                "bookmarks": "92",
                "published": "2017-02-28",
            },
        },
        "11111111": get_work(),
    }

    return deepcopy(works)


class BaseModelTest(unittest.TestCase):
    def test___init__(self):
        works = get_works()
        filter = TagsFilter(works)

        assert filter.workIDs == {"00000000", "11111111"}
        assert filter.tag_to_works == {
            ("fandom", "Merlin (TV)"): {"00000000"},
            ("rating", "Teen And Up Audiences"): {"00000000", "11111111"},
            ("rating", "test"): {"11111111"},
            ("category", "F/M"): {"00000000"},
            ("freeform", "Alternate Universe - Canon Divergence"): {"00000000"},
            ("character", "Guinevere"): {"00000000", "11111111"},
            ("relationship", "Guinevere/Arthur Pendragon"): {"11111111"},
        }

    def test_fetch(self):
        works = get_works()
        filter = TagsFilter(works)
        required_tags = [("rating", "Teen And Up Audiences"), ("character", "Guinevere")]
        excluded_tags = [("fandom", "Merlin (TV)"), ("rating", "test")]
        valid_works = filter.fetch(required_tags, excluded_tags)
        assert valid_works == set()

        excluded_tags = [("fandom", "Merlin (TV)"), ("category", "F/M")]
        valid_works = filter.fetch(required_tags, excluded_tags)
        assert valid_works == {"11111111"}
