from collections import defaultdict
from typing import Set

from ao3_disco_ranking.types import Tags, WorkID


class TagsFilter:
    def __init__(self, works):
        self.workIDs = set()
        self.tag_to_works = defaultdict(set)
        for work in works:
            self.workIDs.add(work["work_id"])
            for tag_type, tag_values in work["tags"].items():
                for tag_value in tag_values:
                    self.tag_to_works[(tag_type, tag_value)].add(work["work_id"])

    def fetch(self, required_tags: Tags = [], excluded_tags: Tags = []) -> Set[WorkID]:
        candidates = set(self.workIDs)
        for tag_type_value in required_tags:
            candidates = candidates.intersection(self.tag_to_works[tag_type_value])
        for tag_type_value in excluded_tags:
            candidates = candidates - self.tag_to_works[tag_type_value]
        return candidates
