import logging
from collections import defaultdict
from typing import Set

from ao3_disco_ranking.types import Tags, WorkID

logger = logging.getLogger()


class TagsFilter:
    def __init__(self, works):
        self.workIDs = set()
        self.tag_to_works = defaultdict(set)
        for work_id, work in works.items():
            self.workIDs.add(work_id)
            for tag_type, tag_values in work["tags"].items():
                for tag_value in tag_values:
                    self.tag_to_works[(tag_type, tag_value)].add(work_id)

    def fetch(
        self, required_tags: Tags = [], excluded_tags: Tags = [], one_or_more_tags: Tags = []
    ) -> Set[WorkID]:
        if not one_or_more_tags:
            candidates = set(self.workIDs)
        else:
            candidates = set()
            for tag_type_value in one_or_more_tags:
                candidates = candidates.union(self.tag_to_works[tag_type_value])
            logger.info(f"Found {len(candidates)} with one_or_more...")

        for tag_type_value in required_tags:
            candidates = candidates.intersection(self.tag_to_works[tag_type_value])
        if required_tags:
            logger.info(f"Found {len(candidates)} after requiring...")

        for tag_type_value in excluded_tags:
            candidates = candidates - self.tag_to_works[tag_type_value]
        if excluded_tags:
            logger.info(f"Found {len(candidates)} after excluding...")

        return candidates
