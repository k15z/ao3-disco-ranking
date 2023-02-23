import gc
import logging
import pickle
import typing
from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

from ao3_disco_ranking.ao3 import get_work_jsons
from ao3_disco_ranking.types import Tags, WorkID

from .embedding_ranker import EmbeddingRanker
from .graph_ranker import GraphRanker
from .tags_filter import TagsFilter

logger = logging.getLogger()


class QueryHandler:
    def __init__(
        self, tags_filter: TagsFilter, graph_ranker: GraphRanker, embedding_ranker: EmbeddingRanker
    ):
        self.tags_filter = tags_filter
        self.graph_ranker = graph_ranker
        self.embedding_ranker = embedding_ranker

    def save(self, path_to_pkl: str):
        with open(path_to_pkl, "wb") as fout:
            pickle.dump(self, fout, protocol=-1)

    @classmethod
    def load(self, path_to_pkl: str) -> "QueryHandler":
        with open(path_to_pkl, "rb") as fin:
            gc.disable()
            handler = pickle.load(fin)
            gc.enable()
        return handler

    def basic_query(
        self,
        work_id: WorkID,
        blocklist: List[WorkID] = [],
        included_tags: Tags = [],
        excluded_tags: Tags = [],
        match_fandom: bool = True,
        num_results: int = 50,
    ) -> Sequence[Tuple[WorkID, float]]:
        logging.info(f"Querying for {work_id}.")
        work_json = list(get_work_jsons([work_id]))[0]

        one_or_more_tags = []
        if match_fandom:
            for fandom in work_json["tags"]["fandom"]:
                one_or_more_tags.append(("fandom", fandom))

        candidates = None
        if included_tags or excluded_tags or one_or_more_tags:
            logging.info("Applying the tags filter...")
            candidates = self.tags_filter.fetch(included_tags, excluded_tags, one_or_more_tags)
            candidates -= set(blocklist)
            if not candidates:
                raise ValueError("No works could be found.")
            logging.info(f"Tags filter selected {len(candidates)}/{num_results} candidates.")

        logging.info("Applying the graph ranker...")
        graph_results = self.graph_ranker.rank(
            work_id, candidates, blocklist=blocklist, num_results=num_results
        )
        logging.info(f"Graph ranker returned {len(graph_results)}/{num_results} candidates.")
        if len(graph_results) >= num_results:
            return graph_results

        logging.info("Falling back to embedding ranker...")
        embedding_results = self.embedding_ranker.rank(
            work_id, candidates, blocklist=blocklist, num_results=num_results
        )
        logging.info(
            f"Embedding ranker returned {len(embedding_results)}/{num_results} candidates."
        )
        return embedding_results

    def multi_query(
        self,
        work_ids: List[WorkID],
        blocklist: List[WorkID],
        included_tags: Tags = [],
        excluded_tags: Tags = [],
        match_fandom: bool = True,
        num_results: int = 50,
    ) -> Sequence[Tuple[WorkID, float]]:
        work_to_score: typing.Counter[str] = Counter()
        work_to_sources = defaultdict(set)
        for work_id in work_ids:
            for other_id, score in self.basic_query(
                work_id, blocklist, included_tags, excluded_tags, match_fandom, num_results
            ):
                work_to_score[other_id] += score  # type: ignore
                work_to_sources[other_id].add(work_id)
        return work_to_score.most_common(num_results)  # type: ignore
