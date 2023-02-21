import logging

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

    def basic_query(
        self,
        work_id: WorkID,
        included_tags: Tags = [],
        excluded_tags: Tags = [],
        match_fandom: bool = True,
        N=50,
    ):
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
            if not candidates:
                raise ValueError("No works could be found.")
            logging.info(f"Tags filter selected {len(candidates)}/{N} candidates.")

        logging.info("Applying the graph ranker...")
        graph_results = self.graph_ranker.rank(work_id, candidates, N=N)
        logging.info(f"Graph ranker returned {len(graph_results)}/{N} candidates.")
        if len(graph_results) >= N:
            return graph_results

        logging.info("Falling back to embedding ranker...")
        embedding_results = self.embedding_ranker.rank(work_id, candidates, N=N)
        logging.info(f"Embedding ranker returned {len(embedding_results)}/{N} candidates.")
        return embedding_results
