from ao3_disco_ranking.types import Tags, WorkID

from .embedding_ranker import EmbeddingRanker
from .graph_ranker import GraphRanker
from .tags_filter import TagsFilter


class QueryHandler:
    def __init__(
        self, tags_filter: TagsFilter, graph_ranker: GraphRanker, embedding_ranker: EmbeddingRanker
    ):
        self.tags_filter = tags_filter
        self.graph_ranker = graph_ranker
        self.embedding_ranker = embedding_ranker

    def basic_query(self, work_id: WorkID, included_tags: Tags = [], excluded_tags: Tags = []):
        candidates = None
        if included_tags or excluded_tags:
            print("Applying the tags filter...")
            candidates = self.tags_filter.fetch(included_tags, excluded_tags)
            if not candidates:
                raise ValueError("No works could be found.")

        print("Applying the graph ranker...")
        graph_results = self.graph_ranker(work_id, candidates)
        if len(graph_results) > 20:
            return graph_results

        print("Falling back to embedding ranker...")
        embedding_results = self.embedding_ranker(work_id, candidates)
        return embedding_results
