"""Build a QueryHandler and export it to a pickle file."""
import logging
from time import time

logging.basicConfig(level=logging.DEBUG)

import pickle

from ao3_disco_ranking.query import (
    EmbeddingRanker,
    GraphRanker,
    QueryHandler,
    TagsFilter,
)

with open("data/works_collections.pkl", "rb") as fin:
    works, collections = pickle.load(fin)
    logging.info(f"Works: {len(works)}")
    logging.info(f"Collections: {len(collections)}")


tags_filter = TagsFilter(works)
graph_ranker = GraphRanker(collections)
with open("/Users/kevz/Downloads/model.pkl", "rb") as fin:
    model = pickle.load(fin)
embedding_ranker = EmbeddingRanker(model, works)

handler = QueryHandler(tags_filter, graph_ranker, embedding_ranker)
handler.basic_query("45212884", num_results=10, match_fandom=True)
handler.save("handler.pkl")

start = time()
new_handler = QueryHandler.load("handler.pkl")
print("Startup Time:", time() - start)

start = time()
new_handler.basic_query("45212884", num_results=10, match_fandom=True)
print("Query Time:", time() - start)
