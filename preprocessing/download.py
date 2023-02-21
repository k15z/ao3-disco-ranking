import pickle
from ao3_disco.ranking.data import get_works_collections

works, collections = get_works_collections(include_content=False)
with open("works_collections.pkl", "wb") as fout:
    pickle.dump((works, collections), fout)
