import json

import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from ao3_disco_ranking.models import BaseModel, SecondModel


def score(model: BaseModel, path_to_jsonl: str):
    """Compute the discounted cumulative gain."""
    ndcg = []
    with open(path_to_jsonl, "rt") as fin:
        for x in tqdm(map(json.loads, fin), "eval"):
            work = x["work"]
            candidates = x["candidates"]
            prediction = model.rank(work, candidates)
            y_true, y_pred = [], []
            for work, score in prediction:
                y_true.append(candidates[work])
                y_pred.append(score)
            ndcg.append(ndcg_score([y_true], [y_pred]))
    return np.mean(ndcg)
