import json
from typing import Any, Dict

import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from ao3_disco_ranking.models import BaseModel


def score(model: BaseModel, works: Dict[str, Any], path_to_jsonl: str, debug: bool = False):
    """Compute the discounted cumulative gain."""
    ndcg = []
    with open(path_to_jsonl, "rt") as fin:
        for x in tqdm(map(json.loads, fin), "eval"):
            work = x["work"]
            candidates = x["candidates"]
            prediction = model.rank(work, candidates, works)
            y_true, y_pred = [], []
            for work, score in prediction:
                y_true.append(candidates[work])
                y_pred.append(score)
            ndcg.append(ndcg_score([y_true], [y_pred]))
            if debug and len(ndcg) > 1000:
                break
    return np.mean(ndcg)
