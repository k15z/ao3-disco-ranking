"""Second stage models.

These models are designed to be used after an initial candidate pool has been identified and, as 
such, are able to compute more complex interaction features.
"""
import json
import pickle
from typing import List, Tuple

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .base import BaseModel, WorkID


class SecondModel(BaseModel):
    def __init__(self, use_xgb=False):
        self.use_xgb = use_xgb

    def fit(self, path_to_jsonl: str):
        # load the works?
        with open("data/works_collections.pkl", "rb") as fin:
            works, _ = pickle.load(fin)
            self.works = works

        # for each row, extract some pairwise features?
        X, y, groups = [], [], []
        with open(path_to_jsonl, "rt") as fin:
            for x in tqdm(map(json.loads, fin)):
                for work, score in x["candidates"].items():
                    X.append(self._get_features(x["work"], work))
                    y.append(score)
                groups.append(len(x["candidates"]))

        # start with a naive linear regression model?
        if self.use_xgb:
            self.model = xgb.XGBRanker(
                tree_method="auto",
                objective="rank:pairwise",
            )
            self.model.fit(X, y, group=groups, verbose=True)
        else:
            self.model = LinearRegression()
            self.model.fit(X, y)

    def _get_features(self, work1, work2):
        work1, work2 = self.works[work1], self.works[work2]
        return [
            len(set(work1["tags"]["fandom"]).intersection(set(work2["tags"]["fandom"]))),
            len(set(work1["tags"]["rating"]).intersection(set(work2["tags"]["rating"]))),
            len(set(work1["tags"]["category"]).intersection(set(work2["tags"]["category"]))),
            len(
                set(work1["tags"]["relationship"]).intersection(set(work2["tags"]["relationship"]))
            ),
            len(set(work1["tags"]["character"]).intersection(set(work2["tags"]["character"]))),
            len(set(work1["tags"]["freeform"]).intersection(set(work2["tags"]["freeform"]))),
        ]

    def rank(self, work_id: WorkID, candidates: List[WorkID]) -> List[Tuple[WorkID, float]]:
        X = []
        for candidate in candidates:
            X.append(self._get_features(work_id, candidate))
        y = self.model.predict(X)
        candidate_score = [(c, s) for s, c in zip(y, candidates)]
        return candidate_score
