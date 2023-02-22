from abc import ABC
from random import shuffle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ao3_disco_ranking.types import WorkID


class BaseModel(ABC):
    def fit(self, path_to_jsonl: str) -> None:
        """Train a ranking model."""
        pass

    def embedding(self, work_id: WorkID) -> np.ndarray:
        """Generate a vector embedding for the work.

        This is undefined for second stage models. However, this is required for first stage models
        which are required to generate work embeddings so we can retrieve candidates using some
        approximate nearest neighbors library.
        """
        raise NotImplementedError()

    def rank(
        self, work_id: WorkID, candidates: Iterable[WorkID], works: Dict[WorkID, Any]
    ) -> List[Tuple[WorkID, float]]:
        """Select most relevant candidates.

        The base class simply returns the candidates in some random order.
        """
        candidates = list(candidates)
        shuffle(candidates)
        return [(candidate, i) for i, candidate in enumerate(candidates)]
