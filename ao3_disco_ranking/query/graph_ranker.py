from collections import Counter
from typing import List, Optional, Tuple

from ao3_disco_ranking.types import WorkID


class GraphRanker:
    def __init__(self, collections):
        self.collections = collections

    def rank(
        self, work_id: WorkID, candidates: Optional[List[WorkID]] = None, N: int = 50
    ) -> List[Tuple[WorkID, float]]:
        work_to_score = Counter()
        for collection in self.collections:
            if work_id in collection:
                for other_id in collection:
                    if other_id == work_id:
                        continue
                    if not candidates or (other_id in candidates):
                        work_to_score[other_id] += 1
        return work_to_score.most_common(N)
