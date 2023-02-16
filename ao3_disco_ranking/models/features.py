"""Extract dense + sparse features from work JSONs."""
from typing import Any, Dict, List

import numpy as np
from sklearn.preprocessing import RobustScaler


class FeatureExtractor:
    def fit(self, works: Dict[str, Any]):
        self.dense_dims = 12  # hits, kudos, words, comments, bookmarks
        self.sparse_features = {}

        # Normal sparse features
        for tag_type in [
            "fandom",
            "rating",
            "category",
            "relationship",
            "character",
            "freeform",
        ]:
            tag_to_i: Dict[str, int] = {}
            for work in works.values():
                for tag in work["tags"][tag_type]:
                    if tag not in tag_to_i:
                        tag_to_i[tag] = len(tag_to_i)
            self.sparse_features[tag_type] = tag_to_i

        X = np.stack([self._raw_transform(work)[0] for work in works.values()])
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, work: Any):
        dense, sparse = self._raw_transform(work)
        return self.scaler.transform(dense.reshape(1, -1))[0], sparse

    def _raw_transform(self, work: Any):
        dense = np.zeros(self.dense_dims)
        for i, tag in enumerate(["hits", "kudos", "words", "comments", "bookmarks"]):
            try:
                dense[i] = int(work["statistics"][tag])
            except KeyError:
                pass
            except ValueError:
                pass
        try:
            dense[5] = "/" in work["chapters"]
            dense[6] = int(work["chapters"].split("/")[0])
            dense[7] = int(work["chapters"].split("/")[1])
            dense[8] = dense[6] == dense[7]  # Is it complete?
        except KeyError:
            pass
        except ValueError:
            pass
        try:
            dense[9] = int(work["status"].split("-")[0])
            dense[10] = int(work["published"].split("-")[0])
        except KeyError:
            pass
        except ValueError:
            pass
        dense[11] = len(work["summary"])

        sparse: Dict[str, List[int]] = {}
        for tag_type in [
            "fandom",
            "rating",
            "category",
            "relationship",
            "character",
            "freeform",
        ]:
            sparse[tag_type] = []
            for tag in work["tags"][tag_type]:
                if tag not in self.sparse_features[tag_type]:
                    continue
                sparse[tag_type].append(self.sparse_features[tag_type][tag])

        return dense, sparse
