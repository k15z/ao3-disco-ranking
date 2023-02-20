"""Generate the ranking dataset.

 - For each work, generate an unfiltered ranking of works based on bookmark score.
 - For each work, generate a ranking of works within the same fandom.
"""
import json
import pickle
from tqdm import tqdm
from collections import Counter
from random import sample, random

num_related = 20
num_random = 20

with open("data/works_collections.pkl", "rb") as fin:
    works, collections = pickle.load(fin)
    print(len(works), len(collections))
    workIDs = list(works.keys())

fandom_to_works = {}
for workID, work in tqdm(works.items(), ""):
    for fandom in work["tags"]["fandom"]:
        if fandom not in fandom_to_works:
            fandom_to_works[fandom] = []
        fandom_to_works[fandom].append(workID)

work_to_related_works = {}
for collection in tqdm(collections):
    for w1 in collection:
        if w1 not in work_to_related_works:
            work_to_related_works[w1] = Counter()
        for w2 in collection:
            if w1 != w2:
                work_to_related_works[w1][w2] += 1

train = open("data/train.jsonl", "wt")
test = open("data/test.jsonl", "wt")
for work, related_works in tqdm(work_to_related_works.items()):
    fout = train if random() > 0.1 else test

    # Most related works (no filter)
    candidates = Counter()
    for k, v in related_works.most_common(num_related):
        candidates[k] = v
    for k in sample(workIDs, num_random):
        candidates[k] = related_works[k]
    fout.write(json.dumps({
        "work": work,
        "candidates": dict(candidates)
    }) + "\n")

    # Most related works (fandom match)
    candidates = Counter()
    for fandom in works[work]["tags"]["fandom"]:
        # Add all works in the fandom as candidates
        for candidate in fandom_to_works[fandom]:
            candidates[candidate] = -random()
    for k, v in related_works.most_common():
        if k in candidates:
            candidates[k] = v
    candidates = {k: v for k, v in candidates.most_common(num_random+num_related)}
    if sum(1 for v in candidates.values() if v > 0) < num_random:
        continue # not enough positives to be worth including
    fout.write(json.dumps({
        "work": work,
        "candidates": dict(candidates)
    }) + "\n")
