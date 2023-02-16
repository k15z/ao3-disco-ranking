from ao3_disco_ranking.models import BaseModel, SecondModel
from ao3_disco_ranking.utils import score

model = BaseModel()
model.fit("data/train.jsonl")
print("cdg:", score(model, "data/test.jsonl"))

model = SecondModel(use_xgb=False)
model.fit("data/train.jsonl")
print("cdg:", score(model, "data/test.jsonl"))

model = SecondModel(use_xgb=True)
model.fit("data/train.jsonl")
print("cdg:", score(model, "data/test.jsonl"))
