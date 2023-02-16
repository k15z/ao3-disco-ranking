from ao3_disco_ranking.models import BaseModel, FirstModel, SecondModel
from ao3_disco_ranking.utils import score

model_type = "baseline"

use_xgb = False

lr = 1e-2
batch_size = 100
num_epochs = 2
output_dim = 128
max_hash_size = 50000
dropout = 0.0
use_batch_norm = False

if model_type == "baseline":
    model = BaseModel()
elif model_type == "first":
    model = FirstModel(
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dim=output_dim,
        max_hash_size=max_hash_size,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )
elif model_type == "second":
    model = SecondModel(use_xgb=use_xgb)

model.fit("data/train.jsonl")
print("ncdg:", score(model, "data/test.jsonl"))
