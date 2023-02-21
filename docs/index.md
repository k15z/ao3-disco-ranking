# AO3 Disco Ranking

This repository hosts the ranking models used by the AO3 Discovery Engine.

## Usage

This library is designed to work with GuildAI which makes it easy to track and manage different
model variants. For example, you can use guild to automatically tune the hyperparameters for 
the first stage model:

```bash
guild run model_type=first \
    num_epochs='[2,5,10,100]' \
    output_dim='[64,128,256]' \
    max_hash_size='[10000,25000,50000]' \
    use_batch_norm='[True,False]' \
    --optimizer gp --maximize ncdg
```

Pre-trained models are stored as pickle files which can be loaded by the AO3 Disco server directly
and used for inference.

## Design

At a high level, the ranking pipeline works at follows:

 - User submits a query (work + filters).
 
 - First, we generate ~200 candidates.
    - **prematch.** We find all works that are connected to the queried work (i.e. same author, 
    same fandom, same bookmarkers, etc.) and generate a score similar to something like PageRank. 
    If the queried work isn't very popular, this could fail to return any results (i.e. if there 
    are very few works in the fandom).
    - **first-stage.** We use a neural network to map the queried work to a vector and find the 100
    approximate nearest neighbors (after filtering). Note that this model doesn't consider any 
    interactions between the works - it independently maps every work to a vector.

 - Then, we apply a **second-stage** model that contains more complex interaction features (i.e. 
   number of shared tags), etc. This model is used to generate the final ranking of the results.

This repository contains the code for training the **first** and **second** stage models.
