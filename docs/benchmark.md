This page will be periodically updated with the latest results.

| **Model Type** | **Description**                 | **NDCG** |
|----------------|---------------------------------|----------|
| deepsecond     | nn + interactions + listnet     | 0.9576   |
| first          | nn + listnet ranking            | 0.9511   |
| second         | xgboost + pairwise ranking loss | 0.9399   |
| second         | linear regression               | 0.9386   |
| baseline       | random sort                     | 0.7341   |
