| run_id | model | accuracy | macro_f1 | recall_negative | recall_positive | train_time_s | val_infer_time_s | model_size_mb |
|---|---|---|---|---|---|---|---|---|
| E0_majority | majority | 0.8539 | 0.4606 | 0.0000 | 1.0000 | 0.0120 | 0.0000 | 0.0010 |
| E1_word | tfidf_lr | 0.8761 | 0.6043 | 0.1619 | 0.9984 | 0.4410 | 0.0680 | 1.6390 |
| E2_word_char | tfidf_lr | 0.9019 | 0.7304 | 0.3571 | 0.9951 | 1.7990 | 0.3810 | 3.6210 |
| E2_best | tfidf_lr | 0.9221 | 0.8474 | 0.7619 | 0.9495 | 2.0140 | 0.3530 | 3.3240 |
| distilbert_seed42 | distilbert | 0.9395 | 0.8750 | 0.7571 | 0.9707 | 262.2870 | 6.7640 | 256.1050 |
| distilbert_seed52 | distilbert | 0.9402 | 0.8746 | 0.7429 | 0.9739 | 203.9620 | 4.7640 | 256.1050 |
| distilbert_seed62 | distilbert | 0.9381 | 0.8762 | 0.7905 | 0.9633 | 193.9470 | 4.6880 | 256.1050 |