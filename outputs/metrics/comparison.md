| run_id | model | accuracy | macro_f1 | recall_negative | recall_positive | train_time_s | val_infer_time_s | peak_ram_mb | model_size_mb |
|---|---|---|---|---|---|---|---|---|---|
| E0_majority | majority | 0.8539 | 0.4606 | 0.0000 | 1.0000 | 0.0120 | 0.0000 | 184.7770 | 0.0010 |
| E1_word | tfidf_lr | 0.8761 | 0.6043 | 0.1619 | 0.9984 | 0.4410 | 0.0680 | 220.2300 | 1.6390 |
| E2_word_char | tfidf_lr | 0.9019 | 0.7304 | 0.3571 | 0.9951 | 1.7990 | 0.3810 | 295.0780 | 3.6210 |
| E2_best | tfidf_lr | 0.9221 | 0.8474 | 0.7619 | 0.9495 | 2.0140 | 0.3530 | 292.7150 | 3.3240 |