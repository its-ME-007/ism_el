# Enron email corpus

- Archive: `enron_mail_20150507.tar.gz` from https://www.cs.cmu.edu/~enron/ (CMU release, May 2015), downloaded 2026-11-18 to `data/raw/enron_mail.tar.gz`.
- Size 443,254,787 bytes; SHA-256 `b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7`. 150 mailboxes, 517,401 message files.
- Released by FERC during its investigation and redistributed by CMU for research. CMU asks users to respect the privacy of the people involved.
- Sample used: mailboxes `dasovich-j`, `kean-s`, `shapiro-r` (Government Affairs). `tiering fetch-enron` parsed 59,656 records to `data/processed/enron_govaffairs.jsonl`. The adapter kept 23,946 unique messages dated 1998-2002 after content dedupe.
- Access events are a thread re-engagement proxy, not file-access logs. See `docs/DATASET_FEASIBILITY.md`.
