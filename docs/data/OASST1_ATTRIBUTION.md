# OpenAssistant/oasst1 source attribution

Dataset: [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1), created by OpenAssistant contributors / LAION. Source revision: [`fdf72ae0827c1cda404aff25b6603abec9e3399b`](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b).

License: [Apache License 2.0](OASST1_LICENSE), as declared by the pinned dataset card and LICENSE. Source file: `2023-04-12_oasst_prompts.messages.jsonl.gz`, SHA256 `621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed`.

E17 selects 480 reviewed, nonsynthetic English/Chinese root prompts, excludes normalized exact/near character duplicates, and creates 120 additional mixed-language variants by appending a request for a short summary in the other language. Native prompts retain their original text. This selection and the appended summary instructions are modifications by this project; they are not original OpenAssistant annotations.

Raw source and prepared prompt text are kept in the ignored experiment-data directory; the public manifest includes source message IDs and text hashes for reproducibility. Original source user IDs and assistant answers are not included. Selection is for an independent local research split, not an official OASST1 leaderboard or a claim that these prompts were absent from the target model's pretraining.
