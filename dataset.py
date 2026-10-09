from datasets import load_dataset

data = load_dataset(
    "json",
    data_files="data/domain_corpus_raw.jsonl"
)

print(data)