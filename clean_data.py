import json


INPUT = "data/domain_corpus_raw.jsonl"
OUTPUT = "data/domain_corpus_clean.jsonl"


clean_records = []


with open(
    INPUT,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        record = json.loads(line)

        if record["quality"] == "ok":

            clean_records.append({
                "id": record["id"],
                "topic": record["topic"],
                "text": record["text"]
            })


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    for record in clean_records:

        file.write(
            json.dumps(record)
            + "\n"
        )


print(
    "Clean records:",
    len(clean_records)
)