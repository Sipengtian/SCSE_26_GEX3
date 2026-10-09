import torch
import json
import os

from transformers import (
    AutoTokenizer
)

from peft import (
    AutoPeftModelForCausalLM
)


MODEL_PATH = (
    os.environ.get(
        "MODEL_PATH",
        "models/specialized_adapter"
    )
)


model = (
    AutoPeftModelForCausalLM
    .from_pretrained(
        MODEL_PATH
    )
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)


model.eval()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)


with open(
    "data/sft_challenge_test.jsonl",
    "r",
    encoding="utf-8"
) as file:

    tests = [
        json.loads(line)
        for line in file
    ]


for test in tests:

    prompt = (
        "You are a university IT "
        "support assistant.\n\n"
        "User: "
        + test["prompt"]
        + "\n\nAssistant:"
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )


    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    print("\n====================")
    print("QUESTION:")
    print(test["prompt"])

    print("\nEXPECTED POINTS:")
    print(test["expected_points"])

    print("\nMODEL:")
    print(response)
