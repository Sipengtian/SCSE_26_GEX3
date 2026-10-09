from datasets import load_dataset

from transformers import (
    AutoTokenizer
)

from peft import (
    AutoPeftModelForCausalLM
)

from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM
)

import random
import numpy as np
import torch

SEED = 48391

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


MODEL_PATH = (
    "models/domain_adapter"
)


## Load domain-adapted model

model = (
    AutoPeftModelForCausalLM
    .from_pretrained(
        MODEL_PATH,
        is_trainable=True
    )
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)


#3 Load SFT dataset

dataset = load_dataset(
    "json",
    data_files=
        "data/sft_clean.jsonl",
    split="train"
)


dataset = dataset.train_test_split(
    test_size=0.1,
    seed=SEED
)



## Convert to prompt/completion format

def format_example(example):

    return {
        "text": (
            "You are a university IT "
            "support assistant.\n\n"
            "User: "
            + example["prompt"]
            + "\n\nAssistant:"
            + " "
            + example["completion"]
        )
    }


dataset = dataset.map(
    format_example,
    remove_columns=
        dataset["train"].column_names
)



## Training configuration


config = SFTConfig(
    output_dir=
        "models/specialized_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_seq_length=256,
    dataset_text_field="text",
    use_liger_kernel=False,
    report_to="none",
    seed=SEED
)


collator = DataCollatorForCompletionOnlyLM(
    response_template="Assistant:",
    tokenizer=tokenizer
)



# Trainer


trainer = SFTTrainer(
    model=model,
    args=config,
    data_collator=collator,
    train_dataset=
        dataset["train"],
    eval_dataset=
        dataset["test"],
    tokenizer=tokenizer
)



# Fine-tune

trainer.train()



# Save

model.save_pretrained(
    "models/specialized_adapter"
)

tokenizer.save_pretrained(
    "models/specialized_adapter"
)
