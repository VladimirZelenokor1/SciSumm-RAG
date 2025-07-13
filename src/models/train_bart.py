#!/usr/bin/env python3
import itertools
import json

import evaluate
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from pathlib import Path

def load_data(path: Path):
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)

def preprocess(batch, tokenizer):
    inputs = tokenizer(
        batch["body"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch["abstract"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }

def make_output_dir(root: Path, params: dict):
    # e.g. bart-finetune_bs4_lr3e-05_ep5
    name = f"bs{params['per_device_train_batch_size']}" \
           f"_lr{params['learning_rate']}" \
           f"_ep{params['num_train_epochs']}"
    return root / "experiments" / "bart-finetune" / name

def train_one(config: dict, project_root: Path, train_ds, val_ds, tokenizer, model, rouge):
    # 1) preprocess → datasets
    tokenized_train = train_ds.map(
        lambda b: preprocess(b, tokenizer),
        batched=True, remove_columns=["body","abstract"], cache_file_name=None
    )
    tokenized_val = val_ds.map(
        lambda b: preprocess(b, tokenizer),
        batched=True, remove_columns=["body","abstract"], cache_file_name=None
    )

    # 2) collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest", label_pad_token_id=tokenizer.pad_token_id
    )

    # 3) build args
    output_dir = make_output_dir(project_root, config)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        warmup_steps=50,
        logging_steps=10,
        eval_steps=50,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
    )

    # 4) metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = torch.where(
            torch.tensor(labels) != -100,
            torch.tensor(labels),
            torch.tensor(tokenizer.pad_token_id)
        ).tolist()
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: v * 100 for k, v in res.items()}

    # 5) trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6) train + save
    trainer.train()
    trainer.save_model(output_dir / "best-model")
    tokenizer.save_pretrained(output_dir / "best-model")

    # 7) final eval
    metrics = trainer.evaluate(eval_dataset=tokenized_val)
    return metrics

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir     = project_root / "data" / "training"
    train_csv    = data_dir / "train_pairs.csv"
    val_csv      = data_dir / "validation_pairs.csv"
    grid_path    = project_root / "experiments" / "configs" / "bart_grid.json"

    # load data & model & grid
    train_ds = load_data(train_csv)
    val_ds   = load_data(val_csv)

    model_name = "facebook/bart-large-cnn"
    tokenizer  = BartTokenizerFast.from_pretrained(model_name)
    base_model = BartForConditionalGeneration.from_pretrained(model_name)

    rouge = evaluate.load("rouge")

    # read grid
    grid = json.load(open(grid_path))
    keys, values = zip(*grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # run
    all_results = []
    for config in combos:
        print(f"\n=== Training with config: {config} ===")
        # fresh model for each run!
        model = BartForConditionalGeneration.from_pretrained(model_name)
        metrics = train_one(config, project_root, train_ds, val_ds, tokenizer, model, rouge)
        print(f"--> Results: {metrics}\n")
        all_results.append({"config": config, "metrics": metrics})

if __name__ == "__main__":
    main()