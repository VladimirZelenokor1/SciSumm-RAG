#!/usr/bin/env python3
import re
import json
import numpy as np
from pathlib import Path

import evaluate
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def load_val_dataset(val_csv: Path, tokenizer: BartTokenizerFast):
    import pandas as pd
    from datasets import Dataset
    df = pd.read_csv(val_csv)
    ds = Dataset.from_pandas(df)
    def preprocess(batch):
        inputs = tokenizer(
            batch["body"],
            truncation=True, padding="max_length", max_length=512
        )

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                batch["abstract"],
                truncation=True, padding="max_length", max_length=128
            )
        return {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels":         targets["input_ids"],
        }
    return ds.map(preprocess, batched=True, remove_columns=["body","abstract"])


def parse_config_from_name(name: str):
    m = re.match(r"bs(\d+)_lr([\d\.e-]+)_ep(\d+)", name)
    if not m:
        return {}
    return {
        "per_device_train_batch_size": int(m.group(1)),
        "learning_rate": float(m.group(2)),
        "num_train_epochs": int(m.group(3)),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    val_csv      = project_root / "data" / "training" / "validation_pairs.csv"
    exp_root     = project_root / "experiments" / "bart-finetune"

    # 1) Basic Tokenizer and Metrics
    base_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
    rouge          = evaluate.load("rouge")
    pad_id         = base_tokenizer.pad_token_id

    # 2) Tokenize the validation dataset once
    tokenized_val = load_val_dataset(val_csv, base_tokenizer)

    # 3) Define compute_metrics here so that it “sees” tokenizer, rouge and pad_id
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds  = np.where(preds  < 0, pad_id, preds)
        labels = np.where(labels < 0, pad_id, labels)

        decoded_preds = base_tokenizer.batch_decode(
            preds,  skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_labels = base_tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        res = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        return {k: v * 100 for k, v in res.items()}

    results = []
    for model_dir in exp_root.iterdir():
        best = model_dir / "best-model"
        if not best.is_dir():
            continue
        config = parse_config_from_name(model_dir.name)
        print(f"Evaluating {model_dir.name} …")

        model     = BartForConditionalGeneration.from_pretrained(str(best))
        tokenizer = BartTokenizerFast.from_pretrained(str(best))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="longest",
            label_pad_token_id=pad_id
        )

        # 7) Trainer
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(model_dir / "tmp"),
            per_device_eval_batch_size=config.get("per_device_train_batch_size", 4),
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,
            logging_strategy="no",
            save_strategy="no",
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()
        results.append({"config": config, "metrics": metrics})

    # results
    out_dir = exp_root
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame([
        {**r["config"], **r["metrics"]} for r in results
    ])
    df.to_csv(out_dir / "grid_search_results.csv", index=False)
    with open(out_dir / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Results:")
    print(" -", out_dir / "grid_search_results.csv")
    print(" -", out_dir / "grid_search_results.json")


if __name__ == "__main__":
    main()
