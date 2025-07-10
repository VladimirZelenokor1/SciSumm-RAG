# src/generator/hf_summarizer.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# choose a model
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Pipeline initialization
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def generate_summary_hf(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    do_sample: bool = False
) -> str:
    """
    Generates a short summary for a given text via Hugging Face summarization pipeline.
    """
    # model automatically splits long texts into suitable chunks
    outputs = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample
    )
    # return the first resume
    return outputs[0]["summary_text"].strip()
