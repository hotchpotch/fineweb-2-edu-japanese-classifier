# base: https://github.com/huggingface/cosmopedia/blob/main/classification/train_edu_bert.py
# Apache 2.0 License

import argparse
import os
import unicodedata

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import ClassLabel, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]  # type: ignore
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]  # type: ignore
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]  # type: ignore
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]  # type: ignore

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)  # type: ignore
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, score_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_weights = (
            score_weights.to(self.args.device) if score_weights is not None else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()  # Keep labels as float type
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(
            -1
        )  # Convert from (batch_size, 1) to (batch_size,)

        # Apply class weights to MSELoss (different weights for each class)
        loss_fn = nn.MSELoss(
            reduction="none"
        )  # Use reduction='none' to get individual losses
        loss = loss_fn(logits, labels)

        if self.score_weights is not None:
            # Apply appropriate class weights based on labels
            weight_per_sample = self.score_weights[
                labels.long()
            ]  # Get weights for each sample
            loss = loss * weight_per_sample  # Apply weights

        loss = loss.mean()  # Calculate mean loss for the entire batch

        return (loss, outputs) if return_outputs else loss


def unicode_normalize(text):
    # Normalize using NFKC
    normalized_text = unicodedata.normalize("NFKC", text)
    return normalized_text


def main(args):
    dataset = load_dataset(args.dataset_name, num_proc=15)
    dataset = dataset.map(
        lambda x: {
            "text": unicode_normalize(x["text"]),
        },
        num_proc=15,
    )
    dataset = dataset.map(
        lambda x: {"score": 4 if x["score"] == 5 else x["score"]}, num_proc=11
    )

    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 4)},
        num_proc=15,  # type: ignore
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(5)])
    )
    test_dataset = dataset["test"]
    dataset = dataset["train"].train_test_split(  # type: ignore
        train_size=0.95, seed=42, stratify_by_column=args.target_column
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,
        classifier_dropout=0.05,
        hidden_dropout_prob=0.05,
        output_hidden_states=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(model.config.max_position_embeddings, 512),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True, num_proc=15)
    test_dataset = test_dataset.map(preprocess, batched=True, num_proc=15)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=2e-5,
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        per_device_train_batch_size=256,
        # per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=512,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        # push_to_hub=True,
    )

    if args.score_weights:
        score_weights = torch.tensor(args.score_weights, dtype=torch.float32)
    else:
        score_weights = None

    trainer = CustomTrainer(
        score_weights=score_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))

    print("\nFinal Evaluation Results:")
    final_metrics = trainer.evaluate()
    print(final_metrics)

    # eval test dataset
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("\nTest Evaluation Results:")
    print(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="hotchpotch/mMiniLMv2-L6-H384"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hotchpotch/fineweb-2-edu-japanese-scores",
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./tmp/transformer_models/",
    )
    parser.add_argument(
        "--score_weights",
        type=float,
        nargs="+",
        default=None,
        help="Class weights for the loss function, expected as a space-separated list of floats.",
    )

    # parser.add_argument(
    #     "--output_model_name", type=str, default="HuggingFaceTB/fineweb-edu-scorer"
    # )
    args = parser.parse_args()

    main(args)
