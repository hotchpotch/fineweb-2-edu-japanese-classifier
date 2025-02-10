#!/usr/bin/env python
import argparse
import os
import unicodedata

import datasets
import evaluate
import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def unicode_normalize(text):
    """Unicode normalization (NFKC)"""
    return unicodedata.normalize("NFKC", text)


def compute_classification_metrics(pred_probs, labels, threshold):
    """
    Function to compute metrics for binary classification.
    pred_probs: Probability scores for positive class from the model (continuous values)
    labels: Ground truth labels (0 or 1)
    threshold: Classification threshold (default is 0.5, but can be adjusted to improve recall)
    """
    # Convert probabilities to class labels using threshold
    preds = (pred_probs >= threshold).astype(int)

    # Calculate various metrics using sklearn functions
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)

    # Display detailed classification report and confusion matrix
    report = classification_report(labels, preds, digits=4)
    cm = confusion_matrix(labels, preds)
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main(args):
    # 1. Load and preprocess dataset
    print("Loading dataset...")
    ds = datasets.load_dataset(args.dataset_name)
    # Unicode normalization
    ds = ds.map(
        lambda x: {"text": unicode_normalize(x["text"])},
        num_proc=args.num_proc,
    )
    # Map score=5 to 4 if needed (preserved but converting to binary for this task)
    ds = ds.map(
        lambda x: {"score": 4 if x[args.target_column] == 5 else x[args.target_column]},
        num_proc=args.num_proc,
    )
    # Create binary labels based on scores (score<=2 -> 0, score>=3 -> 1)
    ds = ds.map(
        lambda x: {"binary_label": 0 if x[args.target_column] <= 2 else 1},
        num_proc=args.num_proc,
    )

    wikipedia_dataset_count = args.add_wikipedia_dataset_count
    if wikipedia_dataset_count > 0:
        wikipedia_dataset = datasets.load_dataset(
            "hpprc/jawiki-paragraphs", "default", split="train"
        )
        # random sampling
        target_indexes = np.random.choice(
            len(wikipedia_dataset), wikipedia_dataset_count, replace=False
        )  # type: ignore
        print(wikipedia_dataset)
        wikipedia_dataset = wikipedia_dataset.select(target_indexes)
        # select columns
        wikipedia_dataset = wikipedia_dataset.map(
            lambda x: {
                "text": unicode_normalize(x["text"]),
                "binary_label": 1,
            },
            num_proc=15,
            remove_columns=wikipedia_dataset.column_names,
        )
        # add wikipedia dataset to the original dataset only for training
        print(f"Adding {wikipedia_dataset_count} samples from wikipedia dataset")
        print(f"Original train dataset size: {len(ds['train'])}")
        ds["train"] = datasets.concatenate_datasets(
            [ds["train"], wikipedia_dataset], axis=0
        )
        print(f"New dataset size: {len(ds['train'])}")

    # Convert train/test to pandas.DataFrame
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    if args.debug:
        print("Debug mode enabled: limiting dataset sizes")
        train_df = train_df.head(10000)
        test_df = test_df.head(100)

    train_text = train_df["text"].tolist()
    # Use "binary_label" as target
    train_labels = train_df["binary_label"].values
    test_text = test_df["text"].tolist()
    test_labels = test_df["binary_label"].values

    # 2. Convert text to embedding vectors using SentenceTransformer
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        args.embedding_model_name, device="cpu", truncate_dim=args.dim
    )
    print("Encoding training texts...")
    train_data = embedder.encode(
        train_text, convert_to_numpy=True, show_progress_bar=True, batch_size=1024 * 4
    )
    print("Encoding test texts...")
    test_data = embedder.encode(
        test_text, convert_to_numpy=True, show_progress_bar=True, batch_size=1024 * 4
    )
    del embedder

    # 3. Split training data into train/validation (using 5% of train data for validation)
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.05, random_state=42, stratify=train_labels
    )

    # 4. Calculate scale_pos_weight for class imbalance
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    # 5. Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    # 6. Set hyperparameters (using binary objective for binary classification)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_gain_to_split": args.min_gain_to_split,
        "min_data_in_leaf": args.min_data_in_leaf,
        "scale_pos_weight": scale_pos_weight,
        "verbose": 1,
    }

    # 7. Train model with early stopping and logging
    print("Training LightGBM model...")
    gbm = lgb.train(
        params,
        train_dataset,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds),
            lgb.log_evaluation(period=args.test_eval_steps),
        ],
    )

    # 8. Predict and evaluate on validation set
    print("Evaluating on validation set...")
    # For binary classification, predict returns probability of positive class
    val_pred_probs = gbm.predict(X_val)
    val_metrics = compute_classification_metrics(val_pred_probs, y_val, args.threshold)
    print("\nValidation Metrics:")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # 9. Display top 10 feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": [f"feature_{i}" for i in range(train_data.shape[1])],
            "importance": gbm.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    # 10. Save model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(args.checkpoint_dir, "lightgbm_model.txt")
    gbm.save_model(model_path)
    print(f"Model saved to {model_path}")

    # 11. Predict and evaluate on test set
    print("Evaluating on test set...")
    test_pred_probs = gbm.predict(test_data)
    test_metrics = compute_classification_metrics(
        test_pred_probs, test_labels, args.threshold
    )
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LightGBM Trainer for Binary Text Classification (score <=2 vs score >=3)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hotchpotch/fineweb-2-edu-japanese-scores",
        help="Name of the dataset to load",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="score",
        help="Name of the target column (original score)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimension of embedding vectors",
    )
    parser.add_argument(
        "--add_wikipedia_dataset_count",
        type=int,
        default=0,
        help="Number of samples to add from the Japanese Wikipedia dataset",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="hotchpotch/static-embedding-japanese",
        help="Model name for SentenceTransformer",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./tmp/lightgbm_binary_models/",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=5000,
        help="Number of boosting rounds for LightGBM",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=50,
        help="Number of rounds for early stopping",
    )
    parser.add_argument(
        "--test_eval_steps",
        type=int,
        default=100,
        help="Interval of rounds for evaluation logging",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=11,
        help="Number of processes for dataset mapping",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (limits dataset size)",
    )
    # LightGBM hyperparameters
    parser.add_argument(
        "--num_leaves", type=int, default=128, help="Number of leaves in trees"
    )
    parser.add_argument("--max_depth", type=int, default=15, help="Maximum tree depth")
    parser.add_argument(
        "--learning_rate", type=float, default=0.02, help="Learning rate"
    )
    parser.add_argument(
        "--feature_fraction",
        type=float,
        default=0.8,
        help="Feature subsampling ratio",
    )
    parser.add_argument(
        "--bagging_fraction",
        type=float,
        default=0.8,
        help="Data subsampling ratio",
    )
    parser.add_argument("--bagging_freq", type=int, default=1, help="Bagging frequency")
    parser.add_argument(
        "--lambda_l1", type=float, default=0.1, help="L1 regularization coefficient"
    )
    parser.add_argument(
        "--lambda_l2", type=float, default=1.0, help="L2 regularization coefficient"
    )
    parser.add_argument(
        "--min_gain_to_split",
        type=float,
        default=0.1,
        help="Minimum gain for splitting",
    )
    parser.add_argument(
        "--min_data_in_leaf",
        type=int,
        default=50,
        help="Minimum number of data in each leaf",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for converting probabilities to binary predictions (adjust to improve recall)",
    )

    args = parser.parse_args()
    main(args)
