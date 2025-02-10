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
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split


def unicode_normalize(text):
    """Unicode normalization (NFKC)"""
    return unicodedata.normalize("NFKC", text)


def compute_metrics(predictions, labels):
    """
    Function to compute evaluation metrics.
    In addition to regression metrics (RMSE, MAE),
    calculates Precision, Recall, F1 (macro) and Accuracy
    by rounding prediction values at the classification level.
    """
    # Regression metrics
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    mae = mean_absolute_error(labels, predictions)

    # Round predictions and labels for classification evaluation
    preds_rounded = np.round(predictions).astype(int)
    labels_rounded = np.round(labels).astype(int)

    # Calculate various classification metrics using evaluate library
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    precision = precision_metric.compute(
        predictions=preds_rounded, references=labels_rounded, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds_rounded, references=labels_rounded, average="macro"
    )["recall"]
    f1 = f1_metric.compute(
        predictions=preds_rounded, references=labels_rounded, average="macro"
    )["f1"]
    accuracy = accuracy_metric.compute(
        predictions=preds_rounded, references=labels_rounded
    )["accuracy"]

    # Display detailed classification report and confusion matrix
    report = classification_report(labels_rounded, preds_rounded)
    cm = confusion_matrix(labels_rounded, preds_rounded)
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "rmse": rmse,
        "mae": mae,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    # 1. Load and preprocess dataset
    print("Loading dataset...")
    ds = datasets.load_dataset(args.dataset_name)
    # Unicode normalization of text
    ds = ds.map(
        lambda x: {"text": unicode_normalize(x["text"])},
        num_proc=args.num_proc,
    )
    # Map score 5 to 4 (specified by args.target_column)
    ds = ds.map(
        lambda x: {
            args.target_column: 4
            if x[args.target_column] == 5
            else x[args.target_column]
        },
        num_proc=args.num_proc,
    )

    # Convert train/test to pandas.DataFrame
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    if args.debug:
        print("Debug mode enabled: limiting dataset sizes")
        train_df = train_df.head(10000)
        test_df = test_df.head(100)

    train_text = train_df["text"].tolist()
    train_labels = train_df[args.target_column].values
    test_text = test_df["text"].tolist()
    test_labels = test_df[args.target_column].values

    # 2. Get embedding vectors
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        args.embedding_model_name, device="cpu", truncate_dim=args.dim
    )
    print("Encoding training texts...")
    train_data = embedder.encode(
        train_text, convert_to_numpy=True, show_progress_bar=True
    )
    print("Encoding test texts...")
    test_data = embedder.encode(
        test_text, convert_to_numpy=True, show_progress_bar=True
    )
    del embedder

    # 3. Split training and validation data (use 5% of training data for validation)
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.05, random_state=42
    )

    # 4. Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    # 5. Set hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
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
        "verbose": 1,
    }

    # 6. Train model (with early stopping and logging)
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

    # 7. Predict and evaluate on validation data
    print("Evaluating on validation set...")
    val_pred = gbm.predict(X_val)
    val_metrics = compute_metrics(val_pred, y_val)
    print("\nValidation Metrics:")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # 8. Display feature importance (top 10)
    feature_importance = pd.DataFrame(
        {
            "feature": [f"feature_{i}" for i in range(train_data.shape[1])],
            "importance": gbm.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    # 9. Save model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(args.checkpoint_dir, "lightgbm_model.txt")
    gbm.save_model(model_path)
    print(f"Model saved to {model_path}")

    # 10. Predict and evaluate on test data
    print("Evaluating on test set...")
    test_pred = gbm.predict(test_data)
    test_metrics = compute_metrics(test_pred, test_labels)
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LightGBM Trainer for Score Prediction"
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
        help="Name of the target column",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimension of embedding vectors",
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
        default="./tmp/lightgbm_models/",
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

    args = parser.parse_args()
    main(args)
