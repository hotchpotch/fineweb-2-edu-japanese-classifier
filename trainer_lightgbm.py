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
    # Normalize using NFKC
    normalized_text = unicodedata.normalize("NFKC", text)
    return normalized_text


def prepare_data(dataset_name="hotchpotch/wip-fineweb-2-edu-japanese-scores"):
    dataset = datasets.load_dataset(dataset_name)
    dataset = dataset.map(lambda x: {"text": unicode_normalize(x["text"])}, num_proc=11)
    # If the value in dataset['score'] is 5, convert it to 4
    dataset = dataset.map(
        lambda x: {"score": 4 if x["score"] == 5 else x["score"]}, num_proc=11
    )
    return dataset


ds = prepare_data()
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

# Uncomment below for debugging (limit dataset sizes)
# if True:
#     # Debug: limit data size
#     train_df = train_df[:10000]
#     test_df = test_df[:100]

train_labels = train_df["score"].values
test_labels = test_df["score"].values

train_text = train_df["text"].values.tolist()
test_text = test_df["text"].values.tolist()

model_name = "hotchpotch/static-embedding-japanese"
model = SentenceTransformer(model_name, device="cpu", truncate_dim=1024)

train_data = model.encode(train_text, convert_to_numpy=True, show_progress_bar=True)
test_data = model.encode(test_text, convert_to_numpy=True, show_progress_bar=True)
del model

# Data preparation
# train_data: features with shape (270000, 1024)
# train_labels: scores with shape (270000,)
# test_data: features with shape (30000, 1024)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels, test_size=0.05, random_state=42
)

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train, label=y_train)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# Example parameter settings
params = {
    "objective": "regression",  # Regression task
    "metric": "rmse",  # Evaluation metric: RMSE
    "boosting_type": "gbdt",
    "num_leaves": 128,  # Increase the number of leaves (default is 31)
    "max_depth": 15,  # Limit tree depth to prevent over-complexity
    "learning_rate": 0.02,  # Small learning rate; increase num_boost_round accordingly
    "feature_fraction": 0.8,  # Subsample features to prevent overfitting and improve diversity
    "bagging_fraction": 0.8,  # Subsample data for similar reasons
    "bagging_freq": 1,  # Apply bagging_fraction at every round
    "lambda_l1": 0.1,  # L1 regularization to promote sparsity of leaf weights
    "lambda_l2": 1.0,  # L2 regularization to avoid overly large leaf weights
    "min_gain_to_split": 0.1,  # Do not split if the gain is too small
    "min_data_in_leaf": 50,  # Minimum data per leaf to avoid extreme splits
    "verbose": 1,  # Enable training log output (-1 to hide)
}

# Train the model
# Set a high number of rounds; early stopping will halt training when improvements stall
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=5000,  # Set a high number of rounds; early stopping will stop training if no improvement for 50 rounds
    valid_sets=[train_dataset, val_dataset],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),  # Stop if no improvement over 50 rounds
        lgb.log_evaluation(period=100),  # Log evaluation every 100 rounds
    ],
)

# Predictions
val_pred = model.predict(X_val)
test_pred = model.predict(test_data)


# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
mae = mean_absolute_error(y_val, val_pred)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")

# Display feature importance
feature_importance = pd.DataFrame(
    {
        "feature": [f"feature_{i}" for i in range(train_data.shape[1])],
        "importance": model.feature_importance(),
    }
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the model
model.save_model("tmp/lightgbm_score_predictor.txt")

# Predict on test data
test_pred = model.predict(test_data)

# Evaluation using the evaluate library
eval_pred = (test_pred, test_labels)  # Format: (predictions, labels)

# Calculate metrics
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

# Convert predictions and true labels to integers
preds = np.round(test_pred).astype(int)
labels = np.round(test_labels).astype(int)

# Compute each metric
precision = precision_metric.compute(
    predictions=preds, references=labels, average="macro"
)["precision"]
recall = recall_metric.compute(predictions=preds, references=labels, average="macro")[
    "recall"
]
f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

# Print detailed report and confusion matrix
report = classification_report(labels, preds)
cm = confusion_matrix(labels, preds)

print("\nTest Data Evaluation:")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)
