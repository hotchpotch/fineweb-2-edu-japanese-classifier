import os
import unicodedata

import evaluate
import fasttext
import fasttext.util
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def unicode_normalize(text):
    # Normalize with NFKC
    normalized_text = unicodedata.normalize("NFKC", text)
    return normalized_text


def prepare_data(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(lambda x: {"text": unicode_normalize(x["text"])}, num_proc=11)
    # Convert ['score'] of 5 to 4 in the dataset
    dataset = dataset.map(
        lambda x: {"score": 4 if x["score"] == 5 else x["score"]}, num_proc=11
    )
    return dataset


def create_fasttext_files(dataset, train_file, test_file):
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    # df["fasttext_label"] = "__label__" + df["score"].astype(str)
    train_df["fasttext_label"] = train_df["score"].apply(lambda x: "__label__" + str(x))
    test_df["fasttext_label"] = test_df["score"].apply(lambda x: "__label__" + str(x))

    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Writing {len(train_df)} training examples...")
    with open(train_file, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            f.write(f"{row['fasttext_label']} {row['text']}\n")

    print(f"Writing {len(test_df)} test examples...")
    with open(test_file, "w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            f.write(f"{row['fasttext_label']} {row['text']}\n")

    return train_df, test_df


def download_pretrained_vectors():
    print("Downloading pretrained vectors...")

    if os.path.exists("cc.ja.300.vec"):
        return "cc.ja.300.vec"
    fasttext.util.download_model("ja", if_exists="ignore")

    print("Converting binary model to vec format...")
    model = fasttext.load_model("cc.ja.300.bin")

    with open("cc.ja.300.vec", "w", encoding="utf-8", errors="ignore") as out_f:
        dim = model.get_dimension()
        words = model.get_words(on_unicode_error="ignore")
        print(f"{len(words)} {dim}", file=out_f)

        # Display progress bar with tqdm
        for word in tqdm(words, desc="Converting vectors", unit="word"):
            try:
                vec = model.get_word_vector(word)
                vec_str = " ".join([f"{x:.6f}" for x in vec])
                print(f"{word} {vec_str}", file=out_f)
            except UnicodeEncodeError:
                continue

    return "cc.ja.300.vec"


def train_model(train_file, test_file, pretrained_vectors, test_df):
    print("Training model...")
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.05,
        epoch=25,
        wordNgrams=2,
        verbose=2,
        minCount=5,
        dim=300,
        pretrainedVectors=pretrained_vectors,
    )

    print("\nEvaluating model...")
    result = model.test(test_file)
    print(f"Number of examples: {result[0]}")
    print(f"Precision: {result[1]:.3f}")
    print(f"Recall: {result[2]:.3f}")
    # カスタム評価指標の計算
    print("\nDetailed evaluation metrics:")
    metrics = evaluate_fasttext_model(model, test_df)
    print(metrics)
    return model


def evaluate_fasttext_model(model, test_df):
    # Get predictions
    predictions = []
    # Display progress with tqdm
    for text in tqdm(test_df["text"], desc="Predicting"):
        # Remove newlines from text and replace with spaces
        cleaned_text = text.replace("\n", " ").strip()
        pred = model.predict(cleaned_text)[0][
            0
        ]  # Get the label with the highest probability
        # Remove "__label__" and convert to integer
        pred_score = int(pred.replace("__label__", ""))
        predictions.append(pred_score)

    # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    true_labels = test_df["score"].values

    # Format input for compute_metrics
    # Make it a 2D array to match the format of logits
    eval_pred = (predictions.reshape(-1, 1), true_labels.reshape(-1, 1))

    # Calculate evaluation metrics
    metrics = compute_metrics(eval_pred)

    return metrics


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


def main():
    dataset_name = "hotchpotch/fineweb-2-edu-japanese-scores"  # Replace with the actual dataset name
    train_file = "tmp/fasttext_train.txt"
    test_file = "tmp/fasttext_test.txt"

    try:
        # Prepare pretrained vectors
        pretrained_vectors = download_pretrained_vectors()

        # Prepare dataset
        print("Preparing dataset...")
        dataset = prepare_data(dataset_name)

        # Create files for FastText
        train_df, test_df = create_fasttext_files(dataset, train_file, test_file)

        # Train and evaluate model
        model = train_model(train_file, test_file, pretrained_vectors, test_df)

        # Save model
        print("Saving model...")
        model.save_model("japanese_sentiment_model.bin")

        # Example prediction
        print("\nTesting prediction...")
        test_text = "日本語の文章ベクトルの性能を評価する JMTEB の結果は以下です。総合スコアでは mE5-small には若干及ばないまでも、タスクによっては勝っていたりしますし、他の日本語baseサイズbertモデルよりもスコアが高いこともあるぐらい、最低限実用できそうな性能が出ていますね。本当にそんなに性能が出るのか実際に学習させてみるまでは半信半疑でしたが、驚きです。"
        prediction = model.predict(test_text)
        print(f"Example prediction for '{test_text}':")
        print(f"Label: {prediction[0][0]}, Probability: {prediction[1][0]:.3f}")

    finally:
        # Delete temporary files
        for file in [train_file, test_file]:  # , "cc.ja.300.vec"]:
            if os.path.exists(file):
                os.remove(file)


if __name__ == "__main__":
    main()
