import unicodedata
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class JapaneseTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Unicode正規化
        normalized_text = unicodedata.normalize("NFKC", text)

        # トークン化
        encoding = self.tokenizer(
            normalized_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class Fineweb2EduJapaneseScoreClassifier:
    def __init__(
        self,
        model_path: str,
        threshold: float = 2.5,
        batch_size: int = 512,
        num_workers: int = 15,
        device: str = "cuda",
        show_progress: bool = True,
        max_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype = dtype
        self.show_progress = show_progress
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.max_length = max_length
        self.model = self._init_model()

        # トークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            "hotchpotch/fineweb-2-edu-japanese-classifier",
            num_labels=1,
            torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        return model

    def predict(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Predict educational content scores for input texts
        Returns list of tuples (is_educational: bool, score: float)
        """
        # カスタムデータセットの作成
        dataset = JapaneseTextDataset(texts, self.tokenizer, self.max_length)

        # データローダーの設定
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        model = self.model
        predictions = []
        iterator = (
            tqdm(dataloader, desc="Predicting") if self.show_progress else dataloader
        )

        with torch.no_grad():
            for batch in iterator:
                batch_ids = batch["input_ids"].to(self.device)
                batch_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = outputs.logits.squeeze().float().cpu().numpy()

                if len(logits.shape) == 0:
                    logits = [logits]

                predictions.extend(logits)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        scores = np.array(predictions)
        return [(float(s) >= self.threshold, float(s)) for s in scores]


if __name__ == "__main__":
    import time
    from typing import List, Tuple

    import datasets
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    def compute_detailed_metrics(pred_scores, true_scores, threshold: float = 2.5):
        scores = np.array(pred_scores)

        # Multi-class classification (0-4)
        multi_preds = np.clip(np.round(scores), 0, 4).astype(int)

        # Binary classification
        binary_preds = (scores >= threshold).astype(int)

        # Generate random labels for demonstration (replace with actual labels)
        # Note: In real usage, you would pass actual labels instead
        np.random.seed(42)  # For reproducible example
        multi_labels = np.array(true_scores)
        binary_labels = (multi_labels >= 2.5).astype(
            int
        )  # Example threshold for educational content

        # Multi-class metrics
        multi_report = classification_report(
            multi_labels, multi_preds, labels=[0, 1, 2, 3, 4], output_dict=True
        )
        multi_cm = confusion_matrix(multi_labels, multi_preds)

        # Binary metrics
        binary_report = classification_report(
            binary_labels,
            binary_preds,
            labels=[0, 1],
            target_names=["それ以外", "教育的"],
            output_dict=True,
        )
        binary_cm = confusion_matrix(binary_labels, binary_preds)

        # Print multi-class results
        print("Multi-class Classification Report:")
        print("-" * 80)
        df_multi = pd.DataFrame(multi_report).transpose()
        print(df_multi.round(4))

        print("\nMulti-class Confusion Matrix:")
        print("-" * 80)
        print(
            pd.DataFrame(
                multi_cm,
                index=[f"Actual {i}" for i in range(5)],
                columns=[f"Pred {i}" for i in range(5)],
            )
        )

        # Print binary results
        print("\nBinary Classification Report (それ以外/教育的):")
        print("-" * 80)
        df_binary = pd.DataFrame(binary_report).transpose()
        print(df_binary.round(4))

        print("\nBinary Confusion Matrix:")
        print("-" * 80)
        print(
            pd.DataFrame(
                binary_cm,
                index=["Actual それ以外", "Actual 教育的"],
                columns=["Pred それ以外", "Pred 教育的"],
            )
        )

        # Print key binary metrics
        print("\nKey Binary Metrics:")
        print("-" * 80)
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall", "F1-score", "Accuracy"],
                "Value": [
                    binary_report["教育的"]["precision"],
                    binary_report["教育的"]["recall"],
                    binary_report["教育的"]["f1-score"],
                    binary_report["accuracy"],
                ],
            }
        )
        print(metrics_df.round(4))

        return {
            "multi_report": multi_report,
            "multi_confusion_matrix": multi_cm,
            "binary_report": binary_report,
            "binary_confusion_matrix": binary_cm,
        }

    def print_score_matrix(scores: List[float], num_bins: int = 10):
        """Print the distribution of scores in a text-based matrix"""
        clipped_scores = np.clip(scores, 0, 4)
        hist, bins = np.histogram(clipped_scores, bins=num_bins, range=(0, 4))
        max_count = max(hist)

        print("\nScore Distribution (★ = approximately 10 samples):")
        print("-" * 50)
        for i in range(len(hist)):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            count = hist[i]
            stars = "★" * int(count / (max_count / 20))
            print(f"{bin_start:4.1f}-{bin_end:4.1f}: {stars} ({count:4d})")
        print("-" * 50)

    # Initialize classifier
    model_path = "hotchpotch/fineweb-2-edu-japanese-classifier"
    classifier = Fineweb2EduJapaneseScoreClassifier(
        model_path, show_progress=True, num_workers=4
    )

    # Load test dataset
    target = "test"
    print(f"Loading {target} dataset...")

    ds = datasets.load_dataset("hotchpotch/fineweb-2-edu-japanese-scores", split=target)
    test_texts = ds["text"]
    scores = ds["score"]
    total_chars = sum(len(text) for text in test_texts)

    # Get predictions with timing
    start_time = time.time()
    predictions = classifier.predict(test_texts)
    elapsed_time = time.time() - start_time

    raw_scores = [score for _, score in predictions]
    clipped_scores = np.clip(raw_scores, 0, 4)
    is_edu_count = sum(1 for score in clipped_scores if score >= classifier.threshold)

    # Print statistics
    print("\nPrediction Results:")
    print(f"Total samples: {len(predictions)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per text: {(elapsed_time * 1000 / len(predictions)):.2f} ms")
    print(f"Processing speed: {total_chars / elapsed_time:.1f} characters/second")
    print(
        f"Educational content ratio: {is_edu_count / len(predictions) * 100:.1f}% ({is_edu_count}/{len(predictions)})"
    )

    # Print score distribution
    print_score_matrix(clipped_scores)

    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(raw_scores, scores)

    # Print sample predictions
    print("\nSample Predictions:")

    # Sort by score
    samples = list(zip(test_texts, predictions))
    samples.sort(key=lambda x: x[1][1], reverse=True)

    print("\nHigh-scoring Examples (raw scores):")
    for text, (is_edu, score) in samples[:3]:
        print(
            f"Score {score:.2f} (Clipped: {min(max(score, 0), 4):.2f}, Educational: {is_edu}): {text[:200]}...\n"
        )

    print("\nLow-scoring Examples (raw scores):")
    for text, (is_edu, score) in samples[-3:]:
        print(
            f"Score {score:.2f} (Clipped: {min(max(score, 0), 4):.2f}, Educational: {is_edu}): {text[:200]}...\n"
        )
