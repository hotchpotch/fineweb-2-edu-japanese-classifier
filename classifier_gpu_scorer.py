import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Fineweb2EduJapaneseScoreClassifier:
    def __init__(
        self,
        model_path: str,
        threshold: float = 2.5,
        batch_size: int = 512,
        device: str = "cuda",
        show_progress: bool = True,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Load model and tokenizer directly from path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _batch_tokenize(self, texts: List[str]) -> dict:
        """Tokenize texts in batches"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    def _batch_predict(self, texts: List[str]) -> np.ndarray:
        """Predict in batches with optional progress bar"""
        predictions = []
        iterator = range(0, len(texts), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]
                inputs = self._batch_tokenize(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze().cpu().numpy()
                predictions.extend(logits)

        return np.array(predictions)

    def predict(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Predict educational content scores for input texts
        Returns list of tuples (is_educational: bool, score: float)
        Raw score values are returned (not clipped)
        """
        # Get predictions
        scores = self._batch_predict(texts)

        # Convert to list of (bool, float) tuples
        return [(float(s) >= self.threshold, float(s)) for s in scores]


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
        stars = "★" * int(count / (max_count / 20))  # Scale to max 20 stars
        print(f"{bin_start:4.1f}-{bin_end:4.1f}: {stars} ({count:4d})")
    print("-" * 50)


if __name__ == "__main__":
    import datasets

    # Initialize classifier
    model_path = "hotchpotch/fineweb-2-edu-japanese-classifier"
    classifier = Fineweb2EduJapaneseScoreClassifier(model_path, show_progress=True)

    # Load test dataset
    print("Loading test dataset...")
    ds = datasets.load_dataset("hotchpotch/fineweb-2-edu-japanese-scores")
    test_texts = ds["test"]["text"]
    total_chars = sum(len(text) for text in test_texts)

    # Get predictions with timing
    start_time = time.time()
    predictions = classifier.predict(test_texts)
    elapsed_time = time.time() - start_time

    raw_scores = [score for _, score in predictions]

    # Clip scores for display and analysis
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
    print(f"\nScore statistics (after clipping):")
    print(f"  Mean: {np.mean(clipped_scores):.2f}")
    print(f"  Median: {np.median(clipped_scores):.2f}")
    print(f"  Std: {np.std(clipped_scores):.2f}")

    # Print score distribution
    print_score_matrix(clipped_scores)

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
