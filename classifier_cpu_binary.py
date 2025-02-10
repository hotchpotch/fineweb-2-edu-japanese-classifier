import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import huggingface_hub
import lightgbm as lgb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


@dataclass
class Fineweb2EduJapaneseBinaryClassifier:
    def __init__(
        self,
        repo_id: str = "hotchpotch/fineweb-2-edu-japanese-classifier",
        model_filename: str = "lightgbm_binary_models/lightgbm_model.txt",
        embedding_model_name: str = "hotchpotch/static-embedding-japanese",
        threshold: float = 0.3,
        embedding_dim: int = 1024,
        batch_size: int = 1024,
        show_progress: bool = True,
    ):
        # Download LightGBM model from HuggingFace Hub
        self.model_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
        )
        self.threshold = threshold
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Load models
        self.gbm = lgb.Booster(model_file=self.model_path)
        self.embedder = SentenceTransformer(
            embedding_model_name, device="cpu", truncate_dim=embedding_dim
        )

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches with optional progress bar"""
        return self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress,
            batch_size=self.batch_size,
        )

    def _batch_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict in batches with optional progress bar"""
        predictions = []
        iterator = range(0, len(embeddings), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for i in iterator:
            batch = embeddings[i : i + self.batch_size]
            pred = self.gbm.predict(batch)
            predictions.extend(pred)
        return np.array(predictions)

    def predict(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Predict educational content probability for input texts
        Returns list of tuples (is_educational: bool, probability_score: float)
        """
        # Generate embeddings
        embeddings = self._batch_encode(texts)

        # Get prediction probabilities
        pred_probs = self._batch_predict(embeddings)

        # Convert to list of (bool, float) tuples
        return [(float(p) >= self.threshold, float(p)) for p in pred_probs]


if __name__ == "__main__":
    # Initialize classifier
    classifier = Fineweb2EduJapaneseBinaryClassifier(show_progress=True)

    # Load test dataset
    print("Loading test dataset...")
    ds = datasets.load_dataset("hotchpotch/fineweb-2-edu-japanese-scores")
    test_texts = ds["test"]["text"]

    # Measure inference time
    print(f"Running inference on {len(test_texts)} texts...")
    start_time = time.time()

    predictions = classifier.predict(test_texts)

    elapsed_time = time.time() - start_time

    # Calculate statistics
    total_chars = sum(len(text) for text in test_texts)
    is_edu_count = sum(1 for is_edu, _ in predictions if is_edu)

    # Print results
    print("\nInference Results:")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per text: {(elapsed_time / len(test_texts)) * 1000:.2f} ms")
    print(f"Processing speed: {total_chars / elapsed_time:.2f} characters/second")
    print(
        f"Educational content ratio: {is_edu_count / len(predictions) * 100:.1f}% ({is_edu_count}/{len(predictions)})"
    )

    # Print sample predictions
    print("\nSample Predictions:")

    # Sort by score
    samples = list(zip(test_texts, predictions))
    samples.sort(key=lambda x: x[1][1], reverse=True)

    print("\nHigh-scoring Examples:")
    for text, (is_edu, score) in samples[:5]:
        print(f"Score {score:.3f} (Educational: {is_edu}): {text[:200]}...\n")

    print("\nLow-scoring Examples:")
    for text, (is_edu, score) in samples[-5:]:
        print(f"Score {score:.3f} (Educational: {is_edu}): {text[:200]}...\n")
