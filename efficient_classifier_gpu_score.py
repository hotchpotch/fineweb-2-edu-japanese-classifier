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


class EfficientFineweb2EduJapaneseScoreClassifier:
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

        # モデルをメモリから解放
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()

        scores = np.array(predictions)
        return [(float(s) >= self.threshold, float(s)) for s in scores]


if __name__ == "__main__":
    import time

    import datasets

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
    classifier = EfficientFineweb2EduJapaneseScoreClassifier(
        model_path, show_progress=True, num_workers=4
    )

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
