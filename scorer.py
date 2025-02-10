"""
Script to calculate dataset scores using the Deepseek API.
"""

import argparse
import asyncio
import hashlib
import json
import os
import warnings
from pathlib import Path

import datasets
import tenacity
from openai import AsyncOpenAI
from tqdm import tqdm

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--base_dataset_dir", type=str)
argparser.add_argument(
    "-o", "--output_dataset_dir", type=str, default="tmp/scored_dataset"
)
argparser.add_argument("-p", "--parallel", type=int, default=10)
argparser.add_argument("-n", "--batch_size", type=int, default=30)
argparser.add_argument("-m", "--max_retries", type=int, default=2)
argparser.add_argument("-k", "--api_key", type=str, default=DEEPSEEK_API_KEY)
argparser.add_argument("-u", "--base_url", type=str, default="https://api.deepseek.com")
argparser.add_argument("--debug", action="store_true")
args = argparser.parse_args()

if not args.api_key:
    raise ValueError("API key is required")

A_CLIENT = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_PATH = PROJECT_ROOT / "cache/scorer"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

PROMPT_PATH = PROJECT_ROOT / "scorer_prompt.txt"
PROMPT = PROMPT_PATH.read_text()

OUTPUT_DS_PATH = Path(args.output_dataset_dir)
if args.debug:
    OUTPUT_DS_PATH = PROJECT_ROOT / "tmp/debug_scored_dataset"


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def is_exist_cache(id: str):
    sha1_id = sha1(id)
    return (CACHE_PATH / sha1_id).exists()


def load_cache(id: str) -> int:
    sha1_id = sha1(id)
    with open(CACHE_PATH / sha1_id, "r") as f:
        score_text = f.read()
    return int(score_text)


def save_cache(id: str, score: int):
    sha1_id = sha1(id)
    with open(CACHE_PATH / sha1_id, "w") as f:
        f.write(str(score))


def log_error(batch_ids: list[str], error: Exception):
    warnings.warn(f"Error occurred during processing: {error}")


@tenacity.retry(
    stop=tenacity.stop_after_attempt(args.max_retries),
    wait=tenacity.wait_fixed(5),
    retry=tenacity.retry_if_exception_type(Exception),
)
async def get_scores(texts: list[str], prompt=PROMPT):
    text_dict = {i + 1: text for i, text in enumerate(texts)}
    text_json = json.dumps(text_dict, ensure_ascii=False)

    response = await A_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_json},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        stream=False,
    )
    res = response.choices[0].message.content
    data = json.loads(res)
    result: list[tuple[str, int]] = []
    for i in range(1, len(data) + 1):
        result.append((text_dict[i], data[str(i)]))
    return result


async def process_batch(target_data_batch: list[tuple[str, str]]):
    try:
        results = await get_scores([x[1] for x in target_data_batch])
        for (id, text), result in zip(target_data_batch, results):
            text, score = result
            save_cache(id, score)
    except Exception as e:
        batch_ids = [x[0] for x in target_data_batch]
        log_error(batch_ids, e)


async def process_batches(batches: list[list[tuple[str, str]]]):
    tasks = [process_batch(batch) for batch in batches]
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    target_ds = datasets.load_from_disk(args.base_dataset_dir)
    print(f"Loaded {len(target_ds)} samples")
    datasets.disable_caching()
    score_target_ds = target_ds.filter(
        lambda x: not is_exist_cache(x["id"]),
        num_proc=4,
    )
    print(f"score target ds: {len(score_target_ds)} samples")
    target_text = score_target_ds["text"]
    target_ids = score_target_ds["id"]
    target_data = list(zip(target_ids, target_text))
    if args.debug:
        target_data = target_data[:5000]
        print(f"debug mode: {len(target_data)} samples")
    print("Start scoring...")

    n_parallel = args.parallel
    batch_size = args.batch_size

    target_data_batch = [
        target_data[i : i + batch_size] for i in range(0, len(target_data), batch_size)
    ]

    for i in tqdm(range(0, len(target_data_batch), n_parallel)):
        batch = target_data_batch[i : i + n_parallel]
        await process_batches(batch)

    if args.debug:
        for id, text in target_data:
            if is_exist_cache(id):
                print(f"{load_cache(id)}: {text}")
                print("-----------------------------\n\n")
            # else:
            #     print(f"Failed to process: {id}: {text}")


if __name__ == "__main__":
    asyncio.run(main())
