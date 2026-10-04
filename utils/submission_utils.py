"""Utilities for generating HW2P2 Kaggle submissions."""

import csv
import os
import random
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from tqdm import tqdm

PUBLIC_RATIO = 0.40
RANDOM_SEED = 11785

TASK_CLS = "classification"
TASK_VER = "verification"

SUBMISSION_COLUMNS = ["ID", "Task", "Prediction", "Usage"]
COMPONENT_COLUMNS = ["ID", "Label", "Usage"]


def get_public_indices(num_samples: int) -> set:
    """Return indices assigned to the public leaderboard split."""
    random.seed(RANDOM_SEED)
    num_public = int(num_samples * PUBLIC_RATIO)
    return set(random.sample(range(num_samples), num_public))


def build_usage_values(num_samples: int) -> List[str]:
    public_indices = get_public_indices(num_samples)
    return ["Public" if i in public_indices else "Private" for i in range(num_samples)]


# def load_cls_image_paths_file_order(cls_test_root: str) -> List[str]:
#     """
#     Load cls test image paths in labels.txt file order.

#     Submission IDs follow labels.txt line order (test_000001.jpg -> id 0),
#     not the label-sorted order used by ImageDataset.
#     """
#     labels_file = os.path.join(cls_test_root, "labels.txt")
#     images_dir = os.path.join(cls_test_root, "images")
#     image_paths = []

#     with open(labels_file, "r") as f:
#         for line in f:
#             img_name = line.strip().split()[0]
#             image_paths.append(os.path.join(images_dir, img_name))

#     return image_paths

def load_cls_image_paths_file_order(cls_test_root: str) -> List[str]:
    """
    Load classification test image paths in numeric filename order.

    Missing indices are allowed because duplicate images were removed
    during dataset deduplication.

    The deduplicated classification test set must contain exactly
    43,097 images.
    """
    images_dir = os.path.join(cls_test_root, "images")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_names = [
        name
        for name in os.listdir(images_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_names:
        raise RuntimeError(f"No images found in: {images_dir}")

    # Extract the original numeric index from test_XXXXXX.jpg
    try:
        numbers_and_names = [
            (
                int(os.path.splitext(name)[0].split("_")[-1]),
                name,
            )
            for name in image_names
        ]
    except ValueError as e:
        raise ValueError(
            "Classification test filenames must follow the format "
            "test_XXXXXX.jpg."
        ) from e

    # Keep inference order consistent with the original numeric file order.
    numbers_and_names.sort(key=lambda x: x[0])

    numbers = [number for number, _ in numbers_and_names]
    image_names = [name for _, name in numbers_and_names]

    # Exact expected size of the deduplicated test set.
    EXPECTED_NUM_IMAGES = 43097

    assert len(image_names) == EXPECTED_NUM_IMAGES, (
        f"Expected {EXPECTED_NUM_IMAGES:,} classification test images "
        f"after deduplication, but found {len(image_names):,}."
    )

    # Missing indices are allowed, but duplicate numeric IDs are not.
    assert len(numbers) == len(set(numbers)), (
        "Duplicate classification test image indices found."
    )

    print(f"Loaded {len(image_names):,} classification test images")
    print("First 3:", image_names[:3])
    print("Last 3: ", image_names[-3:])

    return [
        os.path.join(images_dir, name)
        for name in image_names
    ]

@torch.no_grad()
def run_cls_inference_file_order(
    model,
    cls_test_root: str,
    transform,
    device,
    batch_size: int = 64,
    logits_key: str = "cls_output",
) -> List[int]:
    # """Run classification inference in labels.txt file order."""
    """Run classification inference in sorted filename order."""
    model.eval()
    image_paths = load_cls_image_paths_file_order(cls_test_root)
    predictions = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Test (Cls)", leave=False):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            image = decode_image(path, mode="RGB") / 255.0
            images.append(transform(image))

        images = torch.stack(images).to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs[logits_key]
        batch_preds = logits.argmax(dim=1).cpu().tolist()
        predictions.extend(batch_preds)

    return predictions


@torch.no_grad()
def run_ver_inference(
    model,
    pair_dataloader: DataLoader,
    device,
    embedding_key: str = "embedding",
) -> List[float]:
    """Run verification inference and return cosine similarity scores."""
    model.eval()
    scores = []

    progress = tqdm(pair_dataloader, desc="Test (Veri)", dynamic_ncols=True, leave=False)

    for batch in progress:
        if len(batch) == 3:
            images1, images2, _ = batch
        else:
            images1, images2 = batch

        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)

        images = torch.cat([images1, images2], dim=0)
        outputs = model(images)
        feats = F.normalize(outputs[embedding_key], dim=1)

        feats1, feats2 = feats.chunk(2, dim=0)
        similarity = F.cosine_similarity(feats1, feats2, dim=1)
        scores.extend(similarity.cpu().numpy().tolist())

    return scores


def build_component_rows(predictions: Sequence, usage_values: Sequence[str]) -> List[dict]:
    return [
        {"ID": i, "Label": pred, "Usage": usage}
        for i, (pred, usage) in enumerate(zip(predictions, usage_values))
    ]


def build_merged_submission_rows(
    cls_predictions: Sequence[int],
    ver_scores: Sequence[float],
) -> List[dict]:
    """Build merged submission rows matching merged_solution.csv structure."""
    cls_usage = build_usage_values(len(cls_predictions))
    ver_usage = build_usage_values(len(ver_scores))
    cls_id_offset = len(cls_predictions)
    rows = []

    for i, pred in enumerate(cls_predictions):
        rows.append(
            {
                "ID": i,
                "Task": TASK_CLS,
                "Prediction": pred,
                "Usage": cls_usage[i],
            }
        )

    for i, score in enumerate(ver_scores):
        rows.append(
            {
                "ID": cls_id_offset + i,
                "Task": TASK_VER,
                "Prediction": score,
                "Usage": ver_usage[i],
            }
        )

    return rows


def write_component_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPONENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_submission_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_submission_bundle(
    cls_predictions: Sequence[int],
    ver_scores: Sequence[float],
    submission_path: str = "submission.csv",
    cls_component_path: str = "cls_submission.csv",
    ver_component_path: str = "ver_submission.csv",
) -> List[dict]:
    """Write component CSVs and merged submission CSV."""
    cls_usage = build_usage_values(len(cls_predictions))
    ver_usage = build_usage_values(len(ver_scores))

    cls_rows = build_component_rows(cls_predictions, cls_usage)
    ver_rows = build_component_rows(ver_scores, ver_usage)
    merged_rows = build_merged_submission_rows(cls_predictions, ver_scores)

    write_component_csv(cls_component_path, cls_rows)
    write_component_csv(ver_component_path, ver_rows)
    write_submission_csv(submission_path, merged_rows)

    return merged_rows
