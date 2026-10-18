"""
======================================================================
ISIC 2018 TASK 1
STEP 3A - FINE-TUNE SAM MASK DECODER WITH GROUND-TRUTH BOXES
======================================================================

Goal
----
Fine-tune ONLY SAM's mask decoder on the ISIC 2018 training set.

Frozen:
    - SAM image encoder
    - SAM prompt encoder

Trainable:
    - SAM mask decoder

Prompt:
    - Ground-truth bounding box around the lesion

Data:
    Training   : 2594 images
    Validation : 100 images
    Test       : 1000 images

Protocol
--------
1. Load pretrained SAM ViT-B.
2. Freeze image encoder and prompt encoder.
3. Train only mask decoder using ISIC training images.
4. Use ground-truth bounding boxes as prompts.
5. Use ground-truth masks as segmentation targets.
6. Select best checkpoint using validation mean IoU.
7. NEVER use the test set during training/model selection.
8. Load the best checkpoint.
9. Evaluate on the untouched 1000-image test set.
10. Compare:
       A. Pretrained SAM + GT box
       B. Fine-tuned SAM + GT box

Important
---------
The ground-truth box is intentionally used in this experiment.

This is Step 3A only.

A later experiment can test automatic/Otsu-generated boxes.
"""

# ======================================================================
# 0. Imports
# ======================================================================

import os
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# ======================================================================
# 1. Configuration
# ======================================================================

SEED = 42

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------

DATA_ROOT = Path(r"D:\Projects\ISIC2018")

TRAIN_IMG_DIR = DATA_ROOT / "ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = DATA_ROOT / "ISIC2018_Task1_Training_GroundTruth"

VAL_IMG_DIR = DATA_ROOT / "ISIC2018_Task1-2_Validation_Input"
VAL_MASK_DIR = DATA_ROOT / "ISIC2018_Task1_Validation_GroundTruth"

TEST_IMG_DIR = DATA_ROOT / "ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = DATA_ROOT / "ISIC2018_Task1_Test_GroundTruth"

SAM_CHECKPOINT = DATA_ROOT / "sam_vit_b_01ec64.pth"

OUTPUT_DIR = DATA_ROOT / "step3a_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "sam_vit_b_step3a_best.pth"
RESULTS_CSV = OUTPUT_DIR / "step3a_results.csv"

# ----------------------------------------------------------------------
# Model/training settings
# ----------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Start conservatively.
#
# Batch size 1 is the safest choice for GPU memory.
BATCH_SIZE = 1

NUM_EPOCHS = 5

# Learning rate for the mask decoder.
LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

# Loss:
# total = BCE_WEIGHT * BCE + DICE_WEIGHT * Dice
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# Number of DataLoader workers.
#
# 0 is the safest option on Windows.
NUM_WORKERS = 0

# Mixed precision reduces GPU memory usage and usually speeds up
# training on modern NVIDIA GPUs.
USE_AMP = torch.cuda.is_available()

# Print progress every N batches.
PRINT_EVERY = 100

# For a quick debugging run, change this to e.g. 50.
#
# None = use all images.
DEBUG_LIMIT = None

# Save a few qualitative examples after testing.
NUM_QUALITATIVE_EXAMPLES = 6


# ======================================================================
# 2. Reproducibility
# ======================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ======================================================================
# 3. Utility functions
# ======================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist:\n{path}"
        )


def list_image_files(directory):
    """
    Recursively find image files.

    This also handles nested folders, which was the source of the
    earlier dataset-discovery problem.
    """
    extensions = {".jpg", ".jpeg", ".png"}

    files = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    return sorted(files)


def build_image_mask_pairs(image_dir, mask_dir):
    """
    Build image/mask pairs by image ID.

    Example:
        ISIC_0000000.jpg
        ISIC_0000000_segmentation.png

    The function recursively searches the directories.
    """

    image_files = list_image_files(image_dir)

    mask_files = list(mask_dir.rglob("*_segmentation.png"))

    mask_lookup = {
        p.stem.replace("_segmentation", ""): p
        for p in mask_files
    }

    pairs = []

    for image_path in image_files:

        image_id = image_path.stem

        if image_id not in mask_lookup:
            continue

        mask_path = mask_lookup[image_id]

        pairs.append(
            {
                "id": image_id,
                "image": image_path,
                "mask": mask_path,
            }
        )

    return pairs


# ======================================================================
# 4. Dataset
# ======================================================================

class ISICSamDataset(Dataset):
    """
    ISIC dataset prepared for SAM training.

    Important:
        Images are NOT loaded into RAM during initialization.

    Only file paths are stored.

    Each image is loaded when DataLoader requests it.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        sam_image_size=1024,
        limit=None,
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.sam_image_size = sam_image_size

        self.transform = ResizeLongestSide(sam_image_size)

        self.pairs = build_image_mask_pairs(
            self.image_dir,
            self.mask_dir
        )

        if limit is not None:
            self.pairs = self.pairs[:limit]

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid image/mask pairs found.\n"
                f"Images: {self.image_dir}\n"
                f"Masks : {self.mask_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def bbox_from_mask(mask):
        """
        Return tight bounding box:

            [x_min, y_min, x_max, y_max]

        in ORIGINAL image coordinates.
        """

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        return np.array(
            [
                xs.min(),
                ys.min(),
                xs.max(),
                ys.max(),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index):

        item = self.pairs[index]

        image_path = item["image"]
        mask_path = item["mask"]

        # --------------------------------------------------------------
        # Load image
        # --------------------------------------------------------------

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        # --------------------------------------------------------------
        # Load ground-truth mask
        # --------------------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask:\n{mask_path}"
            )

        # Ensure binary.
        mask = (mask > 127).astype(np.uint8)

        # --------------------------------------------------------------
        # Ground-truth bounding box
        # --------------------------------------------------------------

        box = self.bbox_from_mask(mask)

        if box is None:
            raise RuntimeError(
                f"Empty ground-truth mask:\n{mask_path}"
            )

        # --------------------------------------------------------------
        # Resize image exactly according to SAM's ResizeLongestSide
        # --------------------------------------------------------------

        transformed_image = self.transform.apply_image(image)

        transformed_h, transformed_w = transformed_image.shape[:2]

        # --------------------------------------------------------------
        # Transform ground-truth mask using nearest-neighbor
        #
        # We do this manually instead of using apply_image because
        # segmentation masks must use nearest-neighbor interpolation.
        # --------------------------------------------------------------

        transformed_mask = cv2.resize(
            mask,
            (transformed_w, transformed_h),
            interpolation=cv2.INTER_NEAREST
        )

        # --------------------------------------------------------------
        # Transform box into SAM's coordinate system.
        # --------------------------------------------------------------

        transformed_box = self.transform.apply_boxes(
            box[None, :],
            (original_h, original_w)
        )[0].astype(np.float32)

        # --------------------------------------------------------------
        # Pad image to SAM's 1024 x 1024 input.
        # --------------------------------------------------------------

        padded_image = np.zeros(
            (
                self.sam_image_size,
                self.sam_image_size,
                3
            ),
            dtype=np.uint8
        )

        padded_image[
            :transformed_h,
            :transformed_w
        ] = transformed_image

        # --------------------------------------------------------------
        # Pad mask to 1024 x 1024.
        # --------------------------------------------------------------

        padded_mask = np.zeros(
            (
                self.sam_image_size,
                self.sam_image_size
            ),
            dtype=np.uint8
        )

        padded_mask[
            :transformed_h,
            :transformed_w
        ] = transformed_mask

        # --------------------------------------------------------------
        # Convert image to torch tensor.
        #
        # SAM preprocess() will perform normalization.
        # --------------------------------------------------------------

        image_tensor = torch.from_numpy(
            padded_image
        ).permute(2, 0, 1).float()

        # --------------------------------------------------------------
        # Ground truth at SAM decoder's 256 x 256 resolution.
        #
        # SAM's mask decoder outputs 256 x 256 low-resolution logits.
        # --------------------------------------------------------------

        target_lowres = cv2.resize(
            padded_mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        target_lowres = torch.from_numpy(
            target_lowres
        ).float()

        target_lowres = target_lowres.unsqueeze(0)

        # --------------------------------------------------------------
        # Original mask for final-resolution metrics.
        # --------------------------------------------------------------

        original_mask_tensor = torch.from_numpy(
            mask
        ).float()

        # --------------------------------------------------------------
        # Return everything needed by training/evaluation.
        # --------------------------------------------------------------

        return {
            "image": image_tensor,
            "box": torch.from_numpy(transformed_box),
            "target_lowres": target_lowres,
            "original_mask": original_mask_tensor,
            "original_size": (original_h, original_w),
            "input_size": (transformed_h, transformed_w),
            "id": item["id"],
        }


# ======================================================================
# 5. Collate function
# ======================================================================

def collate_fn(batch):

    images = torch.stack(
        [item["image"] for item in batch],
        dim=0
    )

    boxes = torch.stack(
        [item["box"] for item in batch],
        dim=0
    )

    targets_lowres = torch.stack(
        [item["target_lowres"] for item in batch],
        dim=0
    )

    return {
        "images": images,
        "boxes": boxes,
        "targets_lowres": targets_lowres,

        "original_masks": [
            item["original_mask"]
            for item in batch
        ],

        "original_sizes": [
            item["original_size"]
            for item in batch
        ],

        "input_sizes": [
            item["input_size"]
            for item in batch
        ],

        "ids": [
            item["id"]
            for item in batch
        ],
    }


# ======================================================================
# 6. Metrics
# ======================================================================

def iou_score(pred, gt, eps=1e-7):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    return (
        intersection + eps
    ) / (
        union + eps
    )


def dice_score(pred, gt, eps=1e-7):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    return (
        2 * intersection + eps
    ) / (
        pred.sum() + gt.sum() + eps
    )


# ======================================================================
# 7. Loss
# ======================================================================

def dice_loss_from_logits(logits, targets, eps=1e-6):

    probs = torch.sigmoid(logits)

    probs = probs.reshape(
        probs.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    intersection = (
        probs * targets
    ).sum(dim=1)

    denominator = (
        probs.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


def segmentation_loss(logits, targets):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss_from_logits(
        logits,
        targets
    )

    total = (
        BCE_WEIGHT * bce
        + DICE_WEIGHT * dice
    )

    return total, bce.detach(), dice.detach()


# ======================================================================
# 8. Freeze SAM except mask decoder
# ======================================================================

def freeze_sam_except_decoder(sam):

    # Freeze everything first.
    for param in sam.parameters():
        param.requires_grad = False

    # Unfreeze ONLY mask decoder.
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    # Frozen modules should remain in eval mode.
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()

    # Decoder is trainable.
    sam.mask_decoder.train()

    # Sanity check.
    trainable = sum(
        p.numel()
        for p in sam.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in sam.parameters()
    )

    print("\nSAM parameter configuration:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(
        f"  Trainable percentage : "
        f"{100.0 * trainable / total:.2f}%"
    )


# ======================================================================
# 9. Forward pass for training
# ======================================================================

def sam_decoder_forward(
    sam,
    images,
    boxes,
):
    """
    Run:

        image encoder      FROZEN
        prompt encoder     FROZEN
        mask decoder       TRAINABLE

    Returns:
        low-resolution mask logits
    """

    # --------------------------------------------------------------
    # Frozen image encoder
    # --------------------------------------------------------------

    with torch.no_grad():

        # SAM normalization.
        images_preprocessed = sam.preprocess(
            images
        )

        image_embeddings = sam.image_encoder(
            images_preprocessed
        )

        # ----------------------------------------------------------
        # Frozen prompt encoder
        # ----------------------------------------------------------

        sparse_embeddings, dense_embeddings = (
            sam.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None
            )
        )

        # Dense positional encoding is fixed.
        image_pe = sam.prompt_encoder.get_dense_pe()

    # --------------------------------------------------------------
    # TRAINABLE mask decoder
    #
    # Do NOT put this inside no_grad().
    # --------------------------------------------------------------

    low_res_masks, iou_predictions = (
        sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
    )

    return low_res_masks


# ======================================================================
# 10. Validation
# ======================================================================

@torch.no_grad()
def evaluate_validation(
    sam,
    loader,
):

    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()

    losses = []

    all_ious = []
    all_dices = []

    for batch in loader:

        images = batch["images"].to(
            DEVICE,
            non_blocking=True
        )

        boxes = batch["boxes"].to(
            DEVICE,
            non_blocking=True
        )

        targets = batch["targets_lowres"].to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------------
        # Forward
        # ----------------------------------------------------------

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            logits = sam_decoder_forward(
                sam,
                images,
                boxes
            )

            loss, _, _ = segmentation_loss(
                logits,
                targets
            )

        losses.append(
            loss.item()
        )

        # ----------------------------------------------------------
        # Convert predictions back to original image resolution.
        # ----------------------------------------------------------

        for i in range(logits.shape[0]):

            lowres = logits[i:i + 1]

            input_size = batch["input_sizes"][i]
            original_size = batch["original_sizes"][i]

            # SAM's official post-processing:
            # 256 -> 1024 -> remove padding -> original size
            prediction = sam.postprocess_masks(
                lowres,
                input_size=input_size,
                original_size=original_size
            )

            prediction = (
                prediction[0, 0]
                .sigmoid()
                .cpu()
                .numpy()
                > 0.5
            )

            gt = (
                batch["original_masks"][i]
                .numpy()
                > 0.5
            )

            all_ious.append(
                iou_score(prediction, gt)
            )

            all_dices.append(
                dice_score(prediction, gt)
            )

    return {
        "loss": float(np.mean(losses)),
        "mean_iou": float(np.mean(all_ious)),
        "std_iou": float(np.std(all_ious)),
        "mean_dice": float(np.mean(all_dices)),
        "std_dice": float(np.std(all_dices)),
    }


# ======================================================================
# 11. Training
# ======================================================================

def train_one_epoch(
    sam,
    loader,
    optimizer,
    scaler,
    epoch,
):

    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()

    running_loss = []
    running_bce = []
    running_dice = []

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):

        images = batch["images"].to(
            DEVICE,
            non_blocking=True
        )

        boxes = batch["boxes"].to(
            DEVICE,
            non_blocking=True
        )

        targets = batch["targets_lowres"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------------
        # Forward + loss
        # ----------------------------------------------------------

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            logits = sam_decoder_forward(
                sam,
                images,
                boxes
            )

            loss, bce, dice = segmentation_loss(
                logits,
                targets
            )

        # ----------------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------------

        if USE_AMP:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------

        running_loss.append(
            loss.item()
        )

        running_bce.append(
            bce.item()
        )

        running_dice.append(
            dice.item()
        )

        if (
            batch_idx == 0
            or (batch_idx + 1) % PRINT_EVERY == 0
            or (batch_idx + 1) == len(loader)
        ):

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss {np.mean(running_loss):.4f} | "
                f"BCE {np.mean(running_bce):.4f} | "
                f"DiceLoss {np.mean(running_dice):.4f} | "
                f"Time {elapsed / 60:.1f} min"
            )

    return {
        "loss": float(np.mean(running_loss)),
        "bce": float(np.mean(running_bce)),
        "dice_loss": float(np.mean(running_dice)),
    }


# ======================================================================
# 12. Final test evaluation
# ======================================================================

@torch.no_grad()
def evaluate_test(
    sam,
    loader,
    qualitative_dir=None,
):

    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()

    all_ious = []
    all_dices = []

    qualitative_predictions = []

    for batch_idx, batch in enumerate(loader):

        images = batch["images"].to(
            DEVICE,
            non_blocking=True
        )

        boxes = batch["boxes"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            logits = sam_decoder_forward(
                sam,
                images,
                boxes
            )

        # ----------------------------------------------------------
        # Evaluate each image at original resolution.
        # ----------------------------------------------------------

        for i in range(logits.shape[0]):

            lowres = logits[i:i + 1]

            input_size = batch["input_sizes"][i]
            original_size = batch["original_sizes"][i]

            prediction_logits = sam.postprocess_masks(
                lowres,
                input_size=input_size,
                original_size=original_size
            )

            prediction_prob = (
                prediction_logits[0, 0]
                .sigmoid()
                .cpu()
                .numpy()
            )

            prediction = (
                prediction_prob > 0.5
            ).astype(np.uint8)

            gt = (
                batch["original_masks"][i]
                .numpy()
                > 0.5
            ).astype(np.uint8)

            iou = iou_score(
                prediction,
                gt
            )

            dice = dice_score(
                prediction,
                gt
            )

            all_ious.append(iou)
            all_dices.append(dice)

            # ------------------------------------------------------
            # Save a few examples in memory for visualization.
            # ------------------------------------------------------

            if (
                len(qualitative_predictions)
                < NUM_QUALITATIVE_EXAMPLES
            ):

                image_np = (
                    images[i]
                    .detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

                qualitative_predictions.append(
                    {
                        "id": batch["ids"][i],
                        "image": image_np,
                        "prediction": prediction,
                        "gt": gt,
                        "iou": iou,
                        "dice": dice,
                    }
                )

    results = {
        "mean_iou": float(np.mean(all_ious)),
        "std_iou": float(np.std(all_ious)),
        "median_iou": float(np.median(all_ious)),
        "min_iou": float(np.min(all_ious)),
        "max_iou": float(np.max(all_ious)),

        "mean_dice": float(np.mean(all_dices)),
        "std_dice": float(np.std(all_dices)),
        "median_dice": float(np.median(all_dices)),
        "min_dice": float(np.min(all_dices)),
        "max_dice": float(np.max(all_dices)),

        "ious": all_ious,
        "dices": all_dices,

        "qualitative": qualitative_predictions,
    }

    return results


# ======================================================================
# 13. Save qualitative examples
# ======================================================================

def save_qualitative_examples(
    examples,
    output_dir,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, example in enumerate(examples):

        image = example["image"]
        prediction = example["prediction"]
        gt = example["gt"]

        # ----------------------------------------------------------
        # Save image
        # ----------------------------------------------------------

        cv2.imwrite(
            str(
                output_dir
                / f"{i:02d}_{example['id']}_image.png"
            ),
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------------
        # Save prediction
        # ----------------------------------------------------------

        cv2.imwrite(
            str(
                output_dir
                / f"{i:02d}_{example['id']}_prediction.png"
            ),
            (prediction * 255).astype(np.uint8)
        )

        # ----------------------------------------------------------
        # Save ground truth
        # ----------------------------------------------------------

        cv2.imwrite(
            str(
                output_dir
                / f"{i:02d}_{example['id']}_groundtruth.png"
            ),
            (gt * 255).astype(np.uint8)
        )

        print(
            f"  {example['id']} | "
            f"IoU={example['iou']:.4f} | "
            f"Dice={example['dice']:.4f}"
        )


# ======================================================================
# 14. Save CSV results
# ======================================================================

def save_results_csv(
    pretrained_results,
    finetuned_results,
):

    rows = [
        {
            "Method": "Pretrained SAM + GT box",
            "Mean IoU": pretrained_results["mean_iou"],
            "Std IoU": pretrained_results["std_iou"],
            "Median IoU": pretrained_results["median_iou"],
            "Min IoU": pretrained_results["min_iou"],
            "Max IoU": pretrained_results["max_iou"],
            "Mean Dice": pretrained_results["mean_dice"],
            "Std Dice": pretrained_results["std_dice"],
            "Median Dice": pretrained_results["median_dice"],
            "Min Dice": pretrained_results["min_dice"],
            "Max Dice": pretrained_results["max_dice"],
        },
        {
            "Method": "Fine-tuned SAM + GT box",
            "Mean IoU": finetuned_results["mean_iou"],
            "Std IoU": finetuned_results["std_iou"],
            "Median IoU": finetuned_results["median_iou"],
            "Min IoU": finetuned_results["min_iou"],
            "Max IoU": finetuned_results["max_iou"],
            "Mean Dice": finetuned_results["mean_dice"],
            "Std Dice": finetuned_results["std_dice"],
            "Median Dice": finetuned_results["median_dice"],
            "Min Dice": finetuned_results["min_dice"],
            "Max Dice": finetuned_results["max_dice"],
        },
    ]

    fieldnames = list(rows[0].keys())

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ======================================================================
# 15. Main
# ======================================================================

def main():

    print_header(
        "ISIC 2018 TASK 1\n"
        "STEP 3A - FINE-TUNE SAM MASK DECODER"
    )

    print("\nConfiguration:")
    print(f"  Device           : {DEVICE}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Epochs           : {NUM_EPOCHS}")
    print(f"  Learning rate    : {LEARNING_RATE}")
    print(f"  Weight decay     : {WEIGHT_DECAY}")
    print(f"  AMP              : {USE_AMP}")
    print(f"  DataLoader workers: {NUM_WORKERS}")

    # ------------------------------------------------------------------
    # Check paths
    # ------------------------------------------------------------------

    print_header("CHECKING DATASET")

    paths_to_check = [
        (TRAIN_IMG_DIR, "Training image directory"),
        (TRAIN_MASK_DIR, "Training mask directory"),
        (VAL_IMG_DIR, "Validation image directory"),
        (VAL_MASK_DIR, "Validation mask directory"),
        (TEST_IMG_DIR, "Test image directory"),
        (TEST_MASK_DIR, "Test mask directory"),
        (SAM_CHECKPOINT, "SAM checkpoint"),
    ]

    for path, description in paths_to_check:
        check_file(path, description)

    # ------------------------------------------------------------------
    # Create datasets
    # ------------------------------------------------------------------

    print_header("LOADING DATASET INDEXES")

    print("\nBuilding training dataset index...")

    train_dataset = ISICSamDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        limit=DEBUG_LIMIT
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print("\nBuilding validation dataset index...")

    val_dataset = ISICSamDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print("\nBuilding test dataset index...")

    test_dataset = ISICSamDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if DEBUG_LIMIT is None:

        assert len(train_dataset) == 2594, (
            f"Expected 2594 training samples, "
            f"found {len(train_dataset)}"
        )

        assert len(val_dataset) == 100, (
            f"Expected 100 validation samples, "
            f"found {len(val_dataset)}"
        )

        assert len(test_dataset) == 1000, (
            f"Expected 1000 test samples, "
            f"found {len(test_dataset)}"
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    print_header("CREATING DATALOADERS")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False
    )

    print(
        f"Training batches   : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    print(
        f"Test batches       : {len(test_loader)}"
    )

    # ------------------------------------------------------------------
    # Load SAM
    # ------------------------------------------------------------------

    print_header("LOADING PRETRAINED SAM")

    print(
        f"Checkpoint:\n{SAM_CHECKPOINT}"
    )

    print(
        "\nLoading SAM ViT-B..."
    )

    sam = sam_model_registry["vit_b"](
        checkpoint=str(SAM_CHECKPOINT)
    )

    sam.to(DEVICE)

    print("SAM loaded successfully.")

    # ------------------------------------------------------------------
    # Freeze everything except decoder
    # ------------------------------------------------------------------

    freeze_sam_except_decoder(sam)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )

    # ------------------------------------------------------------------
    # BASELINE: pretrained SAM + GT box
    #
    # This is important because it gives us the exact "before"
    # performance on the same test set.
    # ------------------------------------------------------------------

    print_header(
        "STEP 3A BASELINE\n"
        "PRETRAINED SAM + GROUND-TRUTH BOX"
    )

    pretrained_test_results = evaluate_test(
        sam,
        test_loader
    )

    print(
        "\nPretrained SAM + GT box:"
    )

    print(
        f"  Mean IoU   : "
        f"{pretrained_test_results['mean_iou']:.6f}"
    )

    print(
        f"  Std IoU    : "
        f"{pretrained_test_results['std_iou']:.6f}"
    )

    print(
        f"  Median IoU : "
        f"{pretrained_test_results['median_iou']:.6f}"
    )

    print(
        f"  Mean Dice  : "
        f"{pretrained_test_results['mean_dice']:.6f}"
    )

    print(
        f"  Std Dice   : "
        f"{pretrained_test_results['std_dice']:.6f}"
    )

    print(
        f"  Median Dice: "
        f"{pretrained_test_results['median_dice']:.6f}"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print_header(
        "FINE-TUNING MASK DECODER"
    )

    best_val_iou = -1.0
    best_epoch = -1

    history = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        train_results = train_one_epoch(
            sam,
            train_loader,
            optimizer,
            scaler,
            epoch
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        val_results = evaluate_validation(
            sam,
            val_loader
        )

        # --------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------

        scheduler.step(
            val_results["mean_iou"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            "\nEpoch summary:"
        )

        print(
            f"  Train loss    : "
            f"{train_results['loss']:.6f}"
        )

        print(
            f"  Train BCE     : "
            f"{train_results['bce']:.6f}"
        )

        print(
            f"  Train DiceLoss: "
            f"{train_results['dice_loss']:.6f}"
        )

        print(
            f"  Val loss      : "
            f"{val_results['loss']:.6f}"
        )

        print(
            f"  Val IoU       : "
            f"{val_results['mean_iou']:.6f}"
        )

        print(
            f"  Val Dice      : "
            f"{val_results['mean_dice']:.6f}"
        )

        print(
            f"  Learning rate : "
            f"{current_lr:.2e}"
        )

        print(
            f"  Epoch time    : "
            f"{elapsed / 60:.2f} min"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_results["loss"],
                "train_bce": train_results["bce"],
                "train_dice_loss": train_results["dice_loss"],
                "val_loss": val_results["loss"],
                "val_iou": val_results["mean_iou"],
                "val_iou_std": val_results["std_iou"],
                "val_dice": val_results["mean_dice"],
                "val_dice_std": val_results["std_dice"],
                "lr": current_lr,
            }
        )

        # --------------------------------------------------------------
        # Save best model
        # --------------------------------------------------------------

        if (
            val_results["mean_iou"]
            > best_val_iou
        ):

            best_val_iou = (
                val_results["mean_iou"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_type": "vit_b",

                    "mask_decoder_state_dict":
                        sam.mask_decoder.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_val_iou":
                        best_val_iou,

                    "val_results":
                        val_results,

                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "bce_weight": BCE_WEIGHT,
                        "dice_weight": DICE_WEIGHT,
                        "num_epochs": NUM_EPOCHS,
                    },
                },
                BEST_CHECKPOINT
            )

            print(
                "\n*** NEW BEST MODEL SAVED ***"
            )

            print(
                f"Best validation IoU: "
                f"{best_val_iou:.6f}"
            )

            print(
                f"Checkpoint:\n"
                f"{BEST_CHECKPOINT}"
            )

    # ------------------------------------------------------------------
    # Training finished
    # ------------------------------------------------------------------

    print_header(
        "TRAINING FINISHED"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation IoU: "
        f"{best_val_iou:.6f}"
    )

    # ------------------------------------------------------------------
    # Load best decoder
    # ------------------------------------------------------------------

    print_header(
        "LOADING BEST CHECKPOINT"
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE
    )

    sam.mask_decoder.load_state_dict(
        checkpoint["mask_decoder_state_dict"]
    )

    print(
        f"Loaded epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation IoU: "
        f"{checkpoint['best_val_iou']:.6f}"
    )

    # ------------------------------------------------------------------
    # FINAL TEST
    #
    # This is the first time we use the test set to report the final
    # fine-tuned result.
    # ------------------------------------------------------------------

    print_header(
        "FINAL TEST EVALUATION"
    )

    print(
        "Evaluating the best fine-tuned SAM "
        "on the untouched 1000-image test set..."
    )

    finetuned_test_results = evaluate_test(
        sam,
        test_loader,
        qualitative_dir=OUTPUT_DIR / "qualitative"
    )

    print(
        "\nFine-tuned SAM + GT box:"
    )

    print(
        f"  Mean IoU   : "
        f"{finetuned_test_results['mean_iou']:.6f}"
    )

    print(
        f"  Std IoU    : "
        f"{finetuned_test_results['std_iou']:.6f}"
    )

    print(
        f"  Median IoU : "
        f"{finetuned_test_results['median_iou']:.6f}"
    )

    print(
        f"  Min IoU    : "
        f"{finetuned_test_results['min_iou']:.6f}"
    )

    print(
        f"  Max IoU    : "
        f"{finetuned_test_results['max_iou']:.6f}"
    )

    print(
        f"  Mean Dice  : "
        f"{finetuned_test_results['mean_dice']:.6f}"
    )

    print(
        f"  Std Dice   : "
        f"{finetuned_test_results['std_dice']:.6f}"
    )

    print(
        f"  Median Dice: "
        f"{finetuned_test_results['median_dice']:.6f}"
    )

    print(
        f"  Min Dice   : "
        f"{finetuned_test_results['min_dice']:.6f}"
    )

    print(
        f"  Max Dice   : "
        f"{finetuned_test_results['max_dice']:.6f}"
    )

    # ------------------------------------------------------------------
    # Improvement over pretrained SAM
    # ------------------------------------------------------------------

    print_header(
        "STEP 3A COMPARISON"
    )

    pretrained_iou = (
        pretrained_test_results["mean_iou"]
    )

    finetuned_iou = (
        finetuned_test_results["mean_iou"]
    )

    pretrained_dice = (
        pretrained_test_results["mean_dice"]
    )

    finetuned_dice = (
        finetuned_test_results["mean_dice"]
    )

    iou_difference = (
        finetuned_iou
        - pretrained_iou
    )

    dice_difference = (
        finetuned_dice
        - pretrained_dice
    )

    print(
        "\n"
        "Method                         Mean IoU      Mean Dice"
    )

    print(
        "-" * 60
    )

    print(
        f"Pretrained SAM + GT box       "
        f"{pretrained_iou:.6f}      "
        f"{pretrained_dice:.6f}"
    )

    print(
        f"Fine-tuned SAM + GT box      "
        f"{finetuned_iou:.6f}      "
        f"{finetuned_dice:.6f}"
    )

    print(
        "-" * 60
    )

    print(
        f"Fine-tuning improvement       "
        f"{iou_difference:+.6f}      "
        f"{dice_difference:+.6f}"
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    save_results_csv(
        pretrained_test_results,
        finetuned_test_results
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_CSV}"
    )

    # ------------------------------------------------------------------
    # Save qualitative examples
    # ------------------------------------------------------------------

    print_header(
        "QUALITATIVE EXAMPLES"
    )

    qualitative_dir = (
        OUTPUT_DIR
        / "qualitative"
    )

    save_qualitative_examples(
        finetuned_test_results["qualitative"],
        qualitative_dir
    )

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------

    history_csv = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    if len(history) > 0:

        fieldnames = list(
            history[0].keys()
        )

        with open(
            history_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(history)

    print(
        f"\nTraining history saved to:\n"
        f"{history_csv}"
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print_header(
        "STEP 3A COMPLETE"
    )

    print(
        "\nExperiment:"
    )

    print(
        "  Pretrained SAM ViT-B"
    )

    print(
        "  + ground-truth bounding-box prompt"
    )

    print(
        "  + mask decoder fine-tuned on ISIC"
    )

    print(
        "  + image encoder FROZEN"
    )

    print(
        "  + prompt encoder FROZEN"
    )

    print(
        "\nData:"
    )

    print(
        f"  Training   : {len(train_dataset)} images"
    )

    print(
        f"  Validation : {len(val_dataset)} images"
    )

    print(
        f"  Test       : {len(test_dataset)} images"
    )

    print(
        "\nFinal test results:"
    )

    print(
        f"  Pretrained SAM IoU : "
        f"{pretrained_iou:.6f}"
    )

    print(
        f"  Fine-tuned SAM IoU : "
        f"{finetuned_iou:.6f}"
    )

    print(
        f"  IoU change         : "
        f"{iou_difference:+.6f}"
    )

    print(
        f"\n  Pretrained SAM Dice: "
        f"{pretrained_dice:.6f}"
    )

    print(
        f"  Fine-tuned SAM Dice: "
        f"{finetuned_dice:.6f}"
    )

    print(
        f"  Dice change        : "
        f"{dice_difference:+.6f}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\nStep 3A finished successfully."
    )


# ======================================================================
# 16. Entry point
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nTraining interrupted by user."
        )

    except Exception as e:

        import traceback

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            f"\nException type: "
            f"{type(e).__name__}"
        )

        print(
            f"Exception: {e}"
        )

        print(
            "\nFull traceback:"
        )

        traceback.print_exc()

        print(
            "\n"
            + "=" * 70
        )
