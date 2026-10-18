"""
ISIC 2018 Task 1 - STEP 2
=========================

Zero-shot SAM evaluation using two box-prompting strategies:

    1. SAM + Ground-Truth / Oracle Box
    2. SAM + Automatic Otsu-Derived Box

IMPORTANT:
    This script is designed to work with the memory-efficient
    Step 1 implementation.

    It does NOT require:
        test_images
        test_masks
        classical_preds

    to be stored in RAM.

Instead, the test set is processed one image at a time.

Experiment
----------

                    ISIC image
                        |
              +---------+---------+
              |                   |
              v                   v
         Ground truth          Otsu
             mask                |
              |                   v
              v              Largest component
        Oracle bounding             |
             box                    v
              |              Automatic bounding
              |                    box
              |                   |
              v                   v
          SAM ViT-B            SAM ViT-B
              |                   |
              v                   v
        SAM prediction      SAM prediction
              |                   |
              +---------+---------+
                        |
                        v
                   IoU / Dice


The oracle-box experiment uses the ground-truth mask ONLY to
construct the prompt. It must therefore be described as an
oracle/reference experiment rather than a deployable pipeline.

The automatic-box experiment uses only the image itself:
    image -> Otsu -> box -> SAM

No ground-truth information is used to construct the automatic box.

Requirements
------------

pip install torch torchvision opencv-python numpy matplotlib pandas

SAM installation:

pip install git+https://github.com/facebookresearch/segment-anything.git

SAM ViT-B checkpoint:

sam_vit_b_01ec64.pth

Place the checkpoint in:

D:\\Projects\\ISIC2018\\sam_vit_b_01ec64.pth

or in the same directory as this script.
"""

# ===============================================================
# 0. Imports
# ===============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from segment_anything import (
    sam_model_registry,
    SamPredictor
)


# ===============================================================
# 1. Configuration
# ===============================================================

SEED = 42
np.random.seed(SEED)

IMG_SIZE = 256

NUM_VISUAL_EXAMPLES = 3

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ===============================================================
# 2. Project directory
# ===============================================================

DATA_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ===============================================================
# 3. Dataset directories
# ===============================================================

TEST_IMG_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1-2_Test_Input"
)

TEST_MASK_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1_Test_GroundTruth"
)


# ===============================================================
# 4. SAM checkpoint
# ===============================================================

SAM_CHECKPOINT = os.path.join(
    DATA_ROOT,
    "sam_vit_b_01ec64.pth"
)


# ===============================================================
# 5. File discovery
# ===============================================================

def find_image_files(directory):
    """
    Recursively find image files.

    Recursive search is intentional because the ISIC folders may
    contain an additional nested directory.
    """

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    files = []

    if not os.path.isdir(directory):
        return files

    for root, dirs, filenames in os.walk(directory):

        for filename in filenames:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in valid_extensions:

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return sorted(files)


def find_mask_files(directory):
    """
    Recursively find ISIC segmentation masks.
    """

    files = []

    if not os.path.isdir(directory):
        return files

    for root, dirs, filenames in os.walk(directory):

        for filename in filenames:

            if filename.lower().endswith(
                "_segmentation.png"
            ):

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return sorted(files)


# ===============================================================
# 6. Mask lookup
# ===============================================================

def create_mask_lookup(mask_dir):
    """
    Create:

        image_id -> mask path
    """

    mask_paths = find_mask_files(
        mask_dir
    )

    mask_dict = {}

    for mask_path in mask_paths:

        filename = os.path.basename(
            mask_path
        )

        image_id = filename.replace(
            "_segmentation.png",
            ""
        )

        image_id = image_id.replace(
            "_segmentation.PNG",
            ""
        )

        mask_dict[image_id] = mask_path

    return mask_dict


# ===============================================================
# 7. Streaming test-set iterator
# ===============================================================

def iter_test_pairs():
    """
    Yield one test image and ground-truth mask at a time.

    Nothing is accumulated in RAM.
    """

    image_paths = find_image_files(
        TEST_IMG_DIR
    )

    mask_dict = create_mask_lookup(
        TEST_MASK_DIR
    )

    print(
        f"Found {len(image_paths)} test images."
    )

    print(
        f"Found {len(mask_dict)} test masks."
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            "\nNo test images were found in:\n"
            f"{TEST_IMG_DIR}"
        )

    if len(mask_dict) == 0:

        raise RuntimeError(
            "\nNo test masks were found in:\n"
            f"{TEST_MASK_DIR}"
        )

    missing_masks = 0

    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )

        image_id = os.path.splitext(
            filename
        )[0]

        mask_path = mask_dict.get(
            image_id
        )

        if mask_path is None:

            missing_masks += 1

            continue

        # -------------------------------------------------------
        # Read image
        # -------------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"\nWARNING: Could not read:"
                f"\n{image_path}"
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------------
        # Read mask
        # -------------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            print(
                f"\nWARNING: Could not read:"
                f"\n{mask_path}"
            )

            continue

        # -------------------------------------------------------
        # Resize
        #
        # Same preprocessing as Step 1.
        # -------------------------------------------------------

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------------
        # Binarize GT
        # -------------------------------------------------------

        mask = (
            mask > 127
        ).astype(np.uint8)

        yield image, mask, image_id

    if missing_masks > 0:

        print(
            f"\nWARNING:"
            f"\n{missing_masks} test images "
            f"had no matching mask."
        )


# ===============================================================
# 8. Metrics
# ===============================================================

def iou_score(pred, gt, eps=1e-7):
    """
    Intersection over Union.
    """

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
    """
    Dice coefficient.
    """

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    return (
        2 * intersection + eps
    ) / (
        pred.sum()
        + gt.sum()
        + eps
    )


# ===============================================================
# 9. Classical Otsu segmentation
# ===============================================================

def classical_segment(img):
    """
    Same classical baseline used in Step 1.

    RGB
      |
      v
    grayscale
      |
      v
    blackhat hair detection
      |
      v
    inpainting
      |
      v
    Otsu threshold
      |
      v
    morphology
      |
      v
    largest connected component
      |
      v
    binary mask
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # -----------------------------------------------------------
    # Hair detection
    # -----------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------------
    # Hair removal
    # -----------------------------------------------------------

    gray_clean = cv2.inpaint(
        gray,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    # -----------------------------------------------------------
    # Otsu
    # -----------------------------------------------------------

    _, binary = cv2.threshold(
        gray_clean,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    # -----------------------------------------------------------
    # Morphological cleanup
    # -----------------------------------------------------------

    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel2
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel2
    )

    # -----------------------------------------------------------
    # Largest component
    # -----------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary
        )
    )

    if num_labels > 1:

        largest = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        binary = (
            labels == largest
        ).astype(np.uint8)

    else:

        binary = (
            binary > 0
        ).astype(np.uint8)

    return binary


# ===============================================================
# 10. Bounding-box utilities
# ===============================================================

def bbox_from_mask(mask):
    """
    Return tight bounding box:

        [x_min, y_min, x_max, y_max]

    Coordinates are returned as float32 because SAM expects
    floating-point box coordinates.

    Returns None if mask is empty.
    """

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    return np.array(
        [
            xs.min(),
            ys.min(),
            xs.max(),
            ys.max()
        ],
        dtype=np.float32
    )


def get_oracle_box(gt_mask):
    """
    Ground-truth/oracle box.

    IMPORTANT:
        This uses the ground-truth mask and therefore is NOT
        deployable. It is an upper-bound/reference experiment
        for SAM given accurate localization.
    """

    return bbox_from_mask(
        gt_mask
    )


def get_automatic_box(
    classical_pred,
    img_shape
):
    """
    Generate a box using only the Otsu prediction.

    If Otsu produces an empty mask, use a centered fallback box.

    Returns:
        box
        used_fallback
    """

    box = bbox_from_mask(
        classical_pred
    )

    if box is not None:

        return box, False

    # -----------------------------------------------------------
    # Fallback box
    # -----------------------------------------------------------

    h, w = img_shape[:2]

    box = np.array(
        [
            int(0.20 * w),
            int(0.20 * h),
            int(0.80 * w),
            int(0.80 * h)
        ],
        dtype=np.float32
    )

    return box, True


# ===============================================================
# 11. Load SAM
# ===============================================================

def load_sam():
    """
    Load pretrained SAM ViT-B.
    """

    print("\n")
    print("=" * 70)
    print("LOADING SAM")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint:\n{SAM_CHECKPOINT}"
    )

    if not os.path.isfile(
        SAM_CHECKPOINT
    ):

        raise FileNotFoundError(
            "\nSAM checkpoint not found:\n"
            f"{SAM_CHECKPOINT}\n\n"
            "Download sam_vit_b_01ec64.pth "
            "and place it in the project directory."
        )

    print(
        "\nLoading SAM ViT-B..."
    )

    sam = sam_model_registry[
        "vit_b"
    ](
        checkpoint=SAM_CHECKPOINT
    )

    sam.to(
        device=DEVICE
    )

    sam.eval()

    predictor = SamPredictor(
        sam
    )

    print(
        "SAM loaded successfully."
    )

    print("=" * 70)

    return predictor


# ===============================================================
# 12. Run SAM with a box
# ===============================================================

def sam_predict(
    predictor,
    image,
    box
):
    """
    Run SAM using one box prompt.

    Returns:
        binary prediction
        SAM confidence score
    """

    predictor.set_image(
        image
    )

    # SAM expects XYXY coordinates.
    box = np.asarray(
        box,
        dtype=np.float32
    )

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False
    )

    prediction = (
        masks[0]
        .astype(np.uint8)
    )

    score = float(
        scores[0]
    )

    return prediction, score


# ===============================================================
# 13. Maintain best/worst examples
# ===============================================================

def update_best_examples(
    examples,
    example,
    number=3
):
    """
    Keep only the highest-IoU examples.
    """

    examples.append(
        example
    )

    examples.sort(
        key=lambda x: x["iou"],
        reverse=True
    )

    del examples[number:]


def update_worst_examples(
    examples,
    example,
    number=3
):
    """
    Keep only the lowest-IoU examples.
    """

    examples.append(
        example
    )

    examples.sort(
        key=lambda x: x["iou"]
    )

    del examples[number:]


# ===============================================================
# 14. Visualization
# ===============================================================

def show_examples(
    examples,
    title
):
    """
    Visualize selected examples.

    Rows:

        1. Original image
        2. SAM prediction
        3. Ground truth

    For automatic examples, the Otsu box can also be displayed.
    """

    if len(examples) == 0:

        print(
            f"\nNo examples available for:"
            f"\n{title}"
        )

        return

    n = len(examples)

    fig, axes = plt.subplots(
        3,
        n,
        figsize=(4 * n, 10)
    )

    if n == 1:

        axes = np.asarray(
            axes
        ).reshape(
            3,
            1
        )

    for col, example in enumerate(
        examples
    ):

        image = example[
            "image"
        ]

        prediction = example[
            "prediction"
        ]

        ground_truth = example[
            "ground_truth"
        ]

        image_id = example[
            "id"
        ]

        iou = example[
            "iou"
        ]

        dice = example[
            "dice"
        ]

        box = example.get(
            "box"
        )

        # -------------------------------------------------------
        # Original image
        # -------------------------------------------------------

        axes[
            0,
            col
        ].imshow(
            image
        )

        # Draw box if available

        if box is not None:

            x1, y1, x2, y2 = (
                box
            )

            rectangle = plt.Rectangle(
                (
                    x1,
                    y1
                ),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2
            )

            axes[
                0,
                col
            ].add_patch(
                rectangle
            )

        axes[
            0,
            col
        ].set_title(
            f"{image_id}\n"
            f"IoU={iou:.4f}, Dice={dice:.4f}"
        )

        # -------------------------------------------------------
        # SAM prediction
        # -------------------------------------------------------

        axes[
            1,
            col
        ].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[
            1,
            col
        ].set_title(
            "SAM Prediction"
        )

        # -------------------------------------------------------
        # Ground truth
        # -------------------------------------------------------

        axes[
            2,
            col
        ].imshow(
            ground_truth,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[
            2,
            col
        ].set_title(
            "Ground Truth"
        )

        for row in range(3):

            axes[
                row,
                col
            ].axis(
                "off"
            )

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    filename = (
        title
        .lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("/", "_")
    )

    output_path = os.path.join(
        DATA_ROOT,
        f"{filename}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nSaved:"
        f"\n{output_path}"
    )

    plt.show()


# ===============================================================
# 15. Main experiment
# ===============================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "ISIC 2018 TASK 1 - STEP 2"
    )

    print(
        "ZERO-SHOT SAM EVALUATION"
    )

    print("=" * 70)

    # -----------------------------------------------------------
    # Dataset information
    # -----------------------------------------------------------

    print(
        "\nTest image directory:"
        f"\n{TEST_IMG_DIR}"
    )

    print(
        "\nTest mask directory:"
        f"\n{TEST_MASK_DIR}"
    )

    image_count = len(
        find_image_files(
            TEST_IMG_DIR
        )
    )

    mask_count = len(
        find_mask_files(
            TEST_MASK_DIR
        )
    )

    print(
        f"\nTest images: {image_count}"
    )

    print(
        f"Test masks : {mask_count}"
    )

    if image_count == 0:

        raise RuntimeError(
            "\nNo test images found."
        )

    if mask_count == 0:

        raise RuntimeError(
            "\nNo test masks found."
        )

    # -----------------------------------------------------------
    # Load SAM
    # -----------------------------------------------------------

    predictor = load_sam()

    # -----------------------------------------------------------
    # Result arrays
    # -----------------------------------------------------------

    oracle_ious = []
    oracle_dices = []

    automatic_ious = []
    automatic_dices = []

    oracle_sam_scores = []
    automatic_sam_scores = []

    # -----------------------------------------------------------
    # Track fallback usage
    # -----------------------------------------------------------

    fallback_count = 0

    # -----------------------------------------------------------
    # Best/worst examples
    # -----------------------------------------------------------

    oracle_best = []
    oracle_worst = []

    automatic_best = []
    automatic_worst = []

    # -----------------------------------------------------------
    # Process test images one at a time
    # -----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "RUNNING ZERO-SHOT SAM"
    )
    print("=" * 70)

    processed = 0

    for image, gt, image_id in iter_test_pairs():

        processed += 1

        # =======================================================
        # A. ORACLE BOX
        # =======================================================

        oracle_box = get_oracle_box(
            gt
        )

        if oracle_box is None:

            print(
                f"\nWARNING: Empty GT mask:"
                f" {image_id}"
            )

            continue

        oracle_pred, oracle_score = (
            sam_predict(
                predictor,
                image,
                oracle_box
            )
        )

        oracle_iou = iou_score(
            oracle_pred,
            gt
        )

        oracle_dice = dice_score(
            oracle_pred,
            gt
        )

        oracle_ious.append(
            oracle_iou
        )

        oracle_dices.append(
            oracle_dice
        )

        oracle_sam_scores.append(
            oracle_score
        )

        # =======================================================
        # B. AUTOMATIC OTSU BOX
        # =======================================================

        classical_pred = classical_segment(
            image
        )

        automatic_box, used_fallback = (
            get_automatic_box(
                classical_pred,
                image.shape
            )
        )

        if used_fallback:

            fallback_count += 1

        automatic_pred, automatic_score = (
            sam_predict(
                predictor,
                image,
                automatic_box
            )
        )

        automatic_iou = iou_score(
            automatic_pred,
            gt
        )

        automatic_dice = dice_score(
            automatic_pred,
            gt
        )

        automatic_ious.append(
            automatic_iou
        )

        automatic_dices.append(
            automatic_dice
        )

        automatic_sam_scores.append(
            automatic_score
        )

        # =======================================================
        # C. Save examples
        # =======================================================

        oracle_example = {
            "image": image.copy(),
            "prediction": oracle_pred.copy(),
            "ground_truth": gt.copy(),
            "id": image_id,
            "iou": oracle_iou,
            "dice": oracle_dice,
            "box": oracle_box.copy()
        }

        automatic_example = {
            "image": image.copy(),
            "prediction": automatic_pred.copy(),
            "ground_truth": gt.copy(),
            "id": image_id,
            "iou": automatic_iou,
            "dice": automatic_dice,
            "box": automatic_box.copy()
        }

        update_best_examples(
            oracle_best,
            oracle_example,
            NUM_VISUAL_EXAMPLES
        )

        update_worst_examples(
            oracle_worst,
            oracle_example,
            NUM_VISUAL_EXAMPLES
        )

        update_best_examples(
            automatic_best,
            automatic_example,
            NUM_VISUAL_EXAMPLES
        )

        update_worst_examples(
            automatic_worst,
            automatic_example,
            NUM_VISUAL_EXAMPLES
        )

        # =======================================================
        # Progress
        # =======================================================

        if (
            processed % 25 == 0
            or processed == image_count
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{image_count}"
            )

    # ===========================================================
    # Convert results
    # ===========================================================

    oracle_ious = np.asarray(
        oracle_ious,
        dtype=np.float32
    )

    oracle_dices = np.asarray(
        oracle_dices,
        dtype=np.float32
    )

    automatic_ious = np.asarray(
        automatic_ious,
        dtype=np.float32
    )

    automatic_dices = np.asarray(
        automatic_dices,
        dtype=np.float32
    )

    oracle_sam_scores = np.asarray(
        oracle_sam_scores,
        dtype=np.float32
    )

    automatic_sam_scores = np.asarray(
        automatic_sam_scores,
        dtype=np.float32
    )

    # ===========================================================
    # Safety checks
    # ===========================================================

    if len(oracle_ious) == 0:

        raise RuntimeError(
            "\nNo valid oracle SAM results."
        )

    if len(automatic_ious) == 0:

        raise RuntimeError(
            "\nNo valid automatic SAM results."
        )

    # ===========================================================
    # Final statistics
    # ===========================================================

    oracle_mean_iou = (
        oracle_ious.mean()
    )

    oracle_std_iou = (
        oracle_ious.std()
    )

    oracle_mean_dice = (
        oracle_dices.mean()
    )

    oracle_std_dice = (
        oracle_dices.std()
    )

    automatic_mean_iou = (
        automatic_ious.mean()
    )

    automatic_std_iou = (
        automatic_ious.std()
    )

    automatic_mean_dice = (
        automatic_dices.mean()
    )

    automatic_std_dice = (
        automatic_dices.std()
    )

    # ===========================================================
    # Print SAM results
    # ===========================================================

    print("\n")
    print("=" * 70)
    print(
        "STEP 2 - SAM RESULTS"
    )
    print("=" * 70)

    print(
        f"\nProcessed samples: "
        f"{len(oracle_ious)}"
    )

    print(
        "\nSAM with ground-truth/oracle box:"
    )

    print(
        f"  Mean IoU  : "
        f"{oracle_mean_iou:.4f}"
    )

    print(
        f"  Std IoU   : "
        f"{oracle_std_iou:.4f}"
    )

    print(
        f"  Mean Dice : "
        f"{oracle_mean_dice:.4f}"
    )

    print(
        f"  Std Dice  : "
        f"{oracle_std_dice:.4f}"
    )

    print(
        "\nSAM with automatic Otsu-derived box:"
    )

    print(
        f"  Mean IoU  : "
        f"{automatic_mean_iou:.4f}"
    )

    print(
        f"  Std IoU   : "
        f"{automatic_std_iou:.4f}"
    )

    print(
        f"  Mean Dice : "
        f"{automatic_mean_dice:.4f}"
    )

    print(
        f"  Std Dice  : "
        f"{automatic_std_dice:.4f}"
    )

    print(
        "\nAutomatic-box fallback usage:"
    )

    print(
        f"  Fallbacks : "
        f"{fallback_count}"
    )

    print(
        f"  Percentage: "
        f"{100 * fallback_count / len(automatic_ious):.2f}%"
    )

    print("=" * 70)

    # ===========================================================
    # Oracle vs automatic difference
    # ===========================================================

    iou_difference = (
        oracle_mean_iou
        - automatic_mean_iou
    )

    dice_difference = (
        oracle_mean_dice
        - automatic_mean_dice
    )

    print("\n")
    print("=" * 70)
    print(
        "ORACLE vs AUTOMATIC PROMPT DIFFERENCE"
    )
    print("=" * 70)

    print(
        f"IoU difference  : "
        f"{iou_difference:+.4f}"
    )

    print(
        f"Dice difference : "
        f"{dice_difference:+.4f}"
    )

    print(
        "\nPositive values mean that the "
        "oracle-box result has the higher "
        "mean metric."
    )

    print(
        "This is reported descriptively; "
        "no performance ordering is assumed "
        "before observing the results."
    )

    print("=" * 70)

    # ===========================================================
    # SAM confidence scores
    # ===========================================================

    print("\n")
    print("=" * 70)
    print(
        "SAM PREDICTION SCORES"
    )
    print("=" * 70)

    print(
        f"Oracle-box mean SAM score     : "
        f"{oracle_sam_scores.mean():.4f}"
    )

    print(
        f"Automatic-box mean SAM score  : "
        f"{automatic_sam_scores.mean():.4f}"
    )

    print("=" * 70)

    # ===========================================================
    # Comparison table
    # ===========================================================

    # If Step 1 results are available in a text file, they can
    # be added manually later. For now, Step 2 contains the
    # complete SAM comparison.

    results = pd.DataFrame(
        {
            "Method": [
                "SAM (oracle box)",
                "SAM (automatic Otsu box)"
            ],
            "Mean IoU": [
                oracle_mean_iou,
                automatic_mean_iou
            ],
            "Std IoU": [
                oracle_std_iou,
                automatic_std_iou
            ],
            "Mean Dice": [
                oracle_mean_dice,
                automatic_mean_dice
            ],
            "Std Dice": [
                oracle_std_dice,
                automatic_std_dice
            ]
        }
    )

    print("\n")
    print("=" * 70)
    print(
        "STEP 2 COMPARISON"
    )
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print("=" * 70)

    # ===========================================================
    # Save results
    # ===========================================================

    csv_path = os.path.join(
        DATA_ROOT,
        "step2_sam_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    txt_path = os.path.join(
        DATA_ROOT,
        "step2_sam_results.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ISIC 2018 Task 1\n"
        )

        f.write(
            "STEP 2 - Zero-Shot SAM\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Processed samples: "
            f"{len(oracle_ious)}\n\n"
        )

        f.write(
            "SAM (oracle box)\n"
        )

        f.write(
            f"Mean IoU: "
            f"{oracle_mean_iou:.6f}\n"
        )

        f.write(
            f"Std IoU: "
            f"{oracle_std_iou:.6f}\n"
        )

        f.write(
            f"Mean Dice: "
            f"{oracle_mean_dice:.6f}\n"
        )

        f.write(
            f"Std Dice: "
            f"{oracle_std_dice:.6f}\n\n"
        )

        f.write(
            "SAM (automatic Otsu box)\n"
        )

        f.write(
            f"Mean IoU: "
            f"{automatic_mean_iou:.6f}\n"
        )

        f.write(
            f"Std IoU: "
            f"{automatic_std_iou:.6f}\n"
        )

        f.write(
            f"Mean Dice: "
            f"{automatic_mean_dice:.6f}\n"
        )

        f.write(
            f"Std Dice: "
            f"{automatic_std_dice:.6f}\n\n"
        )

        f.write(
            "Prompt difference\n"
        )

        f.write(
            f"Oracle - Automatic IoU: "
            f"{iou_difference:+.6f}\n"
        )

        f.write(
            f"Oracle - Automatic Dice: "
            f"{dice_difference:+.6f}\n\n"
        )

        f.write(
            "Automatic fallback\n"
        )

        f.write(
            f"Count: "
            f"{fallback_count}\n"
        )

        f.write(
            f"Percentage: "
            f"{100 * fallback_count / len(automatic_ious):.4f}%\n"
        )

    print(
        f"\nSaved CSV:"
        f"\n{csv_path}"
    )

    print(
        f"\nSaved text results:"
        f"\n{txt_path}"
    )

    # ===========================================================
    # Visualizations
    # ===========================================================

    print("\n")
    print("=" * 70)
    print(
        "GENERATING QUALITATIVE EXAMPLES"
    )
    print("=" * 70)

    # -----------------------------------------------------------
    # Oracle
    # -----------------------------------------------------------

    show_examples(
        oracle_best,
        "SAM Oracle Box - Best Cases"
    )

    show_examples(
        oracle_worst,
        "SAM Oracle Box - Worst Cases"
    )

    # -----------------------------------------------------------
    # Automatic
    # -----------------------------------------------------------

    show_examples(
        automatic_best,
        "SAM Automatic Box - Best Cases"
    )

    show_examples(
        automatic_worst,
        "SAM Automatic Box - Worst Cases"
    )

    # ===========================================================
    # IoU distribution
    # ===========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            oracle_ious,
            automatic_ious
        ],
        labels=[
            "SAM\nOracle Box",
            "SAM\nAutomatic Otsu Box"
        ]
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "SAM IoU Distribution"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    boxplot_path = os.path.join(
        DATA_ROOT,
        "step2_sam_iou_distribution.png"
    )

    plt.savefig(
        boxplot_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nSaved:"
        f"\n{boxplot_path}"
    )

    plt.show()

    # ===========================================================
    # Dice distribution
    # ===========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            oracle_dices,
            automatic_dices
        ],
        labels=[
            "SAM\nOracle Box",
            "SAM\nAutomatic Otsu Box"
        ]
    )

    plt.ylabel(
        "Dice"
    )

    plt.title(
        "SAM Dice Distribution"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    dice_boxplot_path = os.path.join(
        DATA_ROOT,
        "step2_sam_dice_distribution.png"
    )

    plt.savefig(
        dice_boxplot_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nSaved:"
        f"\n{dice_boxplot_path}"
    )

    plt.show()

    # ===========================================================
    # Final
    # ===========================================================

    print("\n")
    print("=" * 70)
    print(
        "STEP 2 COMPLETED"
    )
    print("=" * 70)

    print(
        f"Samples processed : "
        f"{len(oracle_ious)}"
    )

    print(
        f"\nSAM + Oracle Box"
        f"\n  IoU  = {oracle_mean_iou:.4f}"
        f"\n  Dice = {oracle_mean_dice:.4f}"
    )

    print(
        f"\nSAM + Automatic Otsu Box"
        f"\n  IoU  = {automatic_mean_iou:.4f}"
        f"\n  Dice = {automatic_mean_dice:.4f}"
    )

    print(
        f"\nAutomatic-box fallbacks:"
        f" {fallback_count}"
    )

    print("=" * 70)


# ===============================================================
# 16. Run
# ===============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(f"\nException type: {type(e).__name__}")
        print(f"Exception: {e}")

        print("\nFull traceback:")
        traceback.print_exc()

        input("\nPress ENTER to close...")
