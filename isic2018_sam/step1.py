"""
Skin Lesion Segmentation on ISIC 2018 (Task 1)
================================================

Memory-efficient classical baseline using:

    Otsu thresholding
    + hair removal
    + morphological cleanup
    + largest connected component

IMPORTANT:
    This version does NOT load the training set into RAM because
    the classical Otsu method does not require training.

    It also does NOT load the entire test set into RAM.

Instead:

    Disk
      |
      v
    One image + mask
      |
      v
    Otsu segmentation
      |
      v
    IoU / Dice
      |
      v
    discard image
      |
      v
    next image

Only the best 3 and worst 3 examples are retained for visualization.

Expected dataset structure:

ISIC2018\\
│
├── segmentation.py
│
├── ISIC2018_Task1-2_Training_Input\\
│   └── ... images ...
│
├── ISIC2018_Task1_Training_GroundTruth\\
│   └── ... masks ...
│
├── ISIC2018_Task1-2_Validation_Input\\
│   └── ... images ...
│
├── ISIC2018_Task1_Validation_GroundTruth\\
│   └── ... masks ...
│
├── ISIC2018_Task1-2_Test_Input\\
│   └── ... images ...
│
└── ISIC2018_Task1_Test_GroundTruth\\
    └── ... masks ...

The loader searches recursively, so an additional nested folder
inside these directories is also supported.
"""

# ===============================================================
# 0. Imports
# ===============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ===============================================================
# 1. Configuration
# ===============================================================

SEED = 42
np.random.seed(SEED)

# Images are resized to this resolution before segmentation.
IMG_SIZE = 256

# Number of examples to retain for visualization.
NUM_VISUAL_EXAMPLES = 3


# ===============================================================
# 2. Dataset root
# ===============================================================

# Automatically use the directory containing this Python file.

DATA_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ===============================================================
# 3. Dataset directories
# ===============================================================

TRAIN_IMG_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1-2_Training_Input"
)

TRAIN_MASK_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1_Training_GroundTruth"
)

VAL_IMG_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1-2_Validation_Input"
)

VAL_MASK_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1_Validation_GroundTruth"
)

TEST_IMG_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1-2_Test_Input"
)

TEST_MASK_DIR = os.path.join(
    DATA_ROOT,
    "ISIC2018_Task1_Test_GroundTruth"
)


# ===============================================================
# 4. File discovery
# ===============================================================

def find_image_files(directory):
    """
    Recursively find image files.

    Searching recursively is intentional because the dataset may
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
# 5. Dataset diagnostics
# ===============================================================

def print_dataset_status():
    """
    Print dataset directories and file counts.

    Note:
        Training and validation counts are displayed for
        verification only. They are NOT loaded into RAM.
    """

    print("\n" + "=" * 70)
    print("DATASET CONFIGURATION")
    print("=" * 70)

    print(
        f"\nDATA_ROOT:\n"
        f"  {DATA_ROOT}"
    )

    datasets = [
        (
            "Training",
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR
        ),
        (
            "Validation",
            VAL_IMG_DIR,
            VAL_MASK_DIR
        ),
        (
            "Test",
            TEST_IMG_DIR,
            TEST_MASK_DIR
        )
    ]

    for name, img_dir, mask_dir in datasets:

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        print(
            f"Images:\n  {img_dir}"
        )

        print(
            f"Masks:\n  {mask_dir}"
        )

        print(
            f"\nImage directory exists: "
            f"{os.path.isdir(img_dir)}"
        )

        print(
            f"Mask directory exists: "
            f"{os.path.isdir(mask_dir)}"
        )

        image_count = len(
            find_image_files(img_dir)
        )

        mask_count = len(
            find_mask_files(mask_dir)
        )

        print(
            f"Images found: {image_count}"
        )

        print(
            f"Masks found : {mask_count}"
        )

    print("\n" + "=" * 70)


# ===============================================================
# 6. Create mask lookup
# ===============================================================

def create_mask_lookup(mask_dir):
    """
    Create:

        image_id -> mask path

    Example:

        ISIC_0000001
            ->
        D:\\...\\ISIC_0000001_segmentation.png
    """

    mask_paths = find_mask_files(
        mask_dir
    )

    mask_dict = {}

    for mask_path in mask_paths:

        filename = os.path.basename(
            mask_path
        )

        image_id = filename

        image_id = image_id.replace(
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
# 7. Streaming image/mask iterator
# ===============================================================

def iter_pairs(img_dir, mask_dir):
    """
    Yield one image/mask pair at a time.

    IMPORTANT:
        Images are NOT accumulated in a list.

    This keeps memory usage low.
    """

    image_paths = find_image_files(
        img_dir
    )

    mask_dict = create_mask_lookup(
        mask_dir
    )

    print(
        f"\nFound {len(image_paths)} images."
    )

    print(
        f"Found {len(mask_dict)} masks."
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            f"\nNo images found in:\n"
            f"{img_dir}"
        )

    if len(mask_dict) == 0:

        raise RuntimeError(
            f"\nNo segmentation masks found in:\n"
            f"{mask_dir}"
        )

    missing_masks = 0

    for image_path in image_paths:

        filename = os.path.basename(
            image_path
        )

        image_id = os.path.splitext(
            filename
        )[0]

        # -------------------------------------------------------
        # Find corresponding mask
        # -------------------------------------------------------

        mask_path = mask_dict.get(
            image_id
        )

        if mask_path is None:

            missing_masks += 1

            continue

        # -------------------------------------------------------
        # Read image
        # -------------------------------------------------------

        img = cv2.imread(
            image_path
        )

        if img is None:

            print(
                f"\nWARNING: Could not read image:"
                f"\n{image_path}"
            )

            continue

        # BGR -> RGB

        img = cv2.cvtColor(
            img,
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
                f"\nWARNING: Could not read mask:"
                f"\n{mask_path}"
            )

            continue

        # -------------------------------------------------------
        # Resize
        # -------------------------------------------------------

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------------
        # Binarize ground truth
        # -------------------------------------------------------

        mask = (
            mask > 127
        ).astype(np.uint8)

        # -------------------------------------------------------
        # Yield one pair
        # -------------------------------------------------------

        yield img, mask, image_id

    if missing_masks > 0:

        print(
            f"\nWARNING:"
            f"\n{missing_masks} images did not have "
            f"a corresponding segmentation mask."
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
        pred.sum() + gt.sum() + eps
    )


# ===============================================================
# 9. Classical segmentation
# ===============================================================

def classical_segment(img):
    """
    Classical Otsu-based skin lesion segmentation.

    Pipeline:

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
        morphological opening
         |
         v
        morphological closing
         |
         v
        largest connected component
         |
         v
        binary lesion mask
    """

    # -----------------------------------------------------------
    # RGB -> grayscale
    # -----------------------------------------------------------

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
    # Hair removal using inpainting
    # -----------------------------------------------------------

    gray_clean = cv2.inpaint(
        gray,
        hair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    # -----------------------------------------------------------
    # Otsu threshold
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
    # Largest connected component
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
# 10. Visualization
# ===============================================================

def show_examples(
    examples,
    title
):
    """
    Show best/worst examples.

    examples is a list containing:

        {
            "image": image,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "id": image_id,
            "score": score
        }
    """

    if len(examples) == 0:

        print(
            f"\nNo examples available for: "
            f"{title}"
        )

        return

    n = len(examples)

    fig, axes = plt.subplots(
        3,
        n,
        figsize=(4 * n, 10)
    )

    if n == 1:

        axes = np.array(
            axes
        ).reshape(
            3,
            1
        )

    for col, example in enumerate(
        examples
    ):

        image = example["image"]

        prediction = example[
            "prediction"
        ]

        ground_truth = example[
            "ground_truth"
        ]

        image_id = example["id"]

        score = example["score"]

        # -------------------------------------------------------
        # Original
        # -------------------------------------------------------

        axes[0, col].imshow(
            image
        )

        axes[0, col].set_title(
            f"{image_id}\nIoU = {score:.4f}"
        )

        # -------------------------------------------------------
        # Prediction
        # -------------------------------------------------------

        axes[1, col].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, col].set_title(
            "Prediction"
        )

        # -------------------------------------------------------
        # Ground truth
        # -------------------------------------------------------

        axes[2, col].imshow(
            ground_truth,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[2, col].set_title(
            "Ground Truth"
        )

        for row in range(3):

            axes[row, col].axis(
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
        f"\nSaved visualization:"
        f"\n{output_path}"
    )

    plt.show()


# ===============================================================
# 11. Main experiment
# ===============================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "ISIC 2018 TASK 1 - "
        "MEMORY-EFFICIENT CLASSICAL BASELINE"
    )
    print("=" * 70)

    # -----------------------------------------------------------
    # Dataset information
    # -----------------------------------------------------------

    print_dataset_status()

    # -----------------------------------------------------------
    # IMPORTANT
    #
    # We intentionally DO NOT load:
    #
    #   train_images
    #   train_masks
    #   val_images
    #   val_masks
    #
    # because the classical Otsu method does not train.
    # -----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MEMORY-EFFICIENT MODE")
    print("=" * 70)

    print(
        "Training images will NOT be loaded."
    )

    print(
        "Validation images will NOT be loaded."
    )

    print(
        "Test images will be processed one at a time."
    )

    print("=" * 70)

    # -----------------------------------------------------------
    # Verify test set
    # -----------------------------------------------------------

    test_image_count = len(
        find_image_files(
            TEST_IMG_DIR
        )
    )

    test_mask_count = len(
        find_mask_files(
            TEST_MASK_DIR
        )
    )

    if test_image_count == 0:

        raise RuntimeError(
            "\nNo test images found."
        )

    if test_mask_count == 0:

        raise RuntimeError(
            "\nNo test masks found."
        )

    print("\nTest dataset:")

    print(
        f"  Images: {test_image_count}"
    )

    print(
        f"  Masks : {test_mask_count}"
    )

    # -----------------------------------------------------------
    # Create streaming iterator
    # -----------------------------------------------------------

    test_iterator = iter_pairs(
        TEST_IMG_DIR,
        TEST_MASK_DIR
    )

    # -----------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------

    ious = []
    dices = []

    # -----------------------------------------------------------
    # Best/worst examples
    #
    # Only 3 of each are stored.
    # -----------------------------------------------------------

    best_examples = []
    worst_examples = []

    # -----------------------------------------------------------
    # Process test set
    # -----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNNING CLASSICAL OTSU BASELINE")
    print("=" * 70)

    processed = 0

    for img, gt, image_id in test_iterator:

        # -------------------------------------------------------
        # Segment
        # -------------------------------------------------------

        pred = classical_segment(
            img
        )

        # -------------------------------------------------------
        # Metrics
        # -------------------------------------------------------

        iou = iou_score(
            pred,
            gt
        )

        dice = dice_score(
            pred,
            gt
        )

        ious.append(
            iou
        )

        dices.append(
            dice
        )

        processed += 1

        # -------------------------------------------------------
        # Example information
        # -------------------------------------------------------

        example = {
            "image": img.copy(),
            "prediction": pred.copy(),
            "ground_truth": gt.copy(),
            "id": image_id,
            "score": iou
        }

        # -------------------------------------------------------
        # Maintain BEST examples
        #
        # Highest IoU = best
        # -------------------------------------------------------

        best_examples.append(
            example
        )

        best_examples.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best_examples = best_examples[
            :NUM_VISUAL_EXAMPLES
        ]

        # -------------------------------------------------------
        # Maintain WORST examples
        #
        # Lowest IoU = worst
        # -------------------------------------------------------

        worst_examples.append(
            example
        )

        worst_examples.sort(
            key=lambda x: x["score"]
        )

        worst_examples = worst_examples[
            :NUM_VISUAL_EXAMPLES
        ]

        # -------------------------------------------------------
        # Progress
        # -------------------------------------------------------

        if (
            processed % 100 == 0
            or processed == test_image_count
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{test_image_count}"
            )

    # ===========================================================
    # Convert metrics to NumPy arrays
    # ===========================================================

    ious = np.array(
        ious,
        dtype=np.float32
    )

    dices = np.array(
        dices,
        dtype=np.float32
    )

    # -----------------------------------------------------------
    # Safety check
    # -----------------------------------------------------------

    if len(ious) == 0:

        raise RuntimeError(
            "\nNo test samples were successfully processed."
        )

    # ===========================================================
    # Final results
    # ===========================================================

    print("\n")
    print("=" * 70)
    print("CLASSICAL OTSU RESULTS")
    print("=" * 70)

    print(
        f"Processed samples : "
        f"{len(ious)}"
    )

    print(
        f"Mean IoU           : "
        f"{ious.mean():.4f}"
    )

    print(
        f"Std IoU            : "
        f"{ious.std():.4f}"
    )

    print(
        f"Median IoU         : "
        f"{np.median(ious):.4f}"
    )

    print(
        f"Minimum IoU        : "
        f"{ious.min():.4f}"
    )

    print(
        f"Maximum IoU        : "
        f"{ious.max():.4f}"
    )

    print(
        f"\nMean Dice          : "
        f"{dices.mean():.4f}"
    )

    print(
        f"Std Dice           : "
        f"{dices.std():.4f}"
    )

    print(
        f"Median Dice        : "
        f"{np.median(dices):.4f}"
    )

    print(
        f"Minimum Dice       : "
        f"{dices.min():.4f}"
    )

    print(
        f"Maximum Dice       : "
        f"{dices.max():.4f}"
    )

    print("=" * 70)

    # ===========================================================
    # Save numerical results
    # ===========================================================

    results_path = os.path.join(
        DATA_ROOT,
        "classical_otsu_results.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ISIC 2018 Task 1\n"
        )

        f.write(
            "Classical Otsu Baseline\n"
        )

        f.write(
            "=" * 50 + "\n"
        )

        f.write(
            f"Test samples: {len(ious)}\n"
        )

        f.write(
            f"Mean IoU: {ious.mean():.6f}\n"
        )

        f.write(
            f"Std IoU: {ious.std():.6f}\n"
        )

        f.write(
            f"Median IoU: {np.median(ious):.6f}\n"
        )

        f.write(
            f"Min IoU: {ious.min():.6f}\n"
        )

        f.write(
            f"Max IoU: {ious.max():.6f}\n"
        )

        f.write(
            f"Mean Dice: {dices.mean():.6f}\n"
        )

        f.write(
            f"Std Dice: {dices.std():.6f}\n"
        )

        f.write(
            f"Median Dice: {np.median(dices):.6f}\n"
        )

        f.write(
            f"Min Dice: {dices.min():.6f}\n"
        )

        f.write(
            f"Max Dice: {dices.max():.6f}\n"
        )

    print(
        f"\nResults saved to:"
        f"\n{results_path}"
    )

    # ===========================================================
    # Visualizations
    # ===========================================================

    show_examples(
        best_examples,
        "Classical Otsu - Best Cases"
    )

    show_examples(
        worst_examples,
        "Classical Otsu - Worst Cases"
    )

    # ===========================================================
    # Finished
    # ===========================================================

    print("\n")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"Processed {len(ious)} test images."
    )

    print(
        f"Mean IoU  = {ious.mean():.4f}"
    )

    print(
        f"Mean Dice = {dices.mean():.4f}"
    )

    print("=" * 70)


# ===============================================================
# 12. Run experiment
# ===============================================================

if __name__ == "__main__":
    main()
