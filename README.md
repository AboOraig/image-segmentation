# Deep Learning for Object Detection and Image Segmentation

Computer-vision pipelines in PyTorch and OpenCV, from classical clustering to YOLO and the Segment Anything Model (SAM). The repository has two parts:

1. **Urban-scene segmentation (Cityscapes)**: K-Means on pixel colours, and YOLOv8n-seg / YOLO11n-seg inference and retraining on a 7-class Cityscapes subset.
2. **Skin-lesion segmentation (ISIC 2018 Task 1)**: a five-step study going from a classical Otsu baseline to zero-shot SAM, fine-tuning of the SAM mask decoder, interactive (point-prompted) SAM and a U-Net comparison.

Report (in `docs/`):

- [`ISIC2018_SAM_report.pdf`](docs/ISIC2018_SAM_report.pdf): the ISIC 2018 study.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   └── ISIC2018_SAM_report.pdf
├── cityscapes_segmentation/
│   ├── kmeans/
│   │   ├── kmeans.py                       # from-scratch K-Means + elbow method
│   │   ├── kmeans_2d.py                    # PCA visualisation of the clusters
│   │   ├── kmeans_3d.py                    # RGB-space 3D visualisation per iteration
│   │   ├── kmeans_3d_fausse_couleurs.py    # fixed false-colour palette
│   │   ├── kmeans_multiple.py              # sensitivity to random initialisation
│   │   └── kmeans_multiple_para.py         # same, sequential variant
│   └── yolo/
│       ├── YOLO_image.py                   # inference on an image
│       ├── YOLO_video.py                   # inference on a video, annotated output
│       ├── YOLO_image_reentrainer.py       # retraining on Cityscapes
│       ├── conversion_json_txt_det.py      # Cityscapes JSON -> YOLO boxes
│       └── conversion_json_txt_seg.py      # Cityscapes JSON -> YOLO polygons
└── isic2018_sam/
    ├── step1.py                            # Otsu baseline
    ├── step1_outputs                       # Step 1 outputs
    ├── step2.py                            # zero-shot SAM (oracle box / Otsu box)
    ├── step2_outputs                       # Step 2 outputs
    ├── step3a.py                           # fine-tune SAM mask decoder (GT boxes)
    ├── step3a_outputs                      # Step 3a outputs
    ├── step3b.py                           # fine-tuned SAM + automatic Otsu box
    ├── step3b_outputs                      # Step 3b outputs
    ├── step4.py                            # interactive SAM (positive/negative points)
    ├── step4_outputs                       # Step 4 outputs
    ├── step5.py                            # U-Net + ResNet-34 comparison
    └── step5_outputs                       # Step 5 outputs
```

## Part 1: Cityscapes (K-Means and YOLO)

**K-Means.** Pixels are treated as 3D RGB vectors. The scripts implement K-Means from scratch (random initialisation, Euclidean assignment, centroid update, empty-cluster re-seeding) and include an elbow-method curve, 2D (PCA) and 3D (RGB) visualisations, and a study of sensitivity to the random initialisation.

**YOLO.** The pretrained `yolov8n-seg` and `yolo11n-seg` models (Ultralytics) are run on images and videos, then retrained on a **7-class Cityscapes subset** (`person, car, truck, bus, motorcycle, bicycle, rider`). The two conversion scripts turn Cityscapes polygon annotations (`*_gtFine_polygons.json`) into YOLO `.txt` labels (boxes for detection, normalised polygons for segmentation). Observations from the report: all four training losses (box, cls, seg, dfl) decrease steadily over 100 epochs, and with only 10 epochs no useful detection is obtained. YOLOv8n-seg was more stable early on, while YOLO11n-seg needed more epochs and then gave more precise results in our tests.

## Part 2: ISIC 2018 Task 1 (SAM)

Lesion segmentation on dermoscopic images, resized to 256x256. Split: **2,594 train / 100 validation / 1,000 test**. The validation set selects checkpoints; the test set is used only for the final evaluation.

| Step | Script | Method |
|---|---|---|
| 1 | `step1.py` | Otsu thresholding + hair removal + morphology + largest component |
| 2 | `step2.py` | Pretrained SAM ViT-B, prompted with a ground-truth box (oracle) or an automatic Otsu box |
| 3A | `step3a.py` | Fine-tune only the SAM mask decoder (frozen image and prompt encoders), GT boxes |
| 3B | `step3b.py` | Fine-tuned SAM evaluated with automatic Otsu boxes |
| 4 | `step4.py` | Interactive SAM: one positive and one negative point on difficult cases |
| 5 | `step5.py` | U-Net with ImageNet-pretrained ResNet-34 encoder (BCE + Dice loss) |

### Results (1,000-image test set)

| Method | Mean IoU | Mean Dice |
|---|---|---|
| Classical Otsu | 0.4866 | 0.5660 |
| Pretrained SAM + GT box (oracle) | 0.7754 | 0.8651 |
| Pretrained SAM + automatic Otsu box | 0.4800 | 0.5617 |
| Fine-tuned SAM + GT box (oracle) | 0.8918 | 0.9413 |
| Fine-tuned SAM + automatic Otsu box | 0.5490 | 0.6249 |
| U-Net + ResNet-34 (no prompt needed) | 0.8106 | 0.8862 |

**How to read this table.** Rows using a ground-truth box are *oracle* experiments: the box comes from the annotation, so they show what SAM can do with a perfect prompt, not a deployable pipeline. The U-Net and the Otsu-box rows are fully automatic. The main finding is that prompt quality dominates: the same fine-tuned SAM goes from 0.892 to 0.549 mean IoU when the GT box is replaced by an Otsu-derived one.

**Step 4 (interactive).** This is a proof of concept. The ground-truth mask is used only by the experiment controller, to decide whether Otsu is "sufficiently wrong" (IoU below 0.70 by default) and, in simulated mode, to place the correction points. SAM itself only receives the image and the two points. `MODE = "manual"` opens a window to click points; `MODE = "simulated"` generates the points automatically. No aggregate score is reported for this step in the report.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

SAM is installed from source:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

A CUDA GPU is recommended for SAM fine-tuning, U-Net training and YOLO retraining.

## Data and weights (not included in the repo)

- **ISIC 2018 Task 1**: download from the [ISIC Challenge 2018](https://challenge.isic-archive.com/data/#2018) and place the folders in `isic2018_sam/`:
  `ISIC2018_Task1-2_Training_Input`, `ISIC2018_Task1_Training_GroundTruth`, `ISIC2018_Task1-2_Validation_Input`, `ISIC2018_Task1_Validation_GroundTruth`, `ISIC2018_Task1-2_Test_Input`, `ISIC2018_Task1_Test_GroundTruth`.
- **SAM ViT-B checkpoint**: download `sam_vit_b_01ec64.pth` from the [Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put it in `isic2018_sam/`.
- **Cityscapes**: download from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/), convert the labels with the scripts in `cityscapes_segmentation/yolo/`, and arrange them as `images/{train,val}` and `labels/{train,val}` with a `cityscapes.yaml` listing the 7 classes.

## Usage

Run the ISIC scripts **from inside `isic2018_sam/`** and in order, since later steps reuse earlier outputs (Step 3B and Step 4 load the checkpoint saved by Step 3A):

```bash
cd isic2018_sam
python step1.py     # Otsu baseline
python step2.py     # zero-shot SAM
python step3a.py    # fine-tune mask decoder -> step3a_outputs/sam_vit_b_step3a_best.pth
python step3b.py    # fine-tuned SAM + Otsu box
python step4.py     # interactive SAM (set MODE at the top of the script)
python step5.py     # U-Net comparison
```

Each step writes its metrics (CSV/JSON) and figures to its own `stepN_outputs/` folder.

Cityscapes part:

```bash
cd cityscapes_segmentation/yolo
python YOLO_image.py              # edit the image path inside the script
python YOLO_video.py              # edit the video path inside the script
python YOLO_image_reentrainer.py  # set the path to your cityscapes.yaml
```

The K-Means scripts load an image whose filename is set near the top of each file (`test2.png`, `car.png`, ...). Put your own image next to the script or edit that line.

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [Segment Anything](https://github.com/facebookresearch/segment-anything) and [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
