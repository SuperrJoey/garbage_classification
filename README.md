# EcoSort — Garbage Identification Model

A MobileNetV2-based image classification model that identifies waste into three categories: **Plastic**, **Paper**, and **Hazardous**. Trained with TensorFlow and exported to TFLite for on-device inference in the EcoSort Flutter app.

---

## Classes

| Index | Label      |
|-------|------------|
| 0     | hazardous  |
| 1     | paper      |
| 2     | plastic    |

> Class order is alphabetical — this matches `tf.keras.utils.image_dataset_from_directory` default behaviour. Always use `class_names.txt` as the source of truth for index-to-label mapping.

---

## Project Structure

```
garbage_identification/
├── dataset/                  # Raw images (not committed)
├── prepared_dataset/         # Organised by class folder (not committed)
│   ├── hazardous/
│   ├── paper/
│   └── plastic/
├── prepare_dataset.py        # Organises raw dataset into class folders
├── train.py                  # Trains MobileNetV2 model, saves .keras + class_names.txt
├── convert.py                # Converts .keras model to .tflite
├── predict_tflite.py         # Runs inference on a single image using .tflite
├── ecosort_model.keras       # Trained Keras model
├── ecosort_model.tflite      # TFLite model for mobile deployment
└── class_names.txt           # Class index order used during training
```

---

## Prerequisites

- Python 3.9+
- A virtual environment (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

> Pillow is **not** required — image loading uses TensorFlow's native I/O.

---

## Pipeline

### 1. Prepare Dataset

Organise your raw images into class subfolders under `prepared_dataset/`:

```bash
python prepare_dataset.py
```

Expected structure after running:

```
prepared_dataset/
├── hazardous/
├── paper/
└── plastic/
```

You can also manually place images directly into these folders.

---

### 2. Train

```bash
python train.py
```

Optional arguments:

| Argument             | Default                  | Description                        |
|----------------------|--------------------------|------------------------------------|
| `--data-dir`         | `prepared_dataset`       | Dataset directory                  |
| `--img-size`         | `224`                    | Input image size (square)          |
| `--batch-size`       | `32`                     | Batch size                         |
| `--epochs`           | `20`                     | Max training epochs                |
| `--learning-rate`    | `0.0001`                 | Initial learning rate              |
| `--output-model`     | `ecosort_model.keras`    | Output model file                  |
| `--class-names-output` | `class_names.txt`      | Output class names file            |

Example:

```bash
python train.py --epochs 25 --batch-size 16
```

**What happens during training:**
- MobileNetV2 backbone (ImageNet weights, frozen)
- Data augmentation: random flip, rotation, zoom, contrast
- Automatic class weights to handle imbalanced data
- Early stopping (patience 5) + learning rate reduction on plateau
- Saves `ecosort_model.keras` and `class_names.txt` on completion

---

### 3. Convert to TFLite

```bash
python convert.py
```

Optional arguments:

| Argument    | Default                  | Description              |
|-------------|--------------------------|--------------------------|
| `--model`   | `ecosort_model.keras`    | Input Keras model path   |
| `--output`  | `ecosort_model.tflite`   | Output TFLite file path  |

---

### 4. Run Inference

```bash
python predict_tflite.py --image path/to/image.jpg
```

Optional arguments:

| Argument          | Default                  | Description                                          |
|-------------------|--------------------------|------------------------------------------------------|
| `--image`         | *(required)*             | Path to input image                                  |
| `--model`         | `ecosort_model.tflite`   | TFLite model path                                    |
| `--img-size`      | `224`                    | Input image size (must match training)               |
| `--dataset-dir`   | `prepared_dataset`       | Used to infer class order if `--class-names` not set |
| `--class-names`   | *(auto)*                 | Comma-separated override e.g. `hazardous,paper,plastic` |

Example output:

```
Predicted: paper (0.9100)
All class scores:
- hazardous: 0.0797
- paper: 0.9100
- plastic: 0.0103
```

---

## Flutter Integration

Copy these two files into your Flutter app's assets:

```
ecosort_model.tflite  →  assets/models/ecosort_model.tflite
class_names.txt       →  assets/models/class_names.txt
```

**Critical:** preprocessing on the Flutter side must match training exactly:
- Resize to `224×224` using **bilinear interpolation**
- Keep pixel range in **`[0, 255]`** as float32 (do not divide by 255 before inference)
- Input shape: `[1, 224, 224, 3]` float32

Reason: the exported model already includes `mobilenet_v2.preprocess_input`, so manual normalization to `[0,1]` in app code will produce wrong predictions.

---

## Model Details

| Property         | Value                        |
|------------------|------------------------------|
| Architecture     | MobileNetV2 (transfer learning) |
| Input size       | 224 × 224 × 3                |
| Output           | Softmax over 3 classes       |
| Base weights     | ImageNet (frozen)            |
| Optimizer        | Adam (lr=1e-4)               |
| Loss             | Sparse categorical crossentropy |
| Export format    | `.keras` → `.tflite`         |

---

## Notes

- Always retrain if you add new classes or significantly change the dataset.
- After retraining, reconvert with `convert.py` and redeploy `ecosort_model.tflite` + `class_names.txt` together — they must stay in sync.
- The `.h5` format is **not supported** for this architecture due to custom layer serialisation issues. Use `.keras` only.

---

## GitHub Publishing Checklist

- Keep `dataset/` and `prepared_dataset/` out of git (already configured in `.gitignore`).
- Keep local training artifacts (`.keras`, `.h5`, logs) out of git.
- Commit only source + docs + required deployment files (`ecosort_model.tflite`, `class_names.txt` if you want consumers to run inference immediately).
- Verify this quick test before pushing:

```bash
python predict_tflite.py --image path/to/test.jpg
```
