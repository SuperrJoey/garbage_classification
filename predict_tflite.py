import argparse
import os
from typing import List

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a TFLite model.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--model", default="ecosort_model.tflite", help="TFLite model path.")
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size used during training.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="prepared_dataset",
        help="Dataset directory used to infer class order if class names are not provided.",
    )
    parser.add_argument(
        "--class-names",
        default="",
        help="Comma-separated class names in model output order. Example: hazardous,paper,plastic",
    )
    parser.add_argument(
        "--debug-input",
        action="store_true",
        help="Print input fingerprint and tensor stats for debugging.",
    )
    return parser.parse_args()


def infer_class_names(dataset_dir: str) -> List[str]:
    if not os.path.isdir(dataset_dir):
        return []
    names = [entry.name for entry in os.scandir(dataset_dir) if entry.is_dir()]
    return sorted(names)


def load_image_as_tensor(image_path: str, img_size: int) -> np.ndarray:
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    arr = tf.expand_dims(img, axis=0).numpy()
    return arr


def fnv1a64_hex(data: bytes) -> str:
    prime = 0x100000001B3
    offset = 0xCBF29CE484222325
    mask = 0xFFFFFFFFFFFFFFFF
    value = offset
    for b in data:
        value ^= b
        value = (value * prime) & mask
    return f"{value:016x}"


def hex_slice(data: bytes, start: int, count: int) -> str:
    if not data:
        return ""
    safe_start = max(0, min(start, len(data) - 1))
    end = max(0, min(safe_start + count, len(data)))
    return data[safe_start:end].hex()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    class_names = (
        [item.strip() for item in args.class_names.split(",") if item.strip()]
        if args.class_names
        else infer_class_names(args.dataset_dir)
    )

    if args.debug_input:
        raw_bytes = tf.io.read_file(args.image).numpy()
        decoded = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
        resized = tf.image.resize(decoded, [args.img_size, args.img_size])
        float_img = tf.image.convert_image_dtype(resized, tf.float32)
        flat = tf.reshape(float_img, [-1]).numpy()
        first_pixel = float_img[0, 0, :].numpy()

        print("=== INPUT DEBUG ===")
        print(f"Image path         : {args.image}")
        print(f"Image bytes length : {len(raw_bytes)}")
        print(f"Image FNV1a64      : {fnv1a64_hex(raw_bytes)}")
        print(f"Image first16 hex  : {hex_slice(raw_bytes, 0, 16)}")
        print(f"Image last16 hex   : {hex_slice(raw_bytes, len(raw_bytes) - 16, 16)}")
        print(f"Decoded size       : {decoded.shape[1]}x{decoded.shape[0]}")
        print(f"Resized size       : {args.img_size}x{args.img_size}")
        print(
            "Input stats        : "
            f"min={float(np.min(flat)):.6f} "
            f"max={float(np.max(flat)):.6f} "
            f"mean={float(np.mean(flat)):.6f}"
        )
        print(
            "First pixel [r,g,b]: "
            f"[{float(first_pixel[0]):.6f}, "
            f"{float(first_pixel[1]):.6f}, "
            f"{float(first_pixel[2]):.6f}]"
        )
        print("===================")

    input_data = load_image_as_tensor(args.image, args.img_size)

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Match model input dtype while preserving normalization done above.
    input_dtype = input_details["dtype"]
    model_input = input_data.astype(input_dtype)

    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])[0]
    scores = np.array(output, dtype=np.float32)

    if class_names and len(class_names) != len(scores):
        print(
            "Warning: number of class names does not match model outputs. "
            "Printing index-based labels instead."
        )
        class_names = []

    top_idx = int(np.argmax(scores))
    top_score = float(scores[top_idx])
    top_label = class_names[top_idx] if class_names else f"class_{top_idx}"

    print(f"Predicted: {top_label} ({top_score:.4f})")
    print("All class scores:")

    for idx, score in enumerate(scores):
        label = class_names[idx] if class_names else f"class_{idx}"
        print(f"- {label}: {float(score):.4f}")


if __name__ == "__main__":
    main()
