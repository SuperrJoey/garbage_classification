import argparse
import os

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite.")
    parser.add_argument(
        "--model",
        default="ecosort_model.keras",
        help="Input Keras model path (.keras recommended).",
    )
    parser.add_argument(
        "--output",
        default="ecosort_model.tflite",
        help="Output .tflite file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}. "
            "If you only have an old .h5 model, retrain and save as .keras."
        )

    model = tf.keras.models.load_model(args.model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(args.output, "wb") as file:
        file.write(tflite_model)

    print(f"TFLite model ready: {args.output}")


if __name__ == "__main__":
    main()