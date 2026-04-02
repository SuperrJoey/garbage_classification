import argparse
import os

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train garbage classification model.")
    parser.add_argument("--data-dir", default="prepared_dataset", help="Dataset directory.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (square).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--output-model",
        default="ecosort_model.keras",
        help="Output Keras model file.",
    )
    parser.add_argument(
        "--class-names-output",
        default="class_names.txt",
        help="Path to save class names in output index order.",
    )
    return parser.parse_args()


def build_model(img_size: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(train_ds: tf.data.Dataset, num_classes: int) -> dict:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in train_ds:
        batch_counts = np.bincount(labels.numpy(), minlength=num_classes)
        counts += batch_counts

    total = int(np.sum(counts))
    class_weights = {}
    for idx, count in enumerate(counts):
        class_weights[idx] = (total / (num_classes * count)) if count > 0 else 1.0
    return class_weights


def save_class_names(path: str, class_names: list) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for name in class_names:
            file.write(f"{name}\n")


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    class_weights = compute_class_weights(train_ds, num_classes)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    model = build_model(args.img_size, num_classes, args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        ,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.save(args.output_model)
    save_class_names(args.class_names_output, class_names)
    print(f"Saved model to {args.output_model}")
    print(f"Class names: {class_names}")
    print(f"Saved class names to {args.class_names_output}")


if __name__ == "__main__":
    main()
