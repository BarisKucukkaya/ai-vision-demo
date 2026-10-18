"""Same small CNN as train_cnn_pytorch.py, trained with Keras compile + fit (synthetic 28x28 data)."""
import numpy as np
import keras
from keras import layers


def make_data(n_per_class=200, seed=0):
    """Class k = a bright horizontal bar at a class-specific row, plus noise. 10 classes, no download."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.2, (10 * n_per_class, 28, 28, 1)).astype("float32")
    y = np.repeat(np.arange(10), n_per_class)
    for i, k in enumerate(y):
        x[i, 2 + 2 * k:4 + 2 * k, 4:24, 0] += 1.0
    order = rng.permutation(len(y))
    return x[order], y[order].astype("int64")


model = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Flatten(), layers.Dense(10),
])


def main():
    x, y = make_data()
    # excerpt: train_keras
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x, y, batch_size=64, epochs=3, verbose=2)
    # endexcerpt


if __name__ == "__main__":
    main()
