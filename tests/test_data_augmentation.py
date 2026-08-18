import tensorflow as tf

from components.data_augmentation import (
    ECGDataAugmentation,
)


def test_data_augmentation():

    augmentation = (
        ECGDataAugmentation()
    )

    dummy_images = tf.random.uniform(
        shape=(4, 224, 224, 3),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    augmented_images = augmentation(
        dummy_images,
        training=True,
    )

    print("=" * 70)
    print("ECG DATA AUGMENTATION TEST")
    print("=" * 70)

    print("\nOriginal shape:")
    print(dummy_images.shape)

    print("\nAugmented shape:")
    print(augmented_images.shape)

    print("\nOriginal dtype:")
    print(dummy_images.dtype)

    print("\nAugmented dtype:")
    print(augmented_images.dtype)

    assert augmented_images.shape == (
        4,
        224,
        224,
        3,
    )

    assert augmented_images.dtype == (
        tf.float32
    )

    print("\n" + "=" * 70)
    print("ECG DATA AUGMENTATION TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_data_augmentation()