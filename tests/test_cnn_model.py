from models.cnn_model import CardioVisionCNN


def test_cnn_model():

    print("=" * 70)
    print("CARDIOVISION CNN MODEL TEST")
    print("=" * 70)

    cnn = CardioVisionCNN()

    model = cnn.build_model()

    print("\nModel name:")
    print(model.name)

    print("\nInput shape:")
    print(model.input_shape)

    print("\nOutput shape:")
    print(model.output_shape)

    print("\nTotal parameters:")
    print(model.count_params())

    print("\nModel summary:")
    model.summary()

    # ---------------------------------------------
    # Architecture assertions
    # ---------------------------------------------

    assert model.input_shape == (
        None,
        224,
        224,
        3,
    )

    assert model.output_shape == (
        None,
        1,
    )

    assert model.name == (
        "CardioVision_Baseline_CNN"
    )

    # ---------------------------------------------
    # Check output layer
    # ---------------------------------------------

    output_layer = model.get_layer(
        "mi_probability"
    )

    assert output_layer.activation.__name__ == (
        "sigmoid"
    )

    print("\n" + "=" * 70)
    print("CARDIOVISION CNN MODEL TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_cnn_model()