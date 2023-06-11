import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from pathlib import Path
import numpy as np
from logging import getLogger as log

CHECKPOINT_PATH = "app/assets/checkpoint.tf"
CHECKPOINT_PATH = Path.cwd().joinpath(CHECKPOINT_PATH)

TARGET_NAMES = ["normal", "violence"]


def configure_tf():
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Predictor:
    def __init__(
        self,
        bin_chk_path: str = CHECKPOINT_PATH,
        target_names: list[str] = TARGET_NAMES,
        multi_chk_path: str = None,
        multi_target_names: list[str] = None,
    ):
        configure_tf()
        self.target_names = target_names
        self.model_bianry = keras.models.load_model(bin_chk_path)
        log().info("Model binary : Model loaded")
        log().info(
            f"Model binary : Accepting queries of shape: {self.model_bianry.layers[0].input_shape[0]}"
        )
        log().info(f"Model binary : Output Shape: {self.model_bianry.layers[-1].units}")
        log().info(f"Model binary : Output classes: {self.target_names}")

        if multi_chk_path is not None:
            self.model_multi = keras.models.load_model(multi_chk_path)
            assert (
                len(multi_target_names) == self.model_multi.layers[-1].units
            ), "Target classes do not match model output shape"
            self.multi_target_names = multi_target_names
            log().info("Model multi : Model loaded")
            log().info(
                f"Model multi : Accepting queries of shape: {self.model_multi.layers[0].input_shape[0]}"
            )
            log().info(
                f"Model multi : Output Shape: {self.model_multi.layers[-1].units}"
            )
            log().info(f"Model multi : Output classes: {self.multi_target_names}")

    def get_input_shape(self) -> tuple:
        log().debug(f"get_input_shape: {self.model_bianry.layers[0].input_shape[0]}")
        return self.model_bianry.layers[0].input_shape[0]

    def get_multi_input_shape(self) -> tuple:
        assert self.model_multi is not None, "Multi model not loaded"
        log().debug(f"get_input_shape: {self.model_multi.layers[0].input_shape[0]}")
        return self.model_multi.layers[0].input_shape[0]

    def get_output_classes(self) -> list[str]:
        return self.target_names

    def get_multi_output_classes(self) -> list[str]:
        assert self.model_multi is not None, "Multi model not loaded"
        return self.multi_target_names

    def predict_binary(self, array: np.ndarray):
        log().debug(f"predict: {array.shape}")
        assert (
            array.shape == self.model_bianry.layers[0].input_shape[0]
        ), "Input shape does not match model input shape"
        prediction = np.argmax(self.model_bianry.predict(array), axis=1)
        return prediction

    def predict_multi(self, array: np.ndarray):
        assert self.model_multi is not None, "Multi model not loaded"
        log().debug(f"predict: {array.shape}")
        assert (
            array.shape == self.model_multi.layers[0].input_shape[0]
        ), "Input shape does not match model input shape"
        prediction = np.argmax(self.model_multi.predict(array), axis=1)
        return prediction
