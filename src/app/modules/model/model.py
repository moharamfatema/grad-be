"""
Model module for the application. Contains the Predictor class which is used
to load and predict from the models.

Classes:
    Predictor: Predictor class for binary and multi-class models

Functions:
    configure_tf: Configure tensorflow to use mixed precision and suppress
        warnings

Variables:
    CHECKPOINT_PATH: Path to the checkpoint file
    TARGET_NAMES: Target names for the binary model

"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision

from app.modules.util.log import log

CHECKPOINT_PATH = "app/assets/checkpoint.tf"
CHECKPOINT_PATH = Path.cwd().joinpath(CHECKPOINT_PATH)

TARGET_NAMES = ("normal", "violence")


def configure_tf():
    """Configure tensorflow to use mixed precision and suppress warnings"""
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Predictor:
    """Predictor class for binary and multi-class models"""

    def __init__(
        self,
        bin_chk_path: str = CHECKPOINT_PATH,
        target_names: list[str] = TARGET_NAMES,
        multi_chk_path: str = None,
        multi_target_names: list[str] = None,
    ):
        configure_tf()
        self.__target_names = np.array(target_names)
        self.__model_bianry = keras.models.load_model(bin_chk_path)
        log.info("Model binary : Model loaded")
        log.info(
            "Model binary : Accepting queries of shape: %s",
            str(self.__model_bianry.layers[0].input_shape[0]),
        )
        log.info(
            "Model binary : Output Shape: %s",
            str(self.__model_bianry.layers[-1].units),
        )
        log.info("Model binary : Output classes: %s", str(self.__target_names))

        if multi_chk_path is not None:
            self.model_multi = keras.models.load_model(multi_chk_path)
            assert (
                len(multi_target_names) == self.model_multi.layers[-1].units
            ), "Target classes do not match model output shape"
            self.multi_target_names = np.array(multi_target_names)
            log.info("Model multi : Model loaded")
            log.info(
                "Model multi : Accepting queries of shape: %s",
                str(self.model_multi.layers[0].input_shape[0]),
            )
            log.info(
                "Model multi : Output Shape: %s",
                str(self.model_multi.layers[-1].units),
            )
            log.info(
                "Model multi : Output classes: %s",
                str(self.multi_target_names),
            )

    def get_input_shape(self) -> tuple:
        """Get the input shape of the binary model
        Returns:
            tuple: Input shape of the binary model
        """
        log.debug(
            "get_input_shape: %s",
            str(self.__model_bianry.layers[0].input_shape[0]),
        )
        return self.__model_bianry.layers[0].input_shape[0]

    def get_multi_input_shape(self) -> tuple:
        """Get the input shape of the multi-class model
        Returns:
            tuple: Input shape of the multi-class model
        """
        assert self.model_multi is not None, "Multi model not loaded"
        log.debug(
            "get_input_shape: %s",
            str(self.model_multi.layers[0].input_shape[0]),
        )
        return self.model_multi.layers[0].input_shape[0]

    def get_output_classes(self) -> list[str]:
        """Get the output classes of the binary model
        Returns:
            list[str]: Output classes of the binary model
        """
        return self.__target_names

    def get_multi_output_classes(self) -> list[str]:
        """Get the output classes of the multi-class model
        Returns:
            list[str]: Output classes of the multi-class model
        """
        assert self.model_multi is not None, "Multi model not loaded"
        return self.multi_target_names

    def predict_binary(self, array: np.ndarray) -> list[str]:
        """Predict the class of a binary model
        Args:
            array (np.ndarray): Input array to predict on
        Returns:
            list[str]: Predicted class
        """
        log.debug("predict binary: %s", str(array.shape))
        assert (
            array.shape[1:] == self.__model_bianry.layers[0].input_shape[0][1:]
        ), (
            f"Input shape {array.shape[1:]} "
            f"does not match model input shape "
            f"{self.__model_bianry.layers[0].input_shape[0][1:]}"
        )
        prediction = self.__target_names[
            np.argmax(
                self.__model_bianry.predict(
                    array, verbose=2
                ),
                axis=1,
            )
        ].tolist()
        return prediction

    def predict_multi(self, array: np.ndarray) -> list[str]:
        """Predict the class of a multi-class model
        Args:
            array (np.ndarray): Input array to predict on
        Returns:
            list[str]: Predicted class
        """
        assert self.model_multi is not None, "Multi model not loaded"
        log.debug("predict multi-class: %s", str(array.shape))
        assert (
            array.shape[1:] == self.__model_bianry.layers[0].input_shape[0][1:]
        ), (
            f"Input shape {array.shape[1:]} "
            f"does not match model input shape "
            f"{self.__model_bianry.layers[0].input_shape[0][1:]}"
        )
        prediction = self.__target_names[
            np.argmax(self.model_multi.predict(array, verbose=2), axis=1)
        ].tolist()
        return prediction
