import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from pathlib import Path
import numpy as np
import colorama
from colorama import Fore, Style

CHECKPOINT_PATH = 'app/assets/checkpoint.tf'
CHECKPOINT_PATH = Path.cwd().joinpath(CHECKPOINT_PATH)

TARGET_NAMES = ['normal','violence']

def configure_tf():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Predictor():
    def __init__(self, checkpoint_path:str = CHECKPOINT_PATH, target_names: list[str] = TARGET_NAMES):
        configure_tf()
        self.target_names = target_names
        self.model = keras.models.load_model(checkpoint_path)
        print(Fore.LIGHTBLUE_EX+'Model : Model loaded')
        print('Model : Accepting queries of shape: ',self.model.layers[0].input_shape)
        print('Model : Output classes: ',self.target_names,Style.RESET_ALL)

    def get_input_shape(self) -> tuple:
        return self.model.layers[0].input_shape

    def get_output_classes(self) -> list[str]:
        return self.target_names

    def predict(self, array:np.ndarray) -> str:
        prediction = np.argmax(self.model.predict(array))
        return self.target_names[prediction]

