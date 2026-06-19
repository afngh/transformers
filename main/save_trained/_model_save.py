import torch
import torch.nn as nn
import pickle

class SaveModel:
    def __init__(self, ModelSaveData, ConfigSaveData, ConfigPathData):
        self.model = ModelSaveData.model
        self.model_path = ConfigPathData.model_path

        self.config = ConfigSaveData
        self.config_path = ConfigPathData.config_path

    def save(self):
        try:
            torch.save(self.model, self.model_path)
            print(f"Model saved successfully at {self.model_path}")

            with open(self.config_path, 'wb') as f:
                pickle.dump(self.config, f)
            print(f"Config saved successfully at {self.config_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

        