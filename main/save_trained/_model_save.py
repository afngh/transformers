import torch
import torch.nn as nn
import pickle

class SaveModel:
    def __init__(self, ModelSaveData, ConfigSaveData, ConfigPathData):
        self.model = ModelSaveData.model
        self.optimizer = ModelSaveData.optimizer
        self.scheduler = ModelSaveData.scheduler
        self.model_path = ConfigPathData.model_path

        self.config = ConfigSaveData
        self.config_path = ConfigPathData.config_path

    def save(self):
        try:
            torch.save({
                "model" : self.model,
                "optimizer" : self.optimizer,
                "scheduler" : self.scheduler
            }, self.model_path)
            print(f"Model & O,S saved successfully at {self.model_path}")

            with open(self.config_path, 'wb') as f:
                pickle.dump(self.config, f)
            print(f"Config saved successfully at {self.config_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

        