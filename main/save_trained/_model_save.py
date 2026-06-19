import torch
import torch.nn as nn

class SaveModel:
    def __init__(self, ModelSaveConfig):
        self.model = ModelSaveConfig.model
        self.path = ModelSaveConfig.path

    def save(self):
        try:
            torch.save(self.model, self.path)
            print(f"Model saved successfully at {self.path}")
        except Exception as e:
            print(f"Error saving model: {e}")