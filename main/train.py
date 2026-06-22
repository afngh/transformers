from .fine_tune._fine_tune_model import FineTuneModel

PATH = 'bin/model/model.pt'
FILE_PATH = 'data/resources.txt'

model_loader = FineTuneModel(checkpoint_path=PATH)

model_loader.train(file_path=FILE_PATH)

model_loader.save()
