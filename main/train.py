from .fine_tune._fine_tune_model import FineTuneModel

PATH = 'bin/model/model.pt'
FILE_PATH = 'data/wiki_00.txt'

model_loader = FineTuneModel(checkpoint_path=PATH)

encoded = model_loader.encode_file(file_path=FILE_PATH)

model_loader.train(encoded)

model_loader.save()