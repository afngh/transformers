import sys
from .fine_tune._fine_tune_model import FineTuneModel

PATH = 'bin/model/model.pt'

if len(sys.argv) < 2:
    print("Usage: python -m main.train <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

model_loader = FineTuneModel(checkpoint_path=PATH)
model_loader.train(file_path=file_path)
model_loader.save()
