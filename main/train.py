import sys
from .fine_tune._fine_tune_model import FineTuneModel

PATH = 'bin/model/model.pt'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m main.train <file_path> [<start_chunk>]")
        sys.exit(1)

    file_path = sys.argv[1]
    start_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else None

    model_loader = FineTuneModel(checkpoint_path=PATH)
    model_loader.train(file_path=file_path, start_chunk=start_chunk)
    model_loader.save()
