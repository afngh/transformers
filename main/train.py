import argparse
from .fine_tune._fine_tune_model import FineTuneModel

PATH = 'bin/model/model.pt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Dynamo transformer model on BPE chunks.")
    parser.add_argument("file_path", type=str, help="Path to pre-tokenized training corpus (.pt file)")
    parser.add_argument("-s", "--start_chunk", type=int, default=None, help="Chunk index to start/resume training from")
    parser.add_argument("-v", "--val_path", type=str, default="bin/data/simple_wiki_val.pt", help="Path to pre-tokenized validation file (.pt file)")

    args = parser.parse_args()

    model_loader = FineTuneModel(checkpoint_path=PATH)
    model_loader.train(file_path=args.file_path, start_chunk=args.start_chunk, val_path=args.val_path)
    model_loader.save()
