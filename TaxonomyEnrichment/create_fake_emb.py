import fasttext
from argparse import ArgumentParser

checkpoint = "/home/data/taxonomy/cc.en.300.bin"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", help="path to data", default=None)

    model = fasttext.load_model(checkpoint)
    result = np.vstack([model.get_sentence_vector(word) for word in batch])
