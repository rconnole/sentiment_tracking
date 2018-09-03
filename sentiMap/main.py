##### imports #####
from sentiMap import utils


class LatentSemanticIndex:
    def __init__(self, stopwords):
        self.stopwords = stopwords


def main():
    # func = utils.Functions()
    # corpus = func.read_in_training()
    # func.remove_stopwords(corpus[0:5])
    sent = utils.SentimentAnalysis()
    sent.get_summary()

    print("Run finished")


if __name__ == '__main__':
    main()

