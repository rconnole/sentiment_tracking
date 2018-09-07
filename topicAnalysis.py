import re
import config
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


class LdaWithTfidf:
    def __init__(self, optimising=False, max_df=config.max_document_freq, min_df=config.min_document_freq, max_feat=config.max_features):
        self.optimising = optimising
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'[a-zA-z]+')
        self.vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_feat,
                                          min_df=min_df, stop_words='english', use_idf=True)

    def read_in_training_as_bow(self):
        print("File directory found at " + config.unity_training_file)
        timestamp_regex = "\[\d\d\:\d\d\]"
        timestamp_pattern = re.compile(timestamp_regex)
        user_regex = "<([a-zA-Z0-9_ ]+)>"
        user_pattern = re.compile(user_regex)
        bag_of_words_output = []
        with open(config.unity_training_file, "rt", encoding="utf-8") as training:
            for line in training:
                output = line
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    output = re.sub(timestamp_regex, "", output)
                user_match = user_pattern.search(line)
                if user_match:
                    output = re.sub(user_regex, "", output)
                bag_of_words_output.append(self.apply_lemmatizer(output))
        print("Training Data Entries: {}".format(len(bag_of_words_output)))
        return bag_of_words_output

    def read_test_as_bow(self):
        print("File directory found at " + config.unity_test_file)
        timestamp_regex = "\[\d\d\:\d\d\]"
        timestamp_pattern = re.compile(timestamp_regex)
        user_regex = "<([a-zA-Z0-9_ ]+)>"
        user_pattern = re.compile(user_regex)
        bag_of_words_output = []
        with open(config.unity_test_file, "rt", encoding="utf-8") as training:
            for line in training:
                print("line is: ", line)
                output = line
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    output = re.sub(timestamp_regex, "", output)
                user_match = user_pattern.search(line)
                if user_match:
                    output = re.sub(user_regex, "", output)
                bag_of_words_output.append(self.apply_lemmatizer(output))
        print("Training Data Entries: {}".format(len(bag_of_words_output)))
        return bag_of_words_output

    def apply_lemmatizer(self, sentence):
        temp = [self.lemmatizer.lemmatize(t) for t in self.tokenizer.tokenize(sentence)]
        # join tokens as vectorizer will split them again
        lemmas = " ".join(temp)
        return lemmas

    def build_topic_model(self, num_of_topics=config.num_of_topics, max_iterations=config.max_iterations):
        corpus = self.read_in_training_as_bow()
        if self.optimising:
            corpus = self.read_test_as_box()
        training_tfidf = self.vectorizer.fit_transform(corpus)
        print("Actual number of tfidf features: %d" % training_tfidf.get_shape()[1])
        feat_names = self.vectorizer.get_feature_names()
        print("First 10 words in the vocabulary: %s", feat_names[0:10])
        # Run LDA
        lda = LatentDirichletAllocation(n_components=num_of_topics, max_iter=max_iterations, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(training_tfidf)
        for topic_idx, topic in enumerate(lda.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feat_names[i] for i in topic.argsort()[:-num_of_topics - 1:-1]]))

        ldaModel = lda.fit_transform(training_tfidf)
        model = (feat_names, lda.components_, lda.exp_dirichlet_component_, lda.doc_topic_prior_)

        if not self.optimising:
            # if used in production save the model
            print(f"Saving model to file: {config.lda_model_file_name}")
            with open(config.lda_model_file_name, 'wb') as fp:
                joblib.dump(model, fp)
        else:
            self.evaluate_model(model)

    def evaluate_model(self, model):

        # # when optimising output graphs
        # for compNum in range(0, self.num_of_topics):
        #     comp = svd.components_[compNum]
        #     indeces = np.argsort(comp).tolist()
        #     indeces.reverse()
        #     terms = [feat_names[weightIndex] for weightIndex in indeces[0:10]]
        #     weights = [comp[weightIndex] for weightIndex in indeces[0:10]]
        #     terms.reverse()
        #     weights.reverse()
        #     positions = np.arange(10) + .5
        #     # plot strongest terms for each term
        #     plt.plot(compNum)
        #     plt.barh(positions, weights, align='center')
        #     plt.yticks(positions, terms)
        #     plt.xlabel('Weight')
        #     plt.title('Strongest terms for component %d' % compNum)
        #     plt.grid(True)
        #     plt.show()
        return
