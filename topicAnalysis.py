import re
import csv
import config
import random
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


class LdaWithTfidf:
    def __init__(self, optimising=True, max_df=config.max_document_freq, min_df=config.min_document_freq, max_feat=config.max_features):
        self.optimising = optimising
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'[a-zA-z]+')
        self.vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_feat,
                                          min_df=min_df, stop_words='english', use_idf=True)


    def split_twitter_file(self):
        print("Dividing file into 70% training and 30% testing")
        allTweets = []
        columnnames = []
        with open(config.twitter_orig_file, "rt", encoding="utf-8") as original:
            line_count = 0
            for line in original:
                if line_count == 0:
                    columnnames.append(line)
                    line_count += 1
                    continue
                allTweets.append(str(line))
        print(columnnames)
        random.shuffle(allTweets)
        with open(config.twitter_training_file, 'wt', encoding="utf-8") as training:
            training.write(columnnames[0])
            for line in allTweets[0:int(len(allTweets) * 0.70)]:
                training.write(line)
        with open(config.twitter_test_file, 'wt', encoding="utf-8") as test:
            test.write(columnnames[0])
            for line in allTweets[int(len(allTweets) * 0.70):]:
                test.write(line)


    def read_twitter_training_dataset(self, limit=0):
        print("File directory found at " + config.twitter_training_file)
        lines = []
        tokenizer = RegexpTokenizer(r'\@\w+')
        with open(config.twitter_training_file, "rt", encoding="utf-8") as training:
            reader = csv.reader(training, delimiter=',')
            line_count = 0
            for line in reader:
                if line_count == 0:
                    print(f'column names are {",".join(line)}')
                    line_count += 1
                    continue
                try:
                    timestamp = line[12]
                    user = line[7]
                    output = line[10]
                    sentence = output
                    mentions = []
                    tokenised = tokenizer.tokenize(output)
                    for token in tokenised:
                        if token.startswith('@'):
                            mentions.append(token[1:])

                except:
                    continue
                lines.append(self.apply_lemmatizer(sentence))
                line_count += 1
        if limit >= 1:
            return lines[0:limit]
        return lines

    def read_twitter_testing_dataset(self, limit=0):
        print("File directory found at " + config.twitter_training_file)
        lines = []
        tokenizer = RegexpTokenizer(r'\@\w+')
        with open(config.twitter_training_file, "rt", encoding="utf-8") as training:
            reader = csv.reader(training, delimiter=',')
            line_count = 0
            for line in reader:
                if line_count == 0:
                    print(f'column names are {",".join(line)}')
                    line_count += 1
                    continue

                timestamp = line[12]
                user = line[7]
                output = line[10]
                sentence = output
                mentions = []
                tokenised = tokenizer.tokenize(output)
                for token in tokenised:
                    if token.startswith('@'):
                        mentions.append(token[1:])
                lines.append(self.apply_lemmatizer(sentence))
                line_count += 1
        if limit >= 1:
            return lines[0:limit]
        return lines

    def read_unity_dataset_training(self):
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

    def read_unity_dataset_testing(self):
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
        corpus = self.read_twitter_training_dataset()
        print(corpus)
        if self.optimising:
            corpus = self.read_twitter_testing_dataset()
        training_tfidf = self.vectorizer.fit_transform(corpus)
        print("number of tfidf features: %d" % training_tfidf.get_shape()[1])
        feat_names = self.vectorizer.get_feature_names()
        # Run LDA
        lda = LatentDirichletAllocation(n_components=num_of_topics, max_iter=max_iterations, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(training_tfidf)

        ldaModel = lda.fit_transform(training_tfidf)
        model = (feat_names, lda.components_, lda.exp_dirichlet_component_, lda.doc_topic_prior_)

        if not self.optimising:
            # if used in production save the model
            print(f"Saving model to file: {config.lda_model_file_name}")
            with open(config.lda_model_file_name, 'wb') as fp:
                joblib.dump(model, fp)
        else:
            self.evaluate_model(model,training_tfidf)

    def load_model_for_eval(self):
        model =( features, components_, exp_dirichlet_component_, doc_topic_prior_) = joblib.load(config.lda_model_file_name)
        # self.tf_vectorizer = CountVectorizer(vocabulary=self.features, stop_words='english')
        self.evaluate_model(model)

    def evaluate_model(self, model, data=None):
        (features, components_, exp_dirichlet_component_, doc_topic_prior_) = model
        print(f"First 10 words in the vocabulary: %s", features[0:10])

        # Print words associated with each topic
        for topic_idx, topic in enumerate(components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([features[i] for i in topic.argsort()[:-config.num_of_topics - 1:-1]]))

        if not data is None:
            # extract from vectrorised data
            termFreq = data.sum(axis=0).getA1()
            docLength = data.sum(axis=1).getA1()
            termDists = components_ / components_.sum(axis=1)[:, None]
            #docTopicDists =
            print("Data present, ", termFreq, docLength)

        # when optimising output graphs
        for compNum in range(0, config.num_of_topics):
            comp = components_[compNum]
            indeces = np.argsort(comp).tolist()
            indeces.reverse()
            terms = [features[weightIndex] for weightIndex in indeces[0:10]]
            weights = [comp[weightIndex] for weightIndex in indeces[0:10]]
            terms.reverse()
            weights.reverse()
            positions = np.arange(10) + .5

            # plot strongest terms for each term
            plt.plot(compNum)
            plt.barh(positions, weights, align='center')
            plt.yticks(positions, terms)
            plt.xlabel('Weight')
            plt.title('Strongest terms for component %d' % compNum)
            plt.grid(True)
            plt.savefig(config.topics_results_dir + "topic%d" % compNum + ".png")
            plt.close()

        return
