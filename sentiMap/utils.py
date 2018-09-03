import re
import config
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .models import User, Document, Word, TimeSlice, create_node

class Functions:
    def __init__(self):
        self.stopwords_file = config.stop_word_file

    def remove_stopwords(self, target_corpus):
        for entry in target_corpus:
            entry[2] = [word for word in entry[2] if word not in stopwords.words('english')]
        print(target_corpus)

    def read_in_training(self, tokenize=True):
        print("File directory found at " + config.unity_training_file)
        lines = []
        timestamp_regex = "\[\d\d\:\d\d\]"
        timestamp_pattern = re.compile(timestamp_regex)
        user_regex = "<([a-zA-Z0-9_ ]+)>"
        user_pattern = re.compile(user_regex)
        actual_lines = []
        tokenizer = RegexpTokenizer(r'\w+')
        with open(config.unity_training_file, "rt", encoding="utf-8") as training:
            for line in training:
                timestamp = ""
                user = ""
                output = line
                timestamp_match = timestamp_pattern.search(line)
                if timestamp_match:
                    output = re.sub(timestamp_regex, "", output)
                    timestamp = timestamp_match.group(0)
                    timestamp = timestamp[1:-1]
                user_match = user_pattern.search(line)
                if user_match:
                    output = re.sub(user_regex, "", output)
                    user = user_match.group(1)
                sentence = output
                # check optional arg
                if tokenize:
                    sentence = tokenizer.tokenize(output)
                actual_lines.append(output)
                lines.append([timestamp, user, sentence])
        print("Training Data Entries: {}".format(len(actual_lines)))
        return lines


class SentimentAnalysis:
    def __init__(self):
        self.func = Functions()
        self.analyser = SentimentIntensityAnalyzer()
        self.sentiments = {}

    def get_sentiment(self):
        corpus = self.func.read_in_training(False)
        sentiment = []
        for sentence in corpus[0:100]:
            snt = self.analyser.polarity_scores(sentence[2])
            tokens = self.get_tags(sentence[2])
            # print("{:-<40} {}".format(sentence, str(snt)))
            sentiment.append([sentence[0], sentence[1], sentence[2], snt])
            user = User(handle=sentence[1], sentiment=snt["compound"])
            user_node = create_node(user)
            document = Document(sentence=sentence[2], tokens=tokens, sentiment=snt["compound"], authoredBy=user_node, happenedOn=sentence[0])
            # document.create()
            create_node(document)
            # if not sentence[1] in self.sentiments.keys():
            #
            #      = UserSentiment(sentence[0], sentence[1], tags, snt)
            #     self.sentiments[userSentiment.handle] = userSentiment
            # else:
            #     self.sentiments[sentence[1]].add_words(tags)
            #     self.sentiments[sentence[1]].add_sentiment(snt)
        return sentiment
    #
    # def create_user_sentiment(self, user, sentence, time):
    #     snt = self.analyser.polarity_scores(sentence)
    #     tags = self.get_tags(sentence[2])
    #     return UserSentiment(user, time, tags, snt)

    def get_summary(self):
        sentiment = self.get_sentiment()
        summary = {"positive": 0, "neutral": 0, "negative": 0}
        for snt in sentiment:
            if snt[3]["compound"] == 0.0:
                summary["neutral"] += 1
            elif snt[3]["compound"] > 0.0:
                summary["positive"] += 1
            else:
                summary["negative"] += 1
        print(summary)

    def get_tags(self, sentence):
        tokens = {}
        for token, tag in pos_tag(word_tokenize(sentence)):
            tokens[token] = tag
        return tokens
