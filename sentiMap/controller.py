import re
import config
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from .models import User, Document, create_node, graph


class Functions:
    def __init__(self):
        self.stopwords_file = config.stop_word_file

    def remove_stopwords(self, target_corpus):
        for entry in target_corpus:
            entry[2] = [word for word in entry[2] if word not in stopwords.words('english')]
        print(target_corpus)

    def read_in_training(self, tokenize=True, limit=0):
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

        if limit >= 1:
            return lines[0:limit]
        return lines


class TopicAnalysis:
    def __init__(self):
        self.lda = LatentDirichletAllocation()
        (self.features, self.lda.components_, self.lda.exp_dirichlet_component_, self.lda.doc_topic_prior_) = joblib.load(config.lda_model_file_name)
        self.tf_vectorizer = CountVectorizer(vocabulary=self.features, stop_words='english')

    def get_topics_from_post(self, sentence):
        tf_sentence = self.tf_vectorizer.fit_transform([sentence])
        post_topic = self.lda.transform(tf_sentence)

        topic_most_pr = post_topic[0].argmax()
        print("This doc refers to topic: {}\n".format(topic_most_pr))
        return topic_most_pr


class SentimentAnalysis:
    def __init__(self):
        self.func = Functions()
        self.analyser = SentimentIntensityAnalyzer()

    def get_sentiment_entire_corpus(self, corpusList):
        sentiment = []
        for sentence in corpusList:
            snt = self.analyser.polarity_scores(sentence[2])
            sentiment.append([sentence[0], sentence[1], sentence[2], snt])
        return sentiment

    def get_sentiment_one_post(self, sentence):
        return self.analyser.polarity_scores(sentence)

    def get_summary(self, corpus):
        sentimentList = self.get_sentiment_entire_corpus(corpus)

        summary = {"positive": 0, "neutral": 0, "negative": 0}
        for snt in sentimentList:
            if snt[3]["compound"] == 0.0:
                summary["neutral"] += 1
            elif snt[3]["compound"] > 0.0:
                summary["positive"] += 1
            else:
                summary["negative"] += 1
        print(summary)
        return summary


class GraphDriver:
    # for use with the API and Web page calls
    def __init__(self):
        self.graph = graph

    def createPost(self, handle, sentiment, sentence, time):
        user = User(handle=handle, sentiment=sentiment)
        user_node = create_node(user)
        tokens = {}
        for token, tag in pos_tag(word_tokenize(sentence)):
            tokens[token] = tag
        document = Document(sentence=sentence[2], tokens=tokens, sentiment=sentiment, authoredBy=user_node,
                            happenedOn=time)
        create_node(document)

    def get_sentiment_summary(self):
        negativeScore = graph.run("MATCH (Document) WHERE Document.Sentiment = 0 RETURN count(Document)").evaluate()
        positiveScore = graph.run("MATCH (Document) WHERE Document.Sentiment > 0 RETURN count(Document)").evaluate()
        neutralScore = graph.run("MATCH (Document) WHERE Document.Sentiment < 0 RETURN count(Document)").evaluate()
        return {"positive": positiveScore, "neutral": neutralScore, "negative": negativeScore}

    def get_user_sentiment(self):

        return {}

