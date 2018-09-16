import re
import config
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import csv

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from .models import User, Document, create_node, graph, Topic, Relationship, matcher


class Functions:
    def __init__(self):
        self.stopwords_file = config.stop_word_file

    def remove_stopwords(self, target_corpus):
        for entry in target_corpus:
            entry[2] = [word for word in entry[2] if word not in stopwords.words('english')]
        print(target_corpus)

    def read_in_training(self, tokenize=True, limit=0):
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
                    # check optional arg
                    if tokenize:
                        sentence = tokenised
                except:
                    continue
                lines.append([timestamp, user, sentence, mentions])
                line_count += 1
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
        # print("This doc refers to topic: {}\n".format(topic_most_pr))
        return topic_most_pr

    def get_topics_from_model(self):
        return self.lda.components_, self.features


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

    def createPost(self, handle, sentiment, sentence, time, topic_indx, mentions=None):
        user = User(handle=handle, sentiment=sentiment)
        user_node = create_node(user)
        tokens = {}
        for token, tag in pos_tag(word_tokenize(sentence)):
            tokens[token] = tag
        document = Document(sentence=sentence, tokens=tokens, sentiment=sentiment, authoredBy=user_node,
                            happenedOn=time)
        doc = create_node(document)
        topic = matcher.match("Topic", Index=topic_indx).first()
        refersTo = Relationship(doc, "RefersTo", topic)
        graph.create(refersTo)

        if not mentions is None:
            if not mentions == []:
                for mention in mentions:
                    mentioned_user = User(handle=mention, sentiment=0)
                    mention_node = create_node(mentioned_user)
                    mentioned = Relationship(doc, "Mentioned", mention_node)
                    graph.create(mentioned)


    def add_topics_to_graph(self, components, features):
        for topic_idx, topic in enumerate(components):
            feature_tags = {}
            for token, tag in pos_tag(word_tokenize(" ".join([features[i] for i in topic.argsort()[:-config.num_of_topics - 1:-1]]))):
                feature_tags[token] = tag
            index = topic_idx
            topic = Topic(topic_idx, feature_tags)
            topic_node  = create_node(topic)

    def get_sentiment_summary(self):
        negativeScore = graph.run("MATCH (Document) WHERE Document.Sentiment = 0 RETURN count(Document)").evaluate()
        positiveScore = graph.run("MATCH (Document) WHERE Document.Sentiment > 0 RETURN count(Document)").evaluate()
        neutralScore = graph.run("MATCH (Document) WHERE Document.Sentiment < 0 RETURN count(Document)").evaluate()
        return {"positive": positiveScore, "neutral": neutralScore, "negative": negativeScore}

    def get_avg_sentiment_about_topic(self, topic_index):
        query = f"MATCH (d:Document)-[r:RefersTo]->(t) where t.Index = \"{topic_index}\" WITH t, avg(d.Sentiment) AS sent RETURN t ,sent"
        results = graph.run(query).to_data_frame()
        return results

    def get_all_topic_sentiments(self):
        query = f"MATCH (d:Document)-[r:RefersTo]->(t) WITH t, avg(d.Sentiment) AS sent RETURN t ,sent"
        results = graph.run(query).to_data_frame()
        return results

    def get_users_topics_as_lists(self, user, limit=25):
        query = f"MATCH (n:User)-[r:AuthoredBy]-(d:Document) WITH n, count(r) AS Posts ORDER BY Posts DESC LIMIT {limit} RETURN n.Handle, Posts"
        results = graph.run(query).to_data_frame()
        return results

    def get_topic_distributions(self):
        query = f"MATCH (d:Document)-[r:RefersTo]->(t) WITH t, count(r) AS REFS ORDER BY REFS DESC RETURN t ,REFS"
        results = graph.run(query).to_data_frame()
        return results

    def get_documents_by_topic(self, topic, limit=25):
        query =f"MATCH (d:Document)-[r:RefersTo]->(t) where t.Index = \"{topic.topic_index}\" RETURN d LIMIT {limit}"
        results = graph.run(query).to_data_frame()
        return results

    def get_user_negative_posts(self, user, limit=25):
        query = f"MATCH (n:User)-[r:AuthoredBy]-(d:Document) Where n.Handle = \"{user}\" WITH n, d AS Posts ORDER BY Posts.Sentiment DESC LIMIT {limit} RETURN n.Handle, Posts"
        results = graph.run(query).to_data_frame()
        return results

    def get_all_posts_by_user_over_time(self, user, limit=25):
        query = f"MATCH (n:User)-[r:AuthoredBy]-(d:Document)-[h:HappenedOn]-(t:CreationDate) Where n.Handle = \"{user}\" WITH t.Time as Time, d.Sentence AS Sentence ORDER BY Time DESC LIMIT {limit} RETURN Sentence, Time"
        results = graph.run(query).to_data_frame()
        return results

    def get_user_sentiment_over_time(self, user, limit=25):
        query = f"MATCH (n:User)-[r:AuthoredBy]-(d:Document)-[h:HappenedOn]-(t:CreationDate) Where n.Handle = \"{user}\" WITH t.Time as Time, d.Sentiment AS Sentiment ORDER BY Time DESC LIMIT {limit} RETURN Sentiment, Time"
        results = graph.run(query).to_data_frame()
        return results

    def get_top_posters(self, limit=25):
        query = f"MATCH (n:User)-[r:AuthoredBy]-(d:Document) WITH n.Handle AS Username, count(r) AS Posts ORDER BY Posts DESC LIMIT {limit} RETURN Username, Posts"
        results = graph.run(query).to_data_frame()
        return results

    def get_all_posts_containing_token(self, token):
        query = "fMATCH (d:Document)-[r:Contains]-(w:Word) Where w.Token = \"{token}\" WITH d RETURN d"
        results = graph.run(query).to_data_frame()
        return results


