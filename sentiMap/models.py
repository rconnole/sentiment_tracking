from py2neo import Graph, Node, Relationship, NodeMatcher
from neo4j.exceptions import ConstraintError
graph = Graph()
matcher = NodeMatcher(graph)


### Data Model ###############################################################
class Word:
    def __init__(self, token, tag):
        self.Token = token
        self.Tag = tag

    def find(self):
        word = graph.match_one("Word", "Token", self.Token)
        return word

    def find_by_tag(self):
        word = graph.match_one("Word", "Tag", self.Tag)


class Document:
    def __init__(self, sentence, tokens, sentiment, authoredBy, happenedOn):
        self.Sentence = sentence
        self.Tokens = tokens
        self.Sentiment = sentiment
        self.AuthoredBy = authoredBy
        self.HappendOn = happenedOn

    def find(self):
        doc = graph.match_one("Document", "Sentence", self.Sentence)
        return doc


class User:
    def __init__(self, handle, sentiment):
        self.Handle = handle
        self.Sentiment = sentiment

    def find(self):
        print(graph)
        print(graph.exists())
        user = graph.exists("User", "Handle", self.Handle)
        return user

    def mentions(self, handle):
        return

    def wrote_document(self, document):
        document.create(self)
        return

    def average_sentiment(self, new_sentiment):
        return


class CreationDate:
    def __init__(self, time):
        self.Time = time

    def find(self):
        time = graph.match_one("CreationDate", "Time", self.Time)
        return time


class Topic:
    def __init__(self, index, feat_names):
        self.topic_index = f"Topic {index}"
        self.feature_names = feat_names


def create_node(o):

    if isinstance(o, User):
        user = Node("User", Handle=o.Handle, Sentiment=o.Sentiment)
        try:
            graph.create(user)
        except ConstraintError:
            user = matcher.match("User", Handle=o.Handle).first()
            oldSenti = user["Sentiment"]
            newSenti = o.Sentiment
            user["Sentiment"] = round(oldSenti + newSenti)/2
            graph.merge(user)
        return user

    elif isinstance(o, Document):
        doc = Node("Document", Sentence=o.Sentence, Sentiment=o.Sentiment)
        graph.create(doc)
        writtenBy = Relationship(o.AuthoredBy, "AuthoredBy", doc)
        graph.create(writtenBy)

        time = Node("CreationDate", Time=o.HappendOn)
        try:
            graph.create(time)
        except ConstraintError:
            time = matcher.match("CreationDate", Time=o.HappendOn).first()
        happendOn = Relationship(doc, "HappenedOn", time)
        graph.create(happendOn)

        for token in o.Tokens.keys():
            word = Node("Word", Token=token, Tag=o.Tokens[token])
            try:
                graph.create(word)
            except ConstraintError:
                word = matcher.match("Word", Token=token).first()
            contains = Relationship(doc, "Contains", word)
            graph.create(contains)
        return doc

    elif isinstance(o, Topic):
        topic = Node("Topic", Index=o.topic_index, Features=[].append(feature for feature in o.feature_names.keys()))
        try:
            graph.create(topic)
            print("Topic Created: ", o.topic_index)
        except ConstraintError:
            topic = matcher.match("Topic", Index=o.topic_index).first()
        for token in o.feature_names.keys():
            word = Node("Word", Token=token, Tag=o.feature_names[token])
            try:
                graph.create(word)
            except ConstraintError:
                word = matcher.match("Word", Token=token).first()
            containsTerm= Relationship(topic, "containsTerm", word)
            graph.create(containsTerm)
        return topic
