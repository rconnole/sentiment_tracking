from .views import app
from .models import graph, Node

from sentiMap import utils


def create_uniqueness_constraint(label, property):
    query = "CREATE CONSTRAINT ON (n:{label}) ASSERT n.{property} IS UNIQUE"
    query = query.format(label=label, property=property)
    graph.run(query)


def create_index_on_node (label, property):
    query = "CREATE INDEX ON :label(property)"
    query = query.format(label=label, property=property)
    graph.run(query)


def add_corpus_to_db():
    print("***** Cleaning Graph *****")
    query = "MATCH (n) DETACH DELETE n"
    graph.run(query)
    print("***** parsing corpus *****")
    func = utils.Functions()
    senti = utils.SentimentAnalysis()
    driver = utils.GraphDriver()
    corpus = func.read_in_training(False, limit=500)
    print("Sentiment Data for Corpus is {}", senti.get_summary(corpus))
    for post in corpus:
        snt = senti.get_sentiment_one_post(post[2])
        driver.createPost(post[1], snt["compound"], post[2], post[0])
    print("***** Startup Finished *****")


# Add constraints on graph objects
create_uniqueness_constraint("User", "Handle")
create_index_on_node("User", "Handle")
create_uniqueness_constraint("Word", "Token")
create_index_on_node("Word", "Token")
create_uniqueness_constraint("TimeSlice", "Time")

add_corpus_to_db()




