from .views import app
from .models import graph, Node

from sentiMap import controller


def create_uniqueness_constraint(label, property):
    print(f"Adding Uniqueness Constraint on Property {property} for Object {label}")
    query = "CREATE CONSTRAINT ON (n:{label}) ASSERT n.{property} IS UNIQUE"
    query = query.format(label=label, property=property)
    graph.run(query)


def create_index_on_node (label, property):
    print(f"Adding index on Property {property} for Object {label}")
    query = "CREATE INDEX ON :label(property)"
    query = query.format(label=label, property=property)
    graph.run(query)


def add_corpus_to_db():

    print("***** parsing corpus *****")
    func = controller.Functions()
    senti = controller.SentimentAnalysis()
    driver = controller.GraphDriver()
    lda = controller.TopicAnalysis()
    add_topics_to_db(lda, driver)
    corpus = func.read_in_training(False)
    print(f"Sentiment Data for Corpus is {senti.get_summary(corpus)}")

    for post in corpus:
        snt = senti.get_sentiment_one_post(post[2])
        closest_topic = f"Topic {lda.get_topics_from_post(post[2])}"
        driver.createPost(post[1], snt["compound"], post[2], post[0], closest_topic, post[0])
    print("***** Startup Finished *****")


def add_topics_to_db(ldaModel, graphDriver):
    comp, feats = ldaModel.get_topics_from_model()
    graphDriver.add_topics_to_graph(comp, feats)


def graph_setup():
    print("***** Cleaning Graph *****")
    query = "MATCH (n) DETACH DELETE n"
    graph.run(query)
    # Add constraints on graph objects
    create_uniqueness_constraint("User", "Handle")
    create_index_on_node("User", "Handle")
    create_uniqueness_constraint("Word", "Token")
    create_index_on_node("Word", "Token")
    create_uniqueness_constraint("CreationDate", "Time")
    create_index_on_node("Topic", "Index")
    create_uniqueness_constraint("Topic", "Index")


# used for graph setup
def add_test_corpus_to_db():
    graph_setup()
    add_corpus_to_db()
