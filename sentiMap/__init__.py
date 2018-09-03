from .views import app
from .models import graph, Node


def create_uniqueness_constraint(label, property):
    query = "CREATE CONSTRAINT ON (n:{label}) ASSERT n.{property} IS UNIQUE"
    query = query.format(label=label, property=property)
    graph.run(query)

# clean graph
query = "MATCH (n) DETACH DELETE n"
graph.run(query)

# TODO: make this mine
create_uniqueness_constraint("User", "Handle")
create_uniqueness_constraint("Word", "Token")
# create_uniqueness_constraint("Document", "Sentence")
create_uniqueness_constraint("TimeSlice", "Time")



