from .views import app
from .models import graph, Node

from sentiMap import controller
from sentiMap.dbSetup import add_test_corpus_to_db

# Uncomment to add entire corups to graph
# add_test_corpus_to_db()
