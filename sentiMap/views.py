#from .models import User, get_todays_recent_posts
from sentiMap import utils
from flask import Flask, request, session, render_template
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiMap', methods=['POST'])
def startSentiMap():
    sent = utils.SentimentAnalysis()
    sent.get_summary()

    print("Run finished")
    return "HTTP - 200 SUCCESS"
