from matplotlib import colors

from sentiMap import controller
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, session, render_template, after_this_request
from flask_bootstrap import Bootstrap

graphDriver = controller.GraphDriver()
app = Flask(__name__)


def create_app():
    Bootstrap(app)
    return app


@app.route('/')
def index():
    users = graphDriver.get_top_posters(limit=10)
    results = []
    for index, row in users.iterrows():
        results.append([row['Username'], row['Posts']])

    topics = graphDriver.get_all_topic_sentiments()
    topicResults = []
    for index, row in topics.iterrows():
        topicResults.append([row['sent'], row['t']['Index']])
    return render_template('index.html', users=results, topics=topicResults)


@app.route('/sentiMap', methods=['GET', 'POST'])
def get_sentiment_data():
    summary = graphDriver.get_sentiment_summary()
    labels = 'Negative', "Positive", "Neutral"
    sizes = [summary["negative"], summary["positive"], summary["neutral"]]
    explode = (0, 0, 0)
    colors= ["red", "green", "brown"]

    img = io.BytesIO()
    plt.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("tags.html", pieChart=plot_url)


@app.route("/user/<username>", methods=['GET'])
def get_user_data(username):

    topics = graphDriver.get_users_topics_as_lists(username)
    negativeSenti = graphDriver.get_user_negative_posts(username)

    sentiOverTime = graphDriver.get_user_sentiment_over_time(username)
    labels = ["Sentiment", "Time"]
    points = []
    time = []
    print(sentiOverTime)
    for row in sentiOverTime:
        points.append(row['Doc.Sentiment'])
        time.append(row['Time'])
    print(points)
    print(time)

    return render_template("userData.html")

create_app()