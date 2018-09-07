#from .models import User, get_todays_recent_posts
from sentiMap import controller
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, session, render_template, after_this_request
app = Flask(__name__)

graphDriver = controller.GraphDriver()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiMap', methods=['GET', 'POST'])
def startSentiMap():
    summary = graphDriver.get_sentiment_summary()
    labels = 'Negative', "Positive", "Neutral"
    sizes = [summary["negative"], summary["positive"], summary["neutral"]]
    explode = (0, 0.1, 0)

    img = io.BytesIO()
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("tags.html", pieChart=plot_url)
