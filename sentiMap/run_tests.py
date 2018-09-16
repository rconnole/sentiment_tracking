from sentiMap import controller
import config
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

graphDriver = controller.GraphDriver()


def test_corpus():
    get_avg_sentiment()

def test_vader_gold_standard():
    print("File directory found at " + config.twitter_training_file)
    word_dist = {}
    lines = []
    tokenizer = RegexpTokenizer(r'\w+')
    with open("../" + config.twitter_training_file, "rt", encoding="utf-8") as training:
        reader = csv.reader(training, delimiter=',')
        line_count = 0
        for line in reader:
            if line_count == 0:
                print(f'column names are {",".join(line)}')
                line_count += 1
                continue
            try:
                gold_standard_sentiment = line[1]
                output = line[10]
                sentence = output
                tokenised = tokenizer.tokenize(output)
                for token in tokenised:
                    if token in word_dist:
                        word_dist[token] += 1
                    else:
                        word_dist[token] = 1
            except:
                continue
            lines.append([gold_standard_sentiment, sentence])

    print(word_dist.keys())

    hist = plt.hist(word_dist.values(), bins=len(word_dist.keys()))
    # plt.show()
    plt.close()


    analyser = SentimentIntensityAnalyzer()
    results = {'falseNeg' : 0, 'falsePos': 0, "trueNeg": 0, "truePos": 0}
    for tweet in lines:
        snt = analyser.polarity_scores(tweet[1])
        if snt["compound"] >= 0.0:
            if tweet[0] == 'positive' or tweet[0] == "neutral":
                results["truePos"] += 1
            else:
                results["falsePos"] += 1
        else:
            if tweet[0] == 'negative':
                results["trueNeg"] += 1
            else:
                results["falseNeg"] += 1

    precision = results["truePos"] / (results["truePos"] + results["falsePos"])
    recall = results["truePos"] / (results["truePos"] + results["falseNeg"])
    fMeasure = 2 * ((precision * recall)/(precision + recall))

    plt.bar(["Precision", "Recall", "F-Measure"], [precision, recall, fMeasure], color=["blue", "red", "green"])
    plt.title("Metrics for Twitter Corpus")
    plt.savefig(config.results_dir + "corpus_metrics.png")
    plt.close()

def get_avg_sentiment():
    summary = graphDriver.get_sentiment_summary()
    labels = 'Negative', "Positive", "Neutral"
    sizes = [summary["negative"], summary["positive"], summary["neutral"]]
    explode = (0, 0, 0)
    colors = ["red", "green", "brown"]

    plt.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Sentiment deistribution for Corpus")
    plt.savefig(config.results_dir + "corpus_average_sentiment.png")
    plt.close()


def get_top_users():
    users = graphDriver.get_top_posters(limit=15)
    results = {}
    for index, row in users.iterrows():
        results[row['Username']] = row['Posts']
    plt.bar(x=range(len(results.values())), height=list(results.values()), align='center')
    plt.xticks(range(len(results.values())), list(results.keys()), rotation=70)
    plt.title("Top Users by Posts")
    plt.ylabel("Number of Posts")
    plt.xlabel("Usernames")
    plt.savefig(config.results_dir + "top_users_by_posts.png")
    plt.close()


def get_top_user_sentiment():
    users = graphDriver.get_top_posters(limit=10)
    results = {}
    for index, row in users.iterrows():
        results[row['Username']] = [[], []]
    for user in results.keys():
        sentiment = graphDriver.get_user_sentiment_over_time(user)
        sentiment.sort_values(by=["Time"])
        dates = []
        for date in sentiment["Time"].tolist():
            dates.append(datetime.strptime(date[:-6], '%Y-%m-%d %H:%M:%S'))

        x = mdates.date2num(dates)
        y = sentiment["Sentiment"].tolist()
        fig = plt.plot_date(x, y, "-")
        plt.title("Graph of user: " + user + " Sentiment Over Time")
        plt.legend(["Sentiment", "TrendLine"])
        plt.xticks(rotation=15)
        plt.xlabel("Time")
        plt.ylabel("Sentiment")
        z = numpy.polyfit(x, y, 1)
        p = numpy.poly1d(z)
        plt.plot(x, p(x), "r-", linestyle="dashed")
        plt.savefig(config.results_dir + "user_sentiment_" + user + ".png")
        plt.close()

def get_sentiment_for_topics():
    topics = graphDriver.get_all_topic_sentiments()
    topicResults = []
    for index, row in topics.iterrows():
        topicResults.append([row['sent'], row['t']['Index']])
        plt.bar(row['t']['Index'], row["sent"])
    plt.title("Topic Sentiments for entire corpus")
    plt.xticks(rotation=70)
    plt.xlabel("Topics")
    plt.ylabel("Sentiment")
    plt.savefig(config.results_dir + "topic_sentiments.png")

# get_avg_sentiment()
# get_top_user_sentiment()
test_vader_gold_standard()
# test_corpus()
# get_sentiment_for_topics()
# get_top_user_sentiment()
# get_top_users()