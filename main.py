##### imports #####
import topicAnalysis


def main():
    print("Building topic model")
    lda = topicAnalysis.LdaWithTfidf(optimising=False)
    # lda.split_twitter_file()
    # lda.build_topic_model()
    lda.load_model_for_eval()


    print("Run finished")


if __name__ == '__main__':
    main()


