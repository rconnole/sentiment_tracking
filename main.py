##### imports #####
import topicAnalysis


def main():
    print("Building topic model")
    lda = topicAnalysis.LdaWithTfidf()
    lda.read_test_as_bow()


    print("Run finished")


if __name__ == '__main__':
    main()


