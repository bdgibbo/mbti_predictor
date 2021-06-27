import argparse
import pandas as pd
import numpy as np
# model testing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# reversely translate a binary representation back to the lettered mbti
def reverseTranslate(binaryType):
    yReverseMap = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]
    mbtiType = ""
    for eachFunction, eachFunctionMapping in enumerate(binaryType):
        mbtiType += yReverseMap[eachFunction][eachFunctionMapping]
    return mbtiType


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data",
                        default="mbti_preprocessed.csv",
                        help="filename for the mbti dataset")

    args = parser.parse_args()
    # load the data
    xFeat = pd.read_csv(args.data)

    X = xFeat.drop(['type', 'posts', 'I-E', 'N-S', 'T-F', 'J-P'], axis=1).values
    y = []

    example = [0, 0, 0, 0]
    print(reverseTranslate(example))

    # translate each type to binary representation
    yMap = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'T': 0, 'F': 1, 'J': 0, 'P': 1}
    for eachType in xFeat['type']:
        yBinary = [yMap[eachFunction] for eachFunction in eachType]
        # print(yBinary)
        y.append(yBinary)
    y = np.array(y)

    # Logistic Regression
    # lr = LogisticRegression().fit(xTrain, yTrain)
    # y_pred1 = lr.predict(xTest)
    # print("Logistic Regression: ", accuracy_score(yTest, y_pred1))
    #
    # Random Forest
    for i in range(4):
        xTrain, xTest, yTrain, yTest = train_test_split(X, y[:, i], test_size=0.1, random_state=5)
        rf = RandomForestClassifier().fit(xTrain, yTrain)
        y_pred2 = rf.predict(xTest)
        print("Random Forest: ", accuracy_score(yTest, y_pred2))
    #
    # Neural Network
    # nn = MLPClassifier().fit(xTrain, yTrain)
    # y_pred3 = nn.predict(xTest)
    # print("Neural Network: ", accuracy_score(yTest, y_pred3))

    # SGD


if __name__ == "__main__":
    main()
