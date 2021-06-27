import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string as string

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def calc_mistakes(yHat, yTrue):
    err = 0
    for index in range(len(yHat)):
        if yHat[index] != yTrue[index]:
            err += 1
    return err

def split_data(data):
    X = data.iloc[:,0:759]
    Y = data.iloc[:, 760:772]
    Y = Y.drop(['type', 'posts', 'avg_countOf_words', 'countOf_links', 'countOf_pics', 'countOf_exclamations', 'countOf_videos', 'sentiment_score'], axis=1).values
    #Y_IE = data[:, 797]
    #Y_NS = data[:, 798]
    #Y_TF = data[:, 799]
    #Y_JP = data[:, 800]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state = 470)
    return X, Y, xTrain, xTest, yTrain, yTest

def run_models(models, xTrain, yTrain, xTest, yTest):
    stats = pd.DataFrame(columns=['model', 'type-test', 'train-score', 'test-score'])
    #trainErr = calc_mistakes(yHatTrain, yTrain)
    #testErr = calc_mistakes(yHatTest, yTest)
    #train_acc = accuracy_score(yTrain, yHatTrain)
    #test_acc = accuracy_score(yTest, yHatTest)
    
    for model in models:
        if model == "DT":
            clf = DecisionTreeClassifier()
        elif model == "NB":
            clf = MultinomialNB()
        elif model == "KNN":
            clf = KNeighborsClassifier(n_neighbors = 5)
        elif model == "LR":
            clf = LogisticRegression(solver='lbfgs')
        elif model == "P":
            clf = Perceptron(tol=1e-3)
        elif model == "SV":
            clf = SVC()
        elif model == "RF":
            clf = RandomForestClassifier()
        elif model == "GB":
            clf = GradientBoostingClassifier()
        elif model == "AB":
            clf = AdaBoostClassifier()
        elif model == "BAG":
            clf = BaggingClassifier()
        else:
            print("ERROR: MODEL SPECIFIER NOT FOUND")
        
        for i in range(4):
            if i == 0:
                type_test = 'I-E'
            elif i == 1:
                type_test = 'N-S'
            elif i == 2:
                type_test = 'T-F'
            elif i == 3:
                type_test = 'J-P'
            else:
                print("ERROR: NO TYPE")
            
            i_yTrain = yTrain[:,i]
            i_yTest = yTest[:,i]
            clf.fit(xTrain, i_yTrain)
            train_score = clf.score(xTrain, i_yTrain)
            test_score = clf.score(xTest, i_yTest)
            stats = stats.append({'model':model, 'type-test':type_test, 'train-score':train_score, 'test-score':test_score}, ignore_index=True)
        
        print(model + ": Model Complete")
    
    return stats

def print_stats(stats, models):
    print(stats)
    #labels = models
    labels = ['I-E', 'N-S', 'T-F', 'J-P']
    x = np.arange(len(labels))
    print(models)
    
    for model in models:
        plt.bar(labels, stats.loc[stats['model'] == model, 'test-score'], tick_label=labels)
        plt.xlabel('Type Test')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of model: ' + model)
        plt.show()
        
   
    #width = 0.35
    
    #fig, ax = plt.subplots()
    #rects1 = ax.bar(x-.35/2, stats['train-score'], width, label='Train')
    #rects2 = ax.bar(x+.35/2, stats['test-score'], width, label='Test')
    
    #ax.set_ylabel('Scores')
    #ax.set_title('Accuracy Scores for ' + model)
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    #ax.legend()
    
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    #fig.tight_layout()
    #plt.show()
    

def main():
    data = pd.read_csv('mbti_preprocessed_3.csv') 
    X, Y, xTrain, xTest, yTrain, yTest = split_data(data)
    #models = ["DT", "NB"]
    models = ["DT", "NB", "KNN", "LR", "SV", "P", "RF", "GB", "AB", "BAG"]
    stats = run_models(models, xTrain, yTrain, xTest, yTest)
    print_stats(stats, models)
    
    #IE_stats = run_models(models, xTrain, IE_yTrain, xTest, IE_yTest)
    #NS_stats = run_models(models, xTrain, NS_yTrain, xTest, NS_yTest)
    #TF_stats = run_models(models, xTrain, TF_yTrain, xTest, TF_yTest)
    #JP_stats = run_models(models, xTrain, JP_yTrain, xTest, JP_yTest)
    #print("Combined")
    #print_stats(combined_stats, models)
    #print("I-E")
    #print_stats(IE_stats, models)
    #print("N-S")
    #print_stats(NS_stats, models)
    #print("T-F")
    #print_stats(TF_stats, models)
    #print("J-P")
    #print_stats(JP_stats, models)
    

if __name__ == "__main__":
    main()