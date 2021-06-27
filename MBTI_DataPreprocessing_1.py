import argparse
import math
import numpy as np
import pandas as pd
import string as string
import re
from afinn import Afinn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data",
                        default="mbti_1.csv",
                        help="filename for the mbti dataset")

    args = parser.parse_args()
    # load the data
    xFeat = pd.read_csv(args.data)

    # sentiment analysis

    # type_quote = xFeat.groupby('type').sum()
    # e_posts = ''
    # i_posts = ''
    # for _type in type_quote.index:
    #     if 'E' in _type:
    #         e_posts += type_quote.loc[_type].posts
    #     else:
    #         i_posts += type_quote.loc[_type].posts
    # print e_posts
    # stopwords = set(STOPWORDS)
    # stopwords.add("think")
    # stopwords.add("people")
    # stopwords.add("thing")
    # my_wordcloud = WordCloud(width=800, height=800, stopwords=stopwords, background_color='white')
    # # Introvert
    # my_wordcloud_i = my_wordcloud.generate(i_posts)
    # print my_wordcloud_i
    # plt.subplots(figsize=(15, 15))
    # plt.imshow(my_wordcloud_infj)
    # plt.axis("off")
    # plt.title('Introvert', fontsize=30)
    # plt.show()
    # # Extrovert
    # my_wordcloud_e = my_wordcloud.generate(e_posts)
    # plt.subplots(figsize=(15, 15))
    # plt.imshow(my_wordcloud_infj)
    # plt.axis("off")
    # plt.title('Extrovert', fontsize=30)
    # plt.show()

    # count of words divided by 50 because each object includes a total of past 50 posts separated by "|||"
    xFeat['avg_countOf_words'] = xFeat['posts'].apply(lambda x: len(x.split()) / 50)
    # count of links within each person's total posts
    xFeat['countOf_links'] = xFeat['posts'].apply(lambda x: x.count('http'))
    # count of pics
    xFeat['countOf_pics'] = xFeat['posts'].apply(
        lambda x: x.count('jpg') + x.count('png') + x.count('jpeg') + x.count('gif'))
    # count of exclamations
    xFeat['countOf_exclamations'] = xFeat['posts'].apply(lambda x: x.count('!'))
    # count of videos
    xFeat['countOf_videos'] = xFeat['posts'].apply(
        lambda x: x.count('vimeo') + x.count('youtube'))

    # replace urls and remove all punctuation and numbers
    translator = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleanPost = []
    for post in xFeat['posts']:
        post = re.sub(r'https?:\/\/\S*', '', post)
        post = re.sub(r'  ', '', post)
        post = re.sub(r'[|||)(?.,:1234567890!]', ' ', post)

        post = post.translate(translator)
        cleanPost.append(post)
    xFeat['posts'] = cleanPost

    # make characters all lowercase
    xFeat['posts'] = [post.lower() for post in xFeat['posts']]

    # remove stop words might do it after this submission also stem the words

    # sentiment analysis
    afinn = Afinn()
    sentimentScores = []
    for post in xFeat['posts']:
        sentiment_score = afinn.score(post)
        print(sentiment_score)
        sentimentScores.append(sentiment_score)
    xFeat['sentiment_score'] = sentimentScores
    print(xFeat['sentiment_score'])



    # PorterStemmer and also lemmatized using WordNet Lemmatization.
    # print(post.translate(translator))
    # table = str.maketrans('', str.punctuation)
    # stripped = [w.translate(table) for w in xFeat['posts']]
    # >>> stemmer = PorterStemmer()
    # Test the stemmer on various pluralised words.
    #
    # >>> plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
    # ...            'died', 'agreed', 'owned', 'humbled', 'sized',
    # ...            'meeting', 'stating', 'siezing', 'itemization',
    # ...            'sensational', 'traditional', 'reference', 'colonizer',
    # ...            'plotted']
    # >>> singles = [stemmer.stem(plural) for plural in plurals]
    # >>> print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACE
    # caress fli die mule deni die agre own humbl size meet
    # state siez item sensat tradit refer colon plot
    # split the whole type into four functions and each orientation under the function is marked by 0 or 1
    map1 = {"I": 0, "E": 1}
    map2 = {"N": 0, "S": 1}
    map3 = {"T": 0, "F": 1}
    map4 = {"J": 0, "P": 1}
    xFeat['I-E'] = xFeat['type'].astype(str).str[0]
    xFeat['I-E'] = xFeat['I-E'].map(map1)
    xFeat['N-S'] = xFeat['type'].astype(str).str[1]
    xFeat['N-S'] = xFeat['N-S'].map(map2)
    xFeat['T-F'] = xFeat['type'].astype(str).str[2]
    xFeat['T-F'] = xFeat['T-F'].map(map3)
    xFeat['J-P'] = xFeat['type'].astype(str).str[3]
    xFeat['J-P'] = xFeat['J-P'].map(map4)

    
    xFeat.to_csv('mbti_preprocessed.csv')

    # Logistic Regression
    X = xFeat.drop(['type', 'posts', 'I-E', 'N-S', 'T-F', 'J-P'], axis=1).values
    print(X)
    y = xFeat['type'].values
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=5)
    lr = LogisticRegression().fit(xTrain, yTrain)

    y_pred = lr.predict(xTest)
    print(accuracy_score(yTest, y_pred))


if __name__ == "__main__":
    main()
