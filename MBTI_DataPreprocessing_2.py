import argparse
import re
import string

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# data preprocessing
import nltk
import pandas as pd
from afinn import Afinn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# def dataPreprocessing(xFeat):

def dataVisualization(xFeat):
    # visualization of data
    cnt_types = xFeat['type'].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(cnt_types.index, cnt_types.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Types', fontsize=12)
    plt.show()

    text = " "

    # for i in range(xFeat.shape[0]):
    #     if xFeat['type'][i] == 'INFJ':
    #         text = text + xFeat['posts'][i]
    # wordcloud = WordCloud().generate(text)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()
    #
    # # wordCloud
    # type_quote = xFeat.groupby('type').sum()
    # e_posts = ''
    # i_posts = ''
    # for _type in type_quote.index:
    #     if 'E' in _type:
    #         e_posts += type_quote.loc[_type].posts
    #     else:
    #         i_posts += type_quote.loc[_type].posts
    # stopwords = set(STOPWORDS)
    # stopwords.add("think")
    # stopwords.add("people")
    # stopwords.add("thing")
    # my_wordcloud = WordCloud(width=800, height=800, stopwords=stopwords, background_color='white')
    #
    # # Introvert
    # my_wordcloud_i = my_wordcloud.generate(i_posts)
    # plt.subplots(figsize=(15, 15))
    # plt.imshow(my_wordcloud_i)
    # plt.axis("off")
    # plt.title('Introvert', fontsize=30)
    # plt.show()
    #
    # # Extrovert
    # my_wordcloud_e = my_wordcloud.generate(e_posts)
    # plt.subplots(figsize=(15, 15))
    # plt.imshow(my_wordcloud_e)
    # plt.axis("off")
    # plt.title('Extrovert', fontsize=30)
    # plt.show()


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

    # dataVisualization(xFeat)

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
    print(len(xFeat['countOf_videos']))

    # remove stop words stem the words
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    cachedStopWords = stopwords.words("english")
    types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ',
             'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
    # replace urls and remove all punctuation and numbers
    # sentiment analysis
    afinn = Afinn()
    sentimentScores = []
    translator = str.maketrans('', '', string.punctuation)
    cleanPost = []
    for post in xFeat['posts']:
        post = re.sub(r'https?:\/\/\S*', '', post)
        post = re.sub(r'  ', '', post)
        post = re.sub(r' [|||)(?.,:1234567890!]', ' ', post)
        post = post.translate(translator)
        post = " ".join([lemmatiser.lemmatize(w) for w in post.split(' ') if w not in cachedStopWords])
        for t in types:
            post = post.replace(t, "")
        cleanPost.append(post)
        sentiment_score = afinn.score(post)
        sentimentScores.append(sentiment_score)
    xFeat['posts'] = cleanPost
    xFeat['sentiment_score'] = sentimentScores

    # make characters all lowercase
    xFeat['posts'] = [post.lower() for post in xFeat['posts']]

    # TFIDF evaluates how important a word is to a document in a collection or corpus.
    cntizer = CountVectorizer(analyzer="word",
                              max_features=1500,
                              tokenizer=None,
                              preprocessor=None,
                              stop_words=None,
                              max_df=0.7,
                              min_df=0.1)

    # Learn the vocabulary dictionary and return term-document matrix
    print("CountVectorizer...")
    X_cnt = cntizer.fit_transform(xFeat['posts'])
    # Transform the count matrix to a normalized tf or tf-idf representation
    tfizer = TfidfTransformer()
    print("Tf-idf...")
    # Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
    X_tfidf = tfizer.fit_transform(X_cnt).toarray()
    print(X_tfidf)
    feature_names = list(enumerate(cntizer.get_feature_names()))
    cleanedXFeat = pd.DataFrame(data=X_tfidf, columns=feature_names)
    frames = [cleanedXFeat, xFeat]
    cleanedXFeat = pd.concat(frames, axis=1)
    print(cleanedXFeat)

    # split the whole type into four functions and each orientation under the function is marked by 0 or 1
    map1 = {"I": 0, "E": 1}
    map2 = {"N": 0, "S": 1}
    map3 = {"T": 0, "F": 1}
    map4 = {"J": 0, "P": 1}
    cleanedXFeat['I-E'] = cleanedXFeat['type'].astype(str).str[0]
    cleanedXFeat['I-E'] = cleanedXFeat['I-E'].map(map1)
    cleanedXFeat['N-S'] = cleanedXFeat['type'].astype(str).str[1]
    cleanedXFeat['N-S'] = cleanedXFeat['N-S'].map(map2)
    cleanedXFeat['T-F'] = cleanedXFeat['type'].astype(str).str[2]
    cleanedXFeat['T-F'] = cleanedXFeat['T-F'].map(map3)
    cleanedXFeat['J-P'] = cleanedXFeat['type'].astype(str).str[3]
    cleanedXFeat['J-P'] = cleanedXFeat['J-P'].map(map4)

    cleanedXFeat.to_csv('mbti_preprocessed.csv')




if __name__ == "__main__":
    main()
