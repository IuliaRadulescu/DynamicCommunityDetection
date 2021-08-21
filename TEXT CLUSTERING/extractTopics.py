import pymongo
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words

class TopicExtractor:

    def __init__(self, dataset):
        self.comments = [x['body'] for x in dataset]

    # Helper function
    def prettyPrintTopics(self, model, count_vectorizer, n_top_words, printResults = False):
        words = count_vectorizer.get_feature_names()
        if 'the' in words:
            print('THE IN WORDS')
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            wordsInTopic = ' '.join([words[i]
                                for i in topic.argsort()[:-n_top_words:-1]])
            topics.append(wordsInTopic)
            if (printResults):
                print('\nTopic #%d:' % topic_idx)
                print('Words:', wordsInTopic)
        
        return topics

    def removeLinks(self):
        self.comments = list(map(lambda x: re.sub(r'(https?://[^\s]+)', '', x), self.comments))

    def removeRedditReferences(self):
        self.comments = list(map(lambda x: re.sub(r'(/r/[^\s]+)', '', x), self.comments))

    def removePunctuation(self):
        # remove 'normal' punctuation
        self.comments = list(map(lambda x: x.strip(string.punctuation), self.comments))

        # remove special chars
        specials = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.',
           '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 
           '`', '{', '|', '}', '~', '»', '«', '“', '”']
        pattern = re.compile("[" + re.escape("".join(specials)) + "]")

        self.comments = list(map(lambda x: re.sub(pattern, '', x), self.comments))

    def doLemmatization(self):

        lemmatizer = WordNetLemmatizer()

        finalStop = list(get_stop_words('english')) # About 900 stopwords
        nltkWords = stopwords.words('english') # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop.extend(['like', 'the', 'this'])
        finalStop = list(set(finalStop))
        
        def removeStopAndlemmatizeComment(comment):

            # tokenize comment
            tokens = word_tokenize(comment)
            
            # convert all to lowercase
            tokens = [x.lower() for x in tokens]

            # remove stop words from comment
            tokens = list(filter(lambda x: (x not in finalStop) and (len(x) > 1 and x != 'i'), tokens))

            if len(tokens) == 0:
                return False

            lemmatizedTokens = [lemmatizer.lemmatize(w) for w in tokens]

            # lemmatize and return result
            return ' '.join(lemmatizedTokens)
        
        self.comments = list(map(lambda x: removeStopAndlemmatizeComment(x), self.comments))

        # filter potential empty values
        self.comments = list(filter(lambda x: x != False, self.comments))

    def prepareForLDA(self):

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.comments)

        return (vectorizer, X)

    def getTopics(self, noTopics, noWords):
        self.removeLinks()
        self.removeRedditReferences()
        self.removePunctuation()
        self.doLemmatization()

        vectorizer, X = self.prepareForLDA()

        lda = LDA(n_components = noTopics, n_jobs = -1)
        lda.fit(X) # Print the topics found by the LDA model
        
        print("Topics found via LDA:")
        topics = self.prettyPrintTopics(lda, vectorizer, noWords, True)

        return topics
        
class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

        self.dbClient = pymongo.MongoClient('localhost', 27017)

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

allComments = list(db.comments.find())

dbClient.close()

print('We have a number of', len(allComments), 'comments')

topicExtractor = TopicExtractor(allComments)
topicExtractor.getTopics(1, 10)

    
