import pymongo
from datetime import datetime
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class Clusterer:

    def __init__(self, dataset, collectionName):
        self.dataset = dataset
        self.comments = [x['body'] for x in dataset]
        self.redditIds = [x['redditId'] for x in dataset]
        self.collectionName = collectionName

    def removeLinks(self):
        self.comments = list(map(lambda x: re.sub(r'(https?://[^\s]+)', '', x), self.comments))

    def removeRedditReferences(self):
        self.comments = list(map(lambda x: re.sub(r'(/r/[^\s]+)', '', x), self.comments))

    def removePunctuation(self):
        self.comments = list(map(lambda x: re.sub('[,.!?"\'\\n:*]', '', x), self.comments))

    def doStemming(self):
        stemmer = PorterStemmer()
        self.comments = [stemmer.stem(comment) for comment in self.comments]

    def prepareForClustering(self):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.comments)

        return (vectorizer, X)

    def doClustering(self):
        self.removeLinks()
        self.removeRedditReferences()
        self.removePunctuation()
        self.doStemming()
        _, X = self.prepareForClustering()

        allCommentsLen = len(self.comments)

        # if just one comment, no need to perform clustering
        if (allCommentsLen == 1):
            return [[0], X[0].toarray()]

        maxSilhouette = -1
        maxNoClusters = min(3, (allCommentsLen - 1))

        for noClusters in range(min(3, (allCommentsLen - 1)), min(20, allCommentsLen)):
            
            km = KMeans(n_clusters=noClusters, init='k-means++', max_iter=600, n_init=1)
            km.fit(X)

            if (len(list(set(km.labels_))) <= 1):
                continue

            sscore = silhouette_score(X, km.labels_)

            if (sscore > maxSilhouette):
                maxSilhouette = sscore
                maxNoClusters = noClusters

            # print('Silhouette for', noClusters, 'is', sscore)

        # print('Best noClusters is', maxNoClusters)

        km = KMeans(n_clusters=maxNoClusters, init='k-means++', max_iter=600, n_init=1)
        km.fit(X)

        return (km.labels_, km.cluster_centers_)

    def updateClusters(self, labels, centroids):

        clusters2RedditIds = {}

        for counter in range(0, len(self.redditIds)):
            
            label = labels[counter]
            centroid = centroids[label]
            redditId = self.redditIds[counter]
            
            if label not in clusters2RedditIds:
                clusters2RedditIds[label] = []

            clusters2RedditIds[label].append(redditId)

        for clusterId in clusters2RedditIds:
            MongoDBClient.getInstance().updateComments(clusters2RedditIds[clusterId], int(clusterId), centroid.tolist(), self.collectionName)


class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance

    def updateComments(self, redditIds, clusterId, centroid, collectionName):

        self.dbClient = pymongo.MongoClient('localhost', 27017)

        db = self.dbClient.communityDetectionUSABidenInauguration

        db[collectionName].update_many(
            {
            'redditId': {
                '$in': redditIds
                }
            },{
                '$set': {
                    'clusterIdKMeans': clusterId,
                    'centroid': centroid
                }
            })

        self.dbClient.close()


def getAllCollections(prefix='quarter'):

    allCollections = db.list_collection_names()

    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

allCollections = getAllCollections()

for collectionName in allCollections:

    allComments = list(db[collectionName].find())

    print('Clustering collection', collectionName, 'with', len(allComments), 'comments')

    kMeansClusterer = Clusterer(allComments, collectionName)
    (labels, centroids) = kMeansClusterer.doClustering()
    kMeansClusterer.updateClusters(labels, centroids)

    print('Clustering collection', collectionName, 'END ==')

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]