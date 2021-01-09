import igraph
import pymongo
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractclassmethod, abstractmethod
import re
import louvain
import louvainSmooth
from sklearn.cluster import KMeans
import numpy as np

class BuildCommunityGraph(ABC):

    def __init__(self, collectionName):
        self.collectionName = collectionName
        self.g = Graph()

    @abstractmethod
    def getEdges(self, dataset):
        return

    @abstractmethod
    def getNodes(self, dataset):
        return

    @abstractmethod
    def updateClusters(self, clusters):
        return

    def buildCommunityGraph(self, dataset):

        nodesList = self.getNodes(dataset)

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges(dataset)
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(edgesList)

        # remove nodes without edges
        # nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
        # self.g.delete_vertices(nodesToRemove)

    def applyLouvain(self):

        part = louvain.find_partition(self.g, louvain.ModularityVertexPartition)
        # plot(part)

        print('The Louvain modularity is ', part.modularity)

        # and also show modularity
        clusters = self.g.community_multilevel()
        modularity_score = self.g.modularity(clusters.membership)

        print('The modularity is ', modularity_score)

        # self.updateClusters(clusters)

    def applyLouvainSmooth(self):

        louvainS = louvainSmooth.LouvainSmooth(self.g)

        clusters = louvainS.applySimpleLouvain()

        print('The modularity is ', clusters.modularity)

        self.updateClusters(clusters)

class BuildCommentsCommunityGraph(BuildCommunityGraph):

    def getEdges(self, dataset):

        def getEdgesFromRecord(record):

            if ('parentId' not in record):
                return None

            parentId = record['parentId']
            parentIdSplit = parentId.split('_')

            return (record['redditId'], parentIdSplit[1])

        return list(map(getEdgesFromRecord, dataset))

    def getNodes(self, dataset):

        nodesList = list(map(lambda x: x['redditId'], dataset))
        return list(set(nodesList))

    def updateClusters(self, clusters):
        # TO DO
        i = 1

class BuildAuthorsCommunityGraph(BuildCommunityGraph):

    def getEdges(self, dataset):

        def getEdgesFromRecord(record, authors):

            if (('parentAuthorId' not in record) or 
            (record['parentAuthorId'] not in authors) or
            ('author' not in record) or
            ('author' == False)):
                return None

            parentId = record['parentAuthorId']

            return (record['author'], parentId)

        authors = list(set(map(lambda x: x['author'], dataset)))

        return list(set(map(lambda x: getEdgesFromRecord(x, authors), dataset)))

    def getNodes(self, dataset):

        # use a generic name for comments with no authors
        genericAuthorName = 'JhonDoe25122020'

        nodesList = list(map(lambda x: x['author'], dataset))
        nodesList = list(set(nodesList))
        nodesList = list(map(lambda x: genericAuthorName if x == False else x, nodesList))
        
        return nodesList

    def updateClusters(self, clusters):

        subgraphs = clusters.subgraphs()

        clusterId = 0

        for subgraph in subgraphs:
            authors = []
            for vertex in subgraph.vs:
                authors.append(vertex['name'])
            MongoDBClient.getInstance().updateAuthors(authors, clusterId, self.collectionName)
            clusterId += 1

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

    def updateAuthors(self, authors, clusterId, collectionName):

        db = self.dbClient.communityDetectionUSAElections

        db[collectionName].update_many(
            {
            'author': {
                '$in': authors
                }
            },{
                '$set': {
                    'authorClusterIdMyLouvain': clusterId
                }
            })

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

allCollections = db.list_collection_names()

prefix = 'oneHour'
allCollections = list(filter(lambda x: prefix in x, allCollections))

# allCollections = ['oneHour_7_20_7_21']

for collectionName in allCollections:

    print('Started processing ' + collectionName)

    allComments = list(db[collectionName].find())

    commentsCommunity = BuildAuthorsCommunityGraph(collectionName)
    commentsCommunity.buildCommunityGraph(allComments)

    print('MY LOUVAIN')
    commentsCommunity.applyLouvainSmooth()

    # print('OFFICIAL LOUVAIN')
    # commentsCommunity.applyLouvain()

    print('Finished processing')

# filter nodes with inexisting parents
# commentsWithoutParents = [comment for comment in allComments if ('parentAuthorId' in comment) and (comment['parentAuthorId'] != False) and (comment['parentAuthorId'] not in [comment['author'] for comment in allComments])]
# print('Comments without parents ', len(commentsWithoutParents))