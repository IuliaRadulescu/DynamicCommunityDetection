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

class CommunityGraphBuilder(ABC):

    def __init__(self, dataset):
        self.dataset = dataset
        self.g = Graph()

    @abstractmethod
    def getEdges(self):
        return

    @abstractmethod
    def getNodes(self):
        return

    def buildCommunityGraph(self):

        nodesList = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(edgesList)

        # remove nodes without edges
        # nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
        # self.g.delete_vertices(nodesToRemove)


class BuildAuthorsCommunityGraph(CommunityGraphBuilder):

    def buildCommunityGraph(self):

        nodesList = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(edgesList)

        self.augumentAuthorNodesWithComments()

    def getEdges(self):

        def getEdgesFromRecord(record, authors):

            if (('parentAuthorId' not in record) or 
            (record['parentAuthorId'] not in authors) or
            ('author' not in record) or
            ('author' == False)):
                return None

            parentId = record['parentAuthorId']

            return (record['author'], parentId)

        authors = list(set(map(lambda x: x['author'], self.dataset)))

        return list(set(map(lambda x: getEdgesFromRecord(x, authors), self.dataset)))

    def getNodes(self):

        # use a generic name for comments with no authors
        genericAuthorName = 'JhonDoe25122020'

        nodesList = list(map(lambda x: x['author'], self.dataset))
        nodesList = list(set(nodesList))
        nodesList = list(map(lambda x: genericAuthorName if x == False else x, nodesList))
        
        return nodesList

    def augumentNodesWithField(self, fieldName, newAttributeName, isList = False):
        
        if (isList):
            self.g.vs[newAttributeName] = [[] for _ in range(len(self.g.vs))]

        for item in self.dataset:

            author = item['author']

            fieldValue = item[fieldName] if fieldName in item else -1

            try:
                authorNode = self.g.vs.find(name = author)
            except ValueError:
                print('Node ', author, ' does not exist')
                continue               
            
            if (isList):
                self.g.vs[authorNode.index][newAttributeName].append(fieldValue)
            else:
                self.g.vs[authorNode.index][newAttributeName] = fieldValue


    def augumentAuthorNodesWithComments(self):

        self.augumentNodesWithField('body', 'comments', True)

    def augumentAuthorNodesWithCommunityId(self):

        self.augumentNodesWithField('clusterIdSmooth', 'community')

    def getGraph(self):
        return self.g

    def plotGraph(self, attributeField = 'authorClusterIdKMeans'):

        attributeDict = dict(zip([x['author'] for x in self.dataset], [x[attributeField] for x in self.dataset]))

        author2Indexes = dict(zip([node['name'] for node in self.g.vs], [node.index for node in self.g.vs]))

        for author in attributeDict:
            if author in author2Indexes:
                self.g.vs[author2Indexes[author]][attributeField] = attributeDict[author]
                
        clusters = VertexClustering(self.g, self.g.vs[attributeField])

        plot(clusters)


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
                    'clusterIdSmooth': clusterId
                }
            })

def applyLouvain(collectionName, g):

    # and also show modularity
    clusters = g.community_multilevel()
    modularity_score = g.modularity(clusters.membership)

    print('The modularity is ', modularity_score)

    updateClusters(clusters, collectionName)

def applyLouvainSmooth(collectionName, g, gOld = None):

    louvainS = louvainSmooth.LouvainSmooth(g, gOld, False)

    clusters = louvainS.applyLouvain()

    # plot(clusters)

    print('The modularity is ', clusters.modularity)

    updateClusters(clusters, collectionName)

def updateClusters(clusters, collectionName):

    subgraphs = clusters.subgraphs()

    clusterId = 0

    for subgraph in subgraphs:
        authors = []
        for vertex in subgraph.vs:
            authors.append(vertex['name'])
        MongoDBClient.getInstance().updateAuthors(authors, clusterId, collectionName)
        clusterId += 1

def getCommentsCommunity(collectionName):

    allComments = list(db[collectionName].find())

    commentsCommunity = BuildAuthorsCommunityGraph(allComments)
    commentsCommunity.buildCommunityGraph()

    return commentsCommunity

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

# allCollections = db.list_collection_names()

# prefix = 'oneHour'
# allCollections = list(filter(lambda x: prefix in x, allCollections))

# allCollections = sorted(allCollections)

allCollections = ['oneHour_05_02_05_03', 'oneHour_05_03_05_04']

for collectionIdx in range(1, len(allCollections)):

    prevCollection = allCollections[collectionIdx - 1]
    curCollection = allCollections[collectionIdx]

    print('Started processing ' + curCollection + ' prev collection is ', prevCollection)

    prevCommunity = getCommentsCommunity(prevCollection)

    curCommunity = getCommentsCommunity(curCollection)

    # print('MY LOUVAIN')
    # if (collectionIdx == 1):
    #     # if first pass, need to do Louvain on the first collection
    #     applyLouvainSmooth(prevCollection, prevCommunity.getGraph())
    #     prevCommunity = getCommentsCommunity(prevCollection)
    #     prevCommunity.augumentAuthorNodesWithCommunityId()
    #     applyLouvainSmooth(curCollection, curCommunity.getGraph(), prevCommunity.getGraph())
    # else:
    prevCommunity.augumentAuthorNodesWithCommunityId()
    applyLouvainSmooth(curCollection, curCommunity.getGraph(), prevCommunity.getGraph())

    # print('OFFICIAL LOUVAIN')
    # curCommunity.applyLouvain()

    print('Finished processing')

# filter nodes with inexisting parents
# commentsWithoutParents = [comment for comment in allComments if ('parentAuthorId' in comment) and (comment['parentAuthorId'] != False) and (comment['parentAuthorId'] not in [comment['author'] for comment in allComments])]
# print('Comments without parents ', len(commentsWithoutParents))