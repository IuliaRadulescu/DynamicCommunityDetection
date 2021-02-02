import igraph
import pymongo
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractclassmethod, abstractmethod
import re
import louvain
import louvainEfficient
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def buildCommunityGraph(self, justNodesWithEdges = False):

        nodesList = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(edgesList)

        # remove nodes without edges
        if (justNodesWithEdges):
            nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
            self.g.delete_vertices(nodesToRemove)


class BuildAuthorsCommunityGraph(CommunityGraphBuilder):

    def __init__(self, dataset):
        super(BuildAuthorsCommunityGraph, self).__init__(dataset)
        
        # use a generic name for comments with no authors
        self.genericAuthorName = 'JhonDoe25122020'

    def buildCommunityGraph(self, justNodesWithEdges = False):

        nodesList = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))
        
        finalEdgesList = []

        for edge in edgesList:
            if ( ((edge[0], edge[1]) in finalEdgesList) or ((edge[1], edge[0]) in finalEdgesList) ):
                continue
            finalEdgesList.append(edge)

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(finalEdgesList)

        if (justNodesWithEdges):
            nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
            self.g.delete_vertices(nodesToRemove)

        self.addTfIdfAsAttributes()

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

        nodesList = list(map(lambda x: x['author'], self.dataset))
        nodesList = list(set(nodesList))
        nodesList = list(map(lambda x: self.genericAuthorName if x == False else x, nodesList))
        
        return nodesList

    def removeLinks(self, comments):
        return list(map(lambda x: re.sub(r'(https?://[^\s]+)', '', x), comments))

    def removeRedditReferences(self, comments):
        return list(map(lambda x: re.sub(r'(/r/[^\s]+)', '', x), comments))

    def removePunctuation(self, comments):
        return list(map(lambda x: re.sub('[,.!?"\'\\n:*]', '', x), comments))

    def getFeatureVectors(self, comments):
        vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
        return vectorizer.fit_transform(comments).toarray()

    def addTfIdfAsAttributes(self):

        authorNames = [node['name'] for node in self.g.vs]
    
        authors2comments = {}

        for elem in self.dataset:
            if elem['author'] == False:
                elem['author'] = self.genericAuthorName
            if elem['author'] not in authorNames:
                continue
            if (elem['author'] not in authors2comments):
                authors2comments[elem['author']] = str(elem['body'])
                continue
            authors2comments[elem['author']] += ' ' + str(elem['body'])

        allComments = [elem[1] for elem in authors2comments.items()]

        allComments = self.removeLinks(allComments)
        allComments = self.removeRedditReferences(allComments)
        allComments = self.removePunctuation(allComments)

        featureVectors = self.getFeatureVectors(allComments)
        featureVectorIterator = 0

        for author in authors2comments.keys():
            nodeId = self.g.vs.find(name = author).index
            self.g.vs[nodeId]['tfIdf'] = featureVectors[featureVectorIterator]
            featureVectorIterator += 1

    def getGraph(self):
        return self.g

    def plotGraph(self, attributeField = 'clusterIdSmooth'):

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

        db = self.dbClient.communityDetectionUSABidenInauguration

        db[collectionName].update_many(
            {
            'author': {
                '$in': authors
                }
            },{
                '$set': {
                    'clusterIdSimple': clusterId
                }
            })

def applyLouvainModule(collectionName, g):

    # and also show modularity
    clusters = g.community_multilevel()
    modularity_score = g.modularity(clusters.membership)

    plot(clusters)

    print('The modularity is ', modularity_score)

    # updateClusters(clusters, collectionName)

def applyLouvain(collectionName, g, louvainEfficientInstance):

    graphAdjMatrix = g.get_adjacency()
    graphAdjMatrixNp = np.array(graphAdjMatrix.data)

    nodes2Communities = louvainEfficientInstance.louvain(graphAdjMatrixNp)

    clusters = VertexClustering(g, nodes2Communities.values())

    print('===> Final modularity ', clusters.modularity)

    # updateClusters(clusters, collectionName)

def updateClusters(clusters, collectionName):

    subgraphs = clusters.subgraphs()

    clusterId = 0

    for subgraph in subgraphs:
        authors = []
        for vertex in subgraph.vs:
            authors.append(vertex['name'])
        MongoDBClient.getInstance().updateAuthors(authors, clusterId, collectionName)
        clusterId += 1

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

def getCommentsCommunity(collectionName, justNodesWithEdges = False):

    allComments = list(db[collectionName].find())

    commentsCommunity = BuildAuthorsCommunityGraph(allComments)
    commentsCommunity.buildCommunityGraph(justNodesWithEdges)

    return commentsCommunity

def getAllCollections(prefix):

    allCollections = db.list_collection_names()

    prefix = 'quarter'
    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

def applySimpleLouvainOnAllCollections():

    allCollections = getAllCollections('quarter')
    louvainEfficientInstance = louvainEfficient.LouvainEfficient()

    for collectionName in allCollections:

        print('===> Running Louvain on ', collectionName)

        community = getCommentsCommunity(collectionName)
        applyLouvain(collectionName, community.getGraph(), louvainEfficientInstance)


def plotCollection(collectionName, attributeField):

    community = getCommentsCommunity(collectionName, True)
    community.plotGraph(attributeField)

def getCommonNodes(collectionName1, collectionName2):

    community1 = getCommentsCommunity(collectionName1, True)
    community2 = getCommentsCommunity(collectionName2, True)

    nodes1 = set([node['name'] for node in community1.getGraph().vs])
    nodes2 = set([node['name'] for node in community2.getGraph().vs])

    return nodes1.intersection(nodes2)

# applySimpleLouvainOnAllCollections()

# commonNodes = getCommonNodes('quarter_05_02_30_05_02_45', 'quarter_05_02_45_05_03_00')

# print('COMMON NODES = ', commonNodes)

# print(len(commonNodes))

community = getCommentsCommunity('quarter_20_20_15_20_20_30', True)

graphAdjMatrix = community.getGraph().get_adjacency()
graphAdjMatrixNp = np.array(graphAdjMatrix.data)

nodeId2TfIdf = dict(zip([node.index for node in community.getGraph().vs], [node['tfIdf'] for node in community.getGraph().vs]))

# print(nodeId2TfIdf)

# print(list(community.getGraph().vs))

louvainEfficientInstance = louvainEfficient.LouvainEfficient()
nodes2Communities = louvainEfficientInstance.louvain(graphAdjMatrixNp, nodeId2TfIdf)

# plotCollection('quarter_05_02_30_05_02_45', 'clusterIdSmooth')
# plotCollection('quarter_20_20_15_20_20_30', 'clusterIdSimple')

# plotCollection('quarter_05_02_45_05_03_00', 'clusterIdSmooth')
# plotCollection('quarter_20_20_30_20_20_45', 'clusterIdSimple')

# plotCollection('quarter_23_01_15_23_01_30', 'clusterIdSimple')
