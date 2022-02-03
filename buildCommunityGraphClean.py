import pymongo
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractmethod

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

    def getGraph(self):
        return self.g

    def plotGraph(self, attributeField = 'clusterIdSmooth'):

        attributeDict = dict(zip([x['author'] for x in self.dataset], [x[attributeField] for x in self.dataset]))

        author2Indexes = dict(zip([node['name'] for node in self.g.vs], [node.index for node in self.g.vs]))

        for author in attributeDict:
            if author in author2Indexes:
                self.g.vs[author2Indexes[author]][attributeField] = attributeDict[author]
                
        clusters = VertexClustering(self.g, self.g.vs[attributeField])

        print('MODULARITY', clusters.modularity)

        plot(clusters)


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

    def updateAuthors(self, authors, clusterId, collectionName):

        self.dbClient = pymongo.MongoClient('localhost', 27017)

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
            
        self.dbClient.close()

def applyLouvainModule(collectionName, g):

    # and also show modularity
    clusters = g.community_multilevel()
    modularity_score = g.modularity(clusters.membership)

    # plot(clusters)

    print('The modularity is ', modularity_score)

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

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

def getCommentsCommunity(collectionName, justNodesWithEdges = False):

    allComments = list(db[collectionName].find())

    commentsCommunity = BuildAuthorsCommunityGraph(allComments)
    commentsCommunity.buildCommunityGraph(justNodesWithEdges)

    return commentsCommunity

def getAllCollections(prefix, startWithCollection = False):

    def filterCollections(c, prefix, startWithCollection):
        startWithPrefix = prefix in c

        if (startWithCollection == False):
            return startWithPrefix
        
        return startWithPrefix and (c > startWithCollection)

    allCollections = db.list_collection_names()

    prefix = 'quarter'
    allCollections = list(filter(lambda c: filterCollections(c, prefix, startWithCollection), allCollections))

    return sorted(allCollections)

def applySimpleLouvainOnAllCollections():

    allCollections = getAllCollections('quarter')

    for collectionName in allCollections:

        print('===> Running Louvain on ', collectionName)

        community = getCommentsCommunity(collectionName)

        applyLouvainModule(collectionName, community.getGraph())


def plotCollection(collectionName, attributeField):

    community = getCommentsCommunity(collectionName, False)
    community.plotGraph(attributeField)

applySimpleLouvainOnAllCollections()