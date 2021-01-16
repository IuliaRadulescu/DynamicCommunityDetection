import numpy as np
from igraph import Graph, VertexClustering
from igraph import plot
import random
from random import shuffle
from numpy import dot
from numpy.linalg import norm
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class LouvainSmooth():

    '''
    g = igraph current snapshot graph g
    gOld = igraph old snapshot graph 
    '''
    def __init__(self, g, gOld, isWeighted = False):
        self.g = g
        self.gOld = gOld

        # add weights if graph is not weighted
        if isWeighted == False:
            self.g.es['weight'] = [1] * len(self.g.es)


    def getCentroid(self, subgr):
        
        vectors = np.array([node['tfIdfVector'] for node in subgr.vs])

        if (len(vectors) == 0):
            return np.array([])

        return vectors.mean(axis = 0)


    '''
    compute the modularity gain obtained by moving node 'n' in community 'neighCommunity'
    applySmoothing = True/ False
    '''
    def computeModularityGain(self, n, neighCommunity, g, applySmoothing, gOld = None, oldClusters = None):
       
        m = np.sum(np.array(g.es['weight']))
        k_n_neighCommunity = np.sum(np.array([g[n.index, g.vs.find(j['name']).index] for j in list(neighCommunity.vs)]))
        k_n = np.sum([(g.es[edgeId]['weight']) for edgeId in g.incident(n)])

        nodesInCommunity = [g.vs.find(j['name']).index for j in list(neighCommunity.vs)]
        sum_neighCommunity = []

        for nodeInCommunity in nodesInCommunity:
            sum_neighCommunity += [(g.es[edgeId]['weight']) for edgeId in g.incident(nodeInCommunity)]

        sum_neighCommunity = np.sum(np.array(sum_neighCommunity))

        modularityGain = (1/m) * ( (k_n_neighCommunity) - (sum_neighCommunity * k_n)/(2*m) )

        if (applySmoothing and oldClusters != None and gOld != None):
            
            if (n['name'] not in list(set([node['name'] for node in self.gOld.vs]))):
                return modularityGain

            oldNodeCommunity = gOld.vs.find(n['name'])['community']

            centroidOld = self.getCentroid(oldClusters.subgraph(oldNodeCommunity))
            centroidCandidate = self.getCentroid(neighCommunity)

            if (len(centroidOld) == 0 or len(centroidCandidate) == 0):
                textualGain = 0
            else:
                textualGain = dot(centroidOld, centroidCandidate)/(norm(centroidOld)*norm(centroidCandidate))

            modularityGain += textualGain

        return modularityGain

    
    def removeLinks(self, comments):
        return list(map(lambda x: re.sub(r'(https?://[^\s]+)', '', x), comments))

    def removeRedditReferences(self, comments):
        return list(map(lambda x: re.sub(r'(/r/[^\s]+)', '', x), comments))

    def removePunctuation(self, comments):
        return list(map(lambda x: re.sub('[,.!?"\'\\n:*]', '', x), comments))

    def getFeatureVectors(self, comments):

        vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
        return vectorizer.fit_transform(comments).toarray()

    def augumentGraphsWithVectors(self):

        docs = []

        for node in list(self.g.vs) + list(self.gOld.vs):
            comments = node['comments']
            comments = self.removeLinks(comments)
            comments = self.removeRedditReferences(comments)
            comments = self.removePunctuation(comments)

            commentsConcat = ' '.join(comments)

            docs.append(commentsConcat)

        featureVectors = self.getFeatureVectors(docs)

        self.g.vs['tfIdfVector'] = featureVectors[0:len(self.g.vs)]
        self.gOld.vs['tfIdfVector'] = featureVectors[len(self.g.vs): len(self.g.vs) + len(self.gOld.vs)]

        print('Augumented graphs')

    
    def applyLouvain(self, applySmoothing = True):

        def reindexMembership(g, membershipList):
            clusterId = 0
            finalMembershipList = []
            member2ClusterId = {}

            for index in range(len(membershipList)):

                member = membershipList[index]

                if member not in member2ClusterId:
                    clusterId += 1
                    member2ClusterId[member] = clusterId
                
                finalMembershipList.append(member2ClusterId[member])

                g.vs[index]['community'] = member2ClusterId[member]

            return finalMembershipList

        def firstPhase(g, gOld, applySmoothing):

            print('Started Louvain Smooth first phase')

            theta = 0.01

            membershipList = list(range(len(g.vs)))
            g.vs['community'] = membershipList
            clusters = VertexClustering(g, membershipList)

            if (gOld != None):
                oldMembershipList = [node['community'] for node in gOld.vs]
                oldClusters = VertexClustering(gOld, oldMembershipList)
            else:
                applySmoothing = False
                oldClusters = None

            initialModularity = newModularity = g.modularity(clusters.membership)

            while True:

                nodes = [node for node in g.vs]

                shuffle(nodes)

                for node in nodes:

                    neis = g.neighbors(node)
                    
                    modularityGains = []

                    for neigh in neis:
                        neighCommunity = clusters.subgraph(g.vs[neigh]['community'])
                        nodeCommunity = clusters.subgraph(node['community'])
                        fullModularityGain = self.computeModularityGain(node, neighCommunity, g, applySmoothing, gOld, oldClusters) + self.computeModularityGain(node, nodeCommunity, g, applySmoothing, gOld, oldClusters)

                        if (fullModularityGain > 0):
                            modularityGains.append((int(neigh), fullModularityGain))

                    if (len(modularityGains) > 0):

                        # get max modularity community
                        modularityGains = np.array(modularityGains, dtype = int)
                        maxModularityGainIndex = np.argmax(modularityGains[:, 1])
                        maxModularityGainIndices = np.where(modularityGains[:,1]==modularityGains[maxModularityGainIndex][1])

                        maxModularityNeighs = [modularityGains[mIndex[0]][0] for mIndex in maxModularityGainIndices]

                        maxModularityNodeId = maxModularityNeighs[0]

                        if (len(maxModularityNeighs) > 0):
                            maxNeighDeg = g.vs[maxModularityNeighs[0]].degree()

                            for maxNeighId in maxModularityNeighs:
                                neighDeg = g.vs[maxNeighId].degree()
                                if (neighDeg > maxNeighDeg):
                                    maxNeighDeg = neighDeg
                                    maxModularityNodeId = maxNeighId

                        # do community switch
                        g.vs[node.index]['community'] = g.vs[maxModularityNodeId]['community']
                        membershipList[node.index] = g.vs[maxModularityNodeId]['community']

                        membershipList = reindexMembership(g, membershipList)
                        
                        # rebuild clusters
                        clusters = VertexClustering(g, membershipList)

                initialModularity = newModularity
                newModularity = clusters.modularity

                if ((newModularity - initialModularity) < theta):
                    break
            
            return clusters

        if (self.gOld != None):
            self.augumentGraphsWithVectors()

        clusters = firstPhase(self.g, self.gOld, applySmoothing)

        return clusters