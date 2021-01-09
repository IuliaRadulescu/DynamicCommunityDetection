import numpy as np
from igraph import Graph, VertexClustering
from igraph import plot
import random
from random import shuffle

class LouvainSmooth():

    '''
    takes igraph g as parameter
    '''
    def __init__(self, g, isWeighted = False):
        self.g = g

        # add weights if graph is not weighted
        if isWeighted == False:
            self.g.es['weight'] = [1] * len(self.g.es)


    '''
    compute the modularity gain obtained by moving node 'n' in community 'neighCommunity'
    '''
    def computeModularityGain(self, n, neighCommunity, g):
       
        m = np.sum(np.array(g.es['weight']))
        k_n_neighCommunity = np.sum(np.array([g[n.index, g.vs.find(j['name']).index] for j in list(neighCommunity.vs)]))
        k_n = np.sum([(g.es[edgeId]['weight']) for edgeId in g.incident(n)])

        nodesInCommunity = [g.vs.find(j['name']).index for j in list(neighCommunity.vs)]
        sum_neighCommunity = []

        for nodeInCommunity in nodesInCommunity:
            sum_neighCommunity += [(g.es[edgeId]['weight']) for edgeId in g.incident(nodeInCommunity)]

        sum_neighCommunity = np.sum(np.array(sum_neighCommunity))

        return (1/m) * ( (k_n_neighCommunity) - (sum_neighCommunity * k_n)/(2*m) )

    def applySimpleLouvain(self):

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

        def firstPhase(g):

            theta = 0.01

            membershipList = list(range(len(g.vs)))
            g.vs['community'] = membershipList

            clusters = VertexClustering(g, membershipList)

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
                        fullModularityGain = self.computeModularityGain(node, neighCommunity, g) + self.computeModularityGain(node, nodeCommunity, g)

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

        clusters = firstPhase(self.g)

        return clusters