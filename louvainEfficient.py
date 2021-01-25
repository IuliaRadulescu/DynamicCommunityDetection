import numpy as np
import time
from random import shuffle

class LouvainEfficient():

    def __init__(self, graphAdjMatrix):
        self.graphAdjMatrix = graphAdjMatrix
        self.node2community = {}
        self.community2nodes = {}

    def initialize(self):
        noNodes = np.shape(self.graphAdjMatrix)[0]
        self.node2community = dict(zip(range(noNodes), range(noNodes)))
        self.community2nodes = dict(zip(range(noNodes), [[node] for node in range(noNodes)]))
        self.m = self.getAllWeigthsSum()

    def getAllWeigthsSum(self):
        return np.sum(self.graphAdjMatrix)

    def getNode2CommunitySum(self, node, communityId):
        communityNodes = self.community2nodes[communityId]
        return sum(list(map(lambda x: self.graphAdjMatrix[node][x], communityNodes)))

    def getCommunityAllNodesSum(self, communityId):
        communityNodes = self.community2nodes[communityId]
        allNodesSum = 0
        for communityNode in communityNodes:
            allNodesSum += np.sum(self.graphAdjMatrix[communityNode, :])
        return allNodesSum

    def getNodeSum(self, node):
        return np.sum(self.graphAdjMatrix[node, :])

    def getNodeNeighs(self, node):
        allNodes = list(range(np.shape(self.graphAdjMatrix)[0]))
        return list(filter(lambda x: self.graphAdjMatrix[node][x] == 1, allNodes))

    
    def computeModularityGain(self, node, communityId):
        k_n_neighCommunity = self.getNode2CommunitySum(node, communityId)
        sum_neighCommunity = self.getCommunityAllNodesSum(communityId)
        k_n = self.getNodeSum(node)

        return (1/self.m) * ( (k_n_neighCommunity) - (sum_neighCommunity * k_n)/(2*self.m) )


    def moveNodeToCommunity(self, node, oldCommunity, newCommunity):
        self.node2community[node] = newCommunity
        self.community2nodes[oldCommunity].remove(node)
        self.community2nodes[newCommunity].append(node)

    def computeModularity(self):

        partialSums = []

        for community in self.community2nodes:
            for i in self.community2nodes[community]:
                for j in self.community2nodes[community]:
                    if (i == j):
                        continue
                    partialSums.append(self.graphAdjMatrix[i][j] - (self.getNodeSum(i) * self.getNodeSum(j))/(2*self.m))

        return sum(partialSums)/(2*self.m)

    def computeNewAdjMatrix(self):
        communities = list(filter(lambda x: len(self.community2nodes[x]) > 0, self.community2nodes.keys()))

        temporaryAdjMatrix = np.zeros((len(communities), len(communities)))

        for community1Id in range(len(communities)):
            for community2Id in range(len(communities)):
                community1 = communities[community1Id]
                community2 = communities[community2Id]
                temporaryAdjMatrix[community1Id][community2Id] = sum(self.interCommunitiesNodeWeights(community1, community2))

        self.graphAdjMatrix = temporaryAdjMatrix

    def interCommunitiesNodeWeights(self, community1, community2):

        if (community1 == community2):
            return []

        interCommunitiesNodeWeights = []

        for i in self.community2nodes[community1]:
            for j in self.community2nodes[community2]:
                if (self.graphAdjMatrix[i][j] != 0):
                    interCommunitiesNodeWeights.append(self.graphAdjMatrix[i][j])

        return interCommunitiesNodeWeights

    def louvain(self):

        start_time = time.time()

        finalNode2Community = {}

        theta = 0.0001

        while True:

            self.initialize()

            noNodes = np.shape(self.graphAdjMatrix)[0]
            nodes = list(range(noNodes))

            initialModularity = self.computeModularity()
                
            print('Started Louvain first phase')

            while True:

                for node in nodes:
                    shuffle(nodes)

                    neis = self.getNodeNeighs(node)

                    modularityGains = []
                    for neigh in neis:
                        
                        neighCommunity = self.node2community[neigh]
                        nodeCommunity = self.node2community[node]

                        if (neighCommunity == nodeCommunity):
                            continue

                        fullModularityGain = self.computeModularityGain(node, neighCommunity) + self.computeModularityGain(node, nodeCommunity)

                        if (fullModularityGain > 0):
                            modularityGains.append((neighCommunity, fullModularityGain))

                    if (len(modularityGains) > 0):
                        modularityGains = np.array(modularityGains)
                        # sort modularity gains by column (the second column of the modularityGains represents the modularity gain - the first being the community)
                        modularityGainsIndicesSorted = np.argsort(modularityGains, axis=0)
                        # get the position of the index corresponding to the greatest modularity gain (argsort direction is asc)
                        last = len(modularityGainsIndicesSorted) - 1
                        # get new community associated value
                        # the modularityGains entry corresponds to the last index
                        # only pick the community (position 0)
                        newCommunity = int(modularityGains[modularityGainsIndicesSorted[last][1]][0])

                        # perform move
                        self.moveNodeToCommunity(node, nodeCommunity, newCommunity)

                newModularity = self.computeModularity()

                if (newModularity - initialModularity <= theta):
                    break
                
                initialModularity = newModularity

            print('Finished Louvain first phase, modularity is', newModularity)

            print('Start Louvain second phase')

            print("--- %s execution time in seconds ---" % (time.time() - start_time))

            self.computeNewAdjMatrix()

            print('Second phase', (newModularity - initialModularity))

        finalNode2Community = self.node2community

        return finalNode2Community

                
