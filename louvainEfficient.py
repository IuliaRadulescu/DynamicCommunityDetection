import numpy as np
import time
from random import shuffle
import collections

class LouvainEfficient():

    def initialize(self, graphAdjMatrix):
        noNodes = np.shape(graphAdjMatrix)[0]
        node2community = dict(zip(range(noNodes), range(noNodes)))
        community2nodes = dict(zip(range(noNodes), [[node] for node in range(noNodes)]))

        return (node2community, community2nodes)

    def getNode2CommunitySum(self, node, communityId, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[communityId]
        return sum(list(map(lambda x: graphAdjMatrix[node][x], communityNodes)))

    def getCommunityAllNodesSum(self, communityId, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[communityId]
        allNodesSum = 0
        for communityNode in communityNodes:
            allNodesSum += np.sum(graphAdjMatrix[communityNode, :])
        return allNodesSum

    def getNodeSum(self, node, graphAdjMatrix):
        return np.sum(graphAdjMatrix[node, :])

    def getNodeNeighs(self, node, graphAdjMatrix):
        allNodes = list(range(np.shape(graphAdjMatrix)[0]))
        return list(filter(lambda x: graphAdjMatrix[node][x] == 1, allNodes))

    
    def computeModularityGain(self, node, communityId, graphAdjMatrix, community2nodes):
        m = self.computeM(graphAdjMatrix)

        k_n_neighCommunity = self.getNode2CommunitySum(node, communityId, graphAdjMatrix, community2nodes)
        sum_neighCommunity = self.getCommunityAllNodesSum(communityId, graphAdjMatrix, community2nodes)
        k_n = self.getNodeSum(node, graphAdjMatrix)

        return (1/m) * ( (k_n_neighCommunity) - (sum_neighCommunity * k_n)/(2*m) )


    def moveNodeToCommunity(self, node, oldCommunity, newCommunity, community2nodes, node2community):
        node2community[node] = newCommunity
        community2nodes[oldCommunity].remove(node)
        community2nodes[newCommunity].append(node)
        return (node2community, community2nodes)

    '''
    Graph is undirected, get only upper/lower side
    '''
    def computeM(self, graphAdjMatrix):
        m = 0

        for k in range(len(graphAdjMatrix)):
            m += np.sum(graphAdjMatrix[k, 0:k])

        return m

    def computeModularity(self, graphAdjMatrix, community2nodes):

        m = self.computeM(graphAdjMatrix)

        partialSums = []

        for community in community2nodes:
            for i in community2nodes[community]:
                for j in community2nodes[community]:
                    if (i == j):
                        continue
                    partialSums.append(graphAdjMatrix[i][j] - (self.getNodeSum(i, graphAdjMatrix) * self.getNodeSum(j, graphAdjMatrix))/(2*m))

        return sum(partialSums)/(2*m)

    '''
    new2oldCommunities = contains mappings between current and prev step
    '''
    def computeNewAdjMatrix(self, community2nodes, new2oldCommunities, graphAdjMatrix):
        
        communities = list(filter(lambda x: len(community2nodes[x]) > 0, community2nodes.keys()))

        temporaryAdjMatrix = np.zeros((len(communities), len(communities)))

        for community1Id in range(len(communities)):
            for community2Id in range(len(communities)):
                community1 = communities[community1Id]
                community2 = communities[community2Id]
                temporaryAdjMatrix[community1Id][community2Id] = sum(self.interCommunitiesNodeWeights(community1, community2, graphAdjMatrix, community2nodes))

        
        newCommunityIterator = 0

        for community in community2nodes:
            # if community is empty, leave it alone
            if (len(community2nodes[community]) == 0):
                continue
            # otherwise, replace it
            new2oldCommunities[newCommunityIterator] = community
            newCommunityIterator += 1
        
        return (temporaryAdjMatrix, new2oldCommunities)

    def interCommunitiesNodeWeights(self, community1, community2, graphAdjMatrix, community2nodes):
        if (community1 == community2):
            return []

        interCommunitiesNodeWeights = []

        for i in community2nodes[community1]:
            for j in community2nodes[community2]:
                if (graphAdjMatrix[i][j] != 0):
                    interCommunitiesNodeWeights.append(graphAdjMatrix[i][j])

        return interCommunitiesNodeWeights


    def decompressSupergraph(self, community2nodes, community2nodesFull, new2oldCommunities):

        for superCommunity in community2nodes:
            if (len(community2nodes[superCommunity]) < 2):
                continue
            # merge inner communities of the superCommunities
            finalCommunity = community2nodes[superCommunity][0]
            for community in community2nodes[superCommunity]:
                if (community != finalCommunity):
                    community2nodesFull[new2oldCommunities[finalCommunity]] += community2nodesFull[new2oldCommunities[community]]
                    community2nodesFull[new2oldCommunities[community]] = []

        node2communityFull = {}

        for community in community2nodesFull:
            if (len(community2nodesFull[community]) == 0):
                continue
            for node in community2nodesFull[community]:
                node2communityFull[node] = community

        community2nodesTemp = {}

        for community in community2nodesFull:
            if len(community2nodesFull[community]) > 0:
                community2nodesTemp[community] = community2nodesFull[community]

        node2communityOrederedTemp = collections.OrderedDict(sorted(node2communityFull.items()))
        node2communityOredered = {}
        for k, v in node2communityOrederedTemp.items():
            node2communityOredered[k] = v

        return (node2communityOredered, community2nodesFull)

    def louvain(self, graphAdjMatrix):

        start_time = time.time()

        theta = 0.0001

        isFirstPass = True

        while True:

            if isFirstPass:
                (node2community, community2nodes) = self.initialize(graphAdjMatrix)
                graphAdjMatrixFull = graphAdjMatrix
                initialModularityFull = self.computeModularity(graphAdjMatrix, community2nodes)
                
            print('Started Louvain first phase')

            while True:

                initialModularity = self.computeModularity(graphAdjMatrix, community2nodes)

                noNodes = np.shape(graphAdjMatrix)[0]
                nodes = list(range(noNodes))

                for node in nodes:
                    shuffle(nodes)

                    neis = self.getNodeNeighs(node, graphAdjMatrix)

                    modularityGains = []

                    for neigh in neis:
                        
                        neighCommunity = node2community[neigh]
                        nodeCommunity = node2community[node]

                        if (neighCommunity == nodeCommunity):
                            continue

                        fullModularityGain = self.computeModularityGain(node, neighCommunity, graphAdjMatrix, community2nodes) - \
                            self.computeModularityGain(node, nodeCommunity, graphAdjMatrix, community2nodes)

                        if (fullModularityGain > 0):
                            modularityGains.append((int(neighCommunity), fullModularityGain))

                    if (len(modularityGains) > 0):
                        
                        # get max modularity community
                        modularityGains = np.array(modularityGains, dtype = int)
                        maxModularityGainIndex = np.argmax(modularityGains[:, 1])
                        maxModularityGainIndices = np.where(modularityGains[:,1]==modularityGains[maxModularityGainIndex][1])

                        maxModularityNeighs = [modularityGains[mIndex[0]][0] for mIndex in maxModularityGainIndices]

                        maxModularityNodeId = maxModularityNeighs[0]

                        if (len(maxModularityNeighs) > 0):
                            
                            maxNeighDeg = self.getNodeSum(maxModularityNeighs[0], graphAdjMatrix)

                            for maxNeighId in maxModularityNeighs:
                                neighDeg = self.getNodeSum(maxNeighId, graphAdjMatrix)
                                if (neighDeg > maxNeighDeg):
                                    maxNeighDeg = neighDeg
                                    maxModularityNodeId = maxNeighId

                        newCommunity = node2community[maxModularityNodeId]

                        # perform move
                        (node2community, community2nodes) = self.moveNodeToCommunity(node, nodeCommunity, newCommunity, community2nodes, node2community)

                newModularity = self.computeModularity(graphAdjMatrix, community2nodes)

                if (newModularity - initialModularity <= theta):
                    break
                
                initialModularity = newModularity

            print('Finished Louvain first phase')

            print('Start Louvain second phase')

            if isFirstPass:
                community2nodesFull = community2nodes
                node2communityFull = node2community
                new2oldCommunities = dict(zip(community2nodes.keys(), community2nodes.keys()))
            else:
                (node2communityFull, community2nodesFull) = self.decompressSupergraph(community2nodes, community2nodesFull, new2oldCommunities)               
            
            newModularityFull = self.computeModularity(graphAdjMatrixFull, community2nodesFull)

            print('Second phase modularity', newModularityFull)

            if (newModularityFull - initialModularityFull <= theta):
                break
            
            initialModularityFull = newModularityFull

            (graphAdjMatrix, new2oldCommunities) = self.computeNewAdjMatrix(community2nodes, new2oldCommunities, graphAdjMatrix)
            (node2community, community2nodes) = self.initialize(graphAdjMatrix)

            isFirstPass = False

        print("--- %s execution time in seconds ---" % (time.time() - start_time))

        return node2communityFull

                
