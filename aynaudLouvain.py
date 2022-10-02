import numpy as np
import time
from random import shuffle
import collections
import copy

class AynaudLouvain():

    def initialize(self, node2community):
        community2nodes = {}
        for k, v in node2community.items():
            community2nodes[v] = community2nodes.get(v, []) + [k]

        return (node2community, community2nodes)

    def getNode2CommunitySum(self, node, communityId, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[communityId]
        return sum(list(map(lambda x: graphAdjMatrix[node][x], communityNodes)))

    def getNode2CommunityNumberOfEdges(self, node, communityId, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[communityId]
        onlyCommunityNodesWithEdgesToNode = list(filter(lambda x: graphAdjMatrix[node][x] == 1, communityNodes))
        return len(onlyCommunityNodesWithEdgesToNode)

    def getCommunityAllNodeDegreesSum(self, communityId, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[communityId]
        return sum([self.getNodeDegree(communityNode, graphAdjMatrix) for communityNode in communityNodes])

    def getNodeDegree(self, node, graphAdjMatrix):
        nodeEdges = graphAdjMatrix[node, :]
        return len(nodeEdges[nodeEdges > 1])

    def getNodeNeighs(self, node, graphAdjMatrix, allNodes):
        return list(filter(lambda x: graphAdjMatrix[node][x] == 1, allNodes))

    def computeModularityGain(self, node, communityId, graphAdjMatrix, community2nodes):
        m = self.computeM(graphAdjMatrix)

        if (m == 0):
            return 0

        l_n_neighCommunity = self.getNode2CommunityNumberOfEdges(node, communityId, graphAdjMatrix, community2nodes)
        sum_neighCommunity = self.getCommunityAllNodeDegreesSum(communityId, graphAdjMatrix, community2nodes)
        k_n = self.getNodeDegree(node, graphAdjMatrix)

        return 1/m * (l_n_neighCommunity - (sum_neighCommunity * k_n) / (2 * m))

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

        for k in range(len(graphAdjMatrix[0])):
            m += np.sum(graphAdjMatrix[k, k:len(graphAdjMatrix[0])])

        return m

    def computeModularity(self, graphAdjMatrix, community2nodes):

        m = self.computeM(graphAdjMatrix)

        if (m == 0):
            return 0

        partialSums = []

        for community in community2nodes:
            for i in community2nodes[community]:
                for j in community2nodes[community]:
                    if (i == j):
                        continue
                    partialSums.append(graphAdjMatrix[i][j] - (self.getNodeDegree(i, graphAdjMatrix) * self.getNodeDegree(j, graphAdjMatrix)) / (2 * m))

        return sum(partialSums)/(2*m)

    '''
    new2oldCommunities = contains mappings between current and prev step
    '''
    def computeNewAdjMatrix(self, community2nodes, new2oldCommunities, graphAdjMatrix):
        
        communities = list(community2nodes.keys())

        temporaryAdjMatrix = np.zeros((len(communities), len(communities)))

        for community1Id in range(len(communities)):
            for community2Id in range(len(communities)):
                community1 = communities[community1Id]
                community2 = communities[community2Id]
                temporaryAdjMatrix[community1Id][community2Id] = sum(self.interCommunitiesNodeWeights(community1, community2, graphAdjMatrix, community2nodes))

        newCommunityIterator = 0

        for community in community2nodes:
            # otherwise, replace it
            new2oldCommunities[newCommunityIterator] = community
            newCommunityIterator += 1
        
        return (temporaryAdjMatrix, new2oldCommunities)

    def interCommunitiesNodeWeights(self, community1, community2, graphAdjMatrix, community2nodes):

        interCommunitiesNodeWeights = []

        for i in community2nodes[community1]:
            for j in community2nodes[community2]:
                if (graphAdjMatrix[i][j] != 0):
                    interCommunitiesNodeWeights.append(graphAdjMatrix[i][j])

        return interCommunitiesNodeWeights

    def expandSuperNode(self, superNode, community2nodesFull):
        return community2nodesFull[superNode]

    def decompressSupergraph(self, community2nodes, community2nodesFull, new2oldCommunities):
        community2expandedNodes = {}
        node2communityOrdered = {}

        # take each super community and expand its super nodes (the super nodes are actually communities of nodes at the previous step)
        for superCommunity in community2nodes:
            oldCommunity = new2oldCommunities[superCommunity]
            expandedNodes = [self.expandSuperNode(new2oldCommunities[superNode], community2nodesFull) for superNode in community2nodes[superCommunity]]
            # flatten expanded nodes
            expandedNodes = [item for sublist in expandedNodes for item in sublist]
            community2expandedNodes[oldCommunity] = expandedNodes
            for node in community2expandedNodes[oldCommunity]:
                node2communityOrdered[node] = oldCommunity

        node2communityOrdered = collections.OrderedDict(sorted(node2communityOrdered.items()))

        return (node2communityOrdered, community2expandedNodes)

    def louvain(self, graphAdjMatrix, node2community):

        start_time = time.time()

        theta = 0.0001

        isFirstPass = True

        while True:

            if isFirstPass:
                (node2community, community2nodes) = self.initialize(node2community)
                graphAdjMatrixFull = graphAdjMatrix

                graphModularity = self.computeModularity(graphAdjMatrix, community2nodes)

            print('Started Louvain first phase')

            modularityFirstPhase = graphModularity

            while True:

                nodes = list(node2community.keys())

                shuffle(nodes)

                for node in nodes:

                    neighs = self.getNodeNeighs(node, graphAdjMatrix, nodes)

                    for neigh in neighs:

                        nodeCommunity = node2community[node]
                        neighCommunity = node2community[neigh]

                        if (neighCommunity == nodeCommunity):
                            continue

                        # try to move node
                        (node2communityTemp, community2nodesTemp) = self.moveNodeToCommunity(node, nodeCommunity, neighCommunity, \
                                                            copy.deepcopy(community2nodes), copy.deepcopy(node2community))

                        fullModularityGain = self.computeModularityGain(node, neighCommunity, graphAdjMatrix, community2nodesTemp) - \
                                             self.computeModularityGain(node, nodeCommunity, graphAdjMatrix, community2nodesTemp)

                        # if modularity gain is positive, perform move
                        if (fullModularityGain > 0):
                            (node2community, community2nodes) = (node2communityTemp, community2nodesTemp)

                newModularity = self.computeModularity(graphAdjMatrix, community2nodes)

                print(modularityFirstPhase, newModularity)

                if (newModularity - modularityFirstPhase <= theta):
                    break
                
                modularityFirstPhase = newModularity

            print('Finished Louvain first phase')

            print('Start Louvain second phase')

            # filter communities with no nodes
            community2nodes = {i: j for i, j in community2nodes.items() if j != []}

            if isFirstPass:
                community2nodesFull = community2nodes
                node2communityFull = node2community
                # cache previous step configuration in case modularity decreases instead of increasing
                new2oldCommunities = dict(zip(community2nodes.keys(), community2nodes.keys()))
            else:
                # cache previous step configuration in case modularity decreases instead of increasing
                (node2communityFull, community2nodesFull) = self.decompressSupergraph(community2nodes, community2nodesFull, new2oldCommunities)

            # if modularity of the first phase is smaller than previous modularity, break
            if (modularityFirstPhase <= graphModularity):
                break
            
            graphModularity = modularityFirstPhase

            (graphAdjMatrix, new2oldCommunities) = self.computeNewAdjMatrix(community2nodes, new2oldCommunities, graphAdjMatrix)
            nodes2communities = dict(zip(range(len(graphAdjMatrix[0])), range(len(graphAdjMatrix[0]))))
            (node2community, community2nodes) = self.initialize(nodes2communities)

            isFirstPass = False

        print("--- %s execution time in seconds ---" % (time.time() - start_time))

        return node2communityFull