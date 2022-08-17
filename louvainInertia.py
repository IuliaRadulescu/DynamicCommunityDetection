import numpy as np
import time
from random import shuffle
import collections

'''
NOTE: This is a personal implmentation of the I-Louvain algorithm by Combe et al.

All credit goes to:

Combe, David, et al. "I-louvain: An attributed graph clustering method." International Symposium on Intelligent Data Analysis. Springer, Cham, 2015.
'''

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

        if (m == 0):
            return 0

        k_n_neighCommunity = self.getNode2CommunitySum(node, communityId, graphAdjMatrix, community2nodes)
        sum_neighCommunity = self.getCommunityAllNodesSum(communityId, graphAdjMatrix, community2nodes)
        k_n = self.getNodeSum(node, graphAdjMatrix)

        return (1/m) * ( (k_n_neighCommunity) - (sum_neighCommunity * k_n)/(2*m) )

    def computeModularityGainInertia(self, node, neighCommunity, nodeCommunity, distanceMatrix, node2Inertia, community2nodes, fullInertia):
        
        a = 0

        communityA = community2nodes[neighCommunity] + [node]

        for neig in communityA:
            if (node < neig):
                distance = distanceMatrix[node][neig]
            else:
                distance = distanceMatrix[neig][node]

            a += (node2Inertia[node] * node2Inertia[neig]) / (2*len(distanceMatrix)*fullInertia) - distance

        b = 0

        communityB = list(set(community2nodes[neighCommunity]) - set([node]))

        for peer in communityB:
            b += (node2Inertia[node] * node2Inertia[peer]) / (2*len(distanceMatrix)*fullInertia)

        distanceb = 0

        for peer1 in communityB:
            for peer2 in communityB:
                if (peer1 == node or peer2 == node or peer2 < peer1):
                    continue
                distanceb += distanceMatrix[peer1][peer2]
        
        b -= distanceb

        return (1/(len(distanceMatrix)*fullInertia))*(a-b)

    def moveNodeToCommunity(self, node, oldCommunity, newCommunity, community2nodes, node2community):
        node2community[node] = newCommunity
        community2nodes[oldCommunity].remove(node)
        community2nodes[newCommunity].append(node)
        return (node2community, community2nodes)

    def getCentroid(self, vectors):

        vectors = np.array(vectors)

        if (len(vectors) == 0):
            return np.array([])

        return np.mean(vectors, axis = 0)

    def getInertia(self, vectors, g = None):

        # if reference (center of gravity) not provided, use centroid
        if (g is None):
            g = self.getCentroid(vectors)

        allDistances = []

        for vector in vectors:
            distance = pow(np.linalg.norm(vector - g), 2)
            allDistances.append(distance)
                
        return np.sum(np.array(allDistances))
        

    def computeModularityInertiaUtils(self, nodeId2Doc2Vec, graphAdjMatrix):

        modularityMatrix = np.zeros((len(nodeId2Doc2Vec.keys()), len(nodeId2Doc2Vec.keys())))
        distanceMatrix = np.zeros((len(nodeId2Doc2Vec.keys()), len(nodeId2Doc2Vec.keys())))

        vectors = list(nodeId2Doc2Vec.values())

        fullInertia = self.getInertia(vectors)

        if fullInertia == float(0):
            return 0

        # cache inertias
        node2inertia = {}

        for nodeId in range(len(graphAdjMatrix)):
            node2inertia[nodeId] = self.getInertia(vectors, nodeId2Doc2Vec[nodeId])

        for i in range(len(graphAdjMatrix)):
            for j in range(len(graphAdjMatrix)):
                if (j <= i or graphAdjMatrix[i][j] == 0):
                    continue

                ijDistance = pow(np.linalg.norm(nodeId2Doc2Vec[i] - nodeId2Doc2Vec[j]), 2)
                distanceMatrix[i][j] = ijDistance
                
                a = (node2inertia[i] * node2inertia[j]) / pow(2*len(nodeId2Doc2Vec.keys())*fullInertia, 2)
                b = ijDistance / (2*len(nodeId2Doc2Vec.keys())*fullInertia)

                modularityMatrix[i][j] = a - b
                     
        return (modularityMatrix, distanceMatrix, node2inertia, fullInertia)

    def computeModularityInertia(self, modularityMatrixInertia, community2nodes):
        
        inertiaSum = 0

        for community in community2nodes:
            if (len(community2nodes[community]) < 2):
                continue
            for i in range(len(community2nodes[community])):
                for j in range(len(community2nodes[community])):
                    if (j <= i):
                        continue
                    nodei = community2nodes[community][i]
                    nodej = community2nodes[community][j]
                    inertiaSum += modularityMatrixInertia[nodei][nodej]

        # print('INERTIA MODULARITY', inertiaSum)

        return inertiaSum
                    

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

        if (m == 0):
            return 0

        partialSums = []

        for community in community2nodes:
            for i in community2nodes[community]:
                for j in community2nodes[community]:
                    if (i == j or graphAdjMatrix[i][j] == 0):
                        continue
                    partialSums.append(graphAdjMatrix[i][j] - (self.getNodeSum(i, graphAdjMatrix) * self.getNodeSum(j, graphAdjMatrix))/(2*m))

        return sum(partialSums)/(2*m)

    def computeNewNode2Doc2Vec(self, community2nodes, nodeId2Doc2Vec):

        communities = list(filter(lambda x: len(community2nodes[x]) > 0, community2nodes.keys()))

        nodeId2Doc2VecTemp = {}

        for communityId in range(len(communities)):
            community = communities[communityId]
            vectors = []
            for node in community2nodes[community]:
                vectors.append(nodeId2Doc2Vec[node])
            nodeId2Doc2VecTemp[communityId] = self.getCentroid(vectors)
                
        return nodeId2Doc2VecTemp

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
                if (graphAdjMatrix[i][j] != 0 and graphAdjMatrix[i][j] != 0):
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

    def louvain(self, graphAdjMatrix, nodeId2Doc2Vec = None):

        start_time = time.time()

        theta = 0.0001
        alpha = 0.0003

        isFirstPass = True

        while True:

            if isFirstPass:
                (node2community, community2nodes) = self.initialize(graphAdjMatrix)
                graphAdjMatrixFull = graphAdjMatrix

                initialModularityFull = self.computeModularity(graphAdjMatrix, community2nodes)

                if (nodeId2Doc2Vec != None):
                    (modularityMatrixInertia, distanceMatrix, node2Inertia, fullInertia) = self.computeModularityInertiaUtils(nodeId2Doc2Vec, graphAdjMatrix)
                    initialModularityFull += self.computeModularityInertia(modularityMatrixInertia, community2nodes)
                    

            print('Started Louvain first phase')

            initialModularity = initialModularityFull

            while True:

                noNodes = np.shape(graphAdjMatrix)[0]
                nodes = list(range(noNodes))

                for node in nodes:
                    # shuffle(nodes)

                    neis = self.getNodeNeighs(node, graphAdjMatrix)

                    modularityGains = []

                    for neigh in neis:
                        
                        neighCommunity = node2community[neigh]
                        nodeCommunity = node2community[node]

                        if (neighCommunity == nodeCommunity):
                            continue

                        fullModularityGain = self.computeModularityGain(node, neighCommunity, graphAdjMatrix, community2nodes) - \
                            self.computeModularityGain(node, nodeCommunity, graphAdjMatrix, community2nodes)

                        if (nodeId2Doc2Vec != None):
                            modularityGainWithInertia = self.computeModularityGainInertia(node, neighCommunity, nodeCommunity, distanceMatrix, node2Inertia, community2nodes, fullInertia)
                            fullModularityGain = alpha * fullModularityGain + (1 - alpha) * modularityGainWithInertia

                        if (fullModularityGain > 0):
                            modularityGains.append((int(neighCommunity), fullModularityGain))

                    if (len(modularityGains) > 0):
                        
                        # get max modularity community
                        modularityGains = np.array(modularityGains, dtype = float)
                        maxModularityGainIndex = np.argmax(modularityGains[:, 1])
                        maxModularityGainIndices = np.where(modularityGains[:, 1]==modularityGains[maxModularityGainIndex][1])

                        maxModularityNeighs = [int(modularityGains[mIndex[0]][0]) for mIndex in maxModularityGainIndices]

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
                
                if (nodeId2Doc2Vec != None):
                    newModularity += self.computeModularityInertia(modularityMatrixInertia, community2nodes)

                if (newModularity - initialModularity <= theta):
                    break
                
                initialModularity = newModularity

            print('Finished Louvain first phase')

            print('Start Louvain second phase')

            if isFirstPass:
                community2nodesFull = community2nodes
                node2communityFull = node2community
                # cache previous step configuration in case modularity decreases instead of increasing
                prevNode2communityFull = node2community
                new2oldCommunities = dict(zip(community2nodes.keys(), community2nodes.keys()))
            else:
                # cache previous step configuration in case modularity decreases instead of increasing
                prevNode2communityFull = node2communityFull
                (node2communityFull, community2nodesFull) = self.decompressSupergraph(community2nodes, community2nodesFull, new2oldCommunities)               
            
            newModularityFull = self.computeModularity(graphAdjMatrixFull, community2nodesFull)

            if (nodeId2Doc2Vec != None):
                newModularityFull += self.computeModularityInertia(modularityMatrixInertia, community2nodes)

            print('Second phase modularity', newModularityFull)

            if (newModularityFull - initialModularityFull <= theta):
                # restore previous step configuration
                node2communityFull = prevNode2communityFull
                break
            
            initialModularityFull = newModularityFull

            if (nodeId2Doc2Vec != None):
                nodeId2Doc2Vec = self.computeNewNode2Doc2Vec(community2nodes, nodeId2Doc2Vec)
            (graphAdjMatrix, new2oldCommunities) = self.computeNewAdjMatrix(community2nodes, new2oldCommunities, graphAdjMatrix)
            (node2community, community2nodes) = self.initialize(graphAdjMatrix)

            if (nodeId2Doc2Vec != None):
                (modularityMatrixInertia, distanceMatrix, node2Inertia, fullInertia) = self.computeModularityInertiaUtils(nodeId2Doc2Vec, graphAdjMatrix)
            isFirstPass = False

        print("--- %s execution time in seconds ---" % (time.time() - start_time))

        return node2communityFull

                
