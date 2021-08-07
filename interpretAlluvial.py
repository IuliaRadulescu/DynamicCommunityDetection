import numpy as np
import alluvialDataRetriever
import plotAlluvial
import sys
import os
from collections import deque
import humanize
import igraph
from igraph import Graph, plot
import pickle

# https://plotly.com/python/sankey-diagram/

def filterAlluvialData(alluvialData, communitiesToMonitor):

    print('STARTED FILTERING')

    filteredAlluvial = {}
    alreadyParsedGlobal = []

    for community in communitiesToMonitor:

        if (community not in alluvialData) or (community in alreadyParsedGlobal):
            continue

        communitiesSet = list(set(alluvialData[community]))

        if community not in filteredAlluvial:
            filteredAlluvial[community] = communitiesSet
        else:
            filteredAlluvial[community].extend(communitiesSet)

        alreadyParsedGlobal.append(community)
        alreadyParsedGlobal.extend(communitiesSet)
        alreadyParsedGlobal = list(set(alreadyParsedGlobal))

        communitiesQueue = deque()
        communitiesQueue.extend(communitiesSet)
        communitiesQueue = deque(set(communitiesQueue))

        alreadyParsed = []

        while len(communitiesQueue) > 0:

            communityInQueue = communitiesQueue.popleft()

            if (communityInQueue not in alluvialData) or (community in alreadyParsedGlobal):
                continue
            
            # filter duplicates
            communitiesToAddSet = set(alluvialData[communityInQueue])

            # check to see if any of the communities to add were previosly in the queue
            setDifferenceAddVsAlreadyParsed = communitiesToAddSet - set(alreadyParsed)
            listDifferenceAddVsAlradyParsed = list(setDifferenceAddVsAlreadyParsed)

            if len(listDifferenceAddVsAlradyParsed) == 0:
                continue

            if community not in filteredAlluvial:
                filteredAlluvial[communityInQueue] = listDifferenceAddVsAlradyParsed
            else:
                filteredAlluvial[communityInQueue].extend(listDifferenceAddVsAlradyParsed)
                filteredAlluvial[communityInQueue] = list(set(filteredAlluvial[communityInQueue]))

            alreadyParsed.extend(listDifferenceAddVsAlradyParsed)
            alreadyParsed = list(set(alreadyParsed))
            
            communitiesQueue.extend(listDifferenceAddVsAlradyParsed)
            communitiesQueue = deque(set(communitiesQueue))
            
            alreadyParsedGlobal.extend(alreadyParsed)
            alreadyParsedGlobal = list(set(alreadyParsedGlobal))

            print('QUEUE LEN', len(communitiesQueue))

    return filteredAlluvial

def determineDynamicCommunitiesAsGraph(alluvialData, pickleFilePath):

    def getNodesAndEdges():

        nodes = []
        edges = []

        for community in alluvialData:

            # skip already parsed roots
            if community in nodes:
                continue

            if community not in nodes:
                nodes.append(community)
                
            communitiesQueue = deque()

            communitiesQueue.extend(alluvialData[community])
            communitiesQueue = deque(set(communitiesQueue))

            alreadyParsed = []

            while len(communitiesQueue) > 0:
                
                communityInQueue = communitiesQueue.popleft()

                if (communityInQueue not in alluvialData) or (communityInQueue in nodes):
                    continue

                nodes.append(communityInQueue)

                # filter duplicates
                communitiesToAddSet = set(alluvialData[communityInQueue])

                # check to see if any of the communities to add were previosly in the queue
                setDifferenceAddVsAlreadyParsed = communitiesToAddSet - set(alreadyParsed)
                listDifferenceAddVsAlradyParsed = list(setDifferenceAddVsAlreadyParsed)

                if len(listDifferenceAddVsAlradyParsed) == 0:
                    continue

                alreadyParsed.extend(listDifferenceAddVsAlradyParsed)
                alreadyParsed = list(set(alreadyParsed))

                nodes.extend(listDifferenceAddVsAlradyParsed)
                nodes = list(set(nodes))
                
                edgesToAdd = list(tuple(zip([communityInQueue] * len(listDifferenceAddVsAlradyParsed), listDifferenceAddVsAlradyParsed)))
                edges.extend(edgesToAdd)
                edges = list(set(edges))

                communitiesQueue.extend(listDifferenceAddVsAlradyParsed)
                communitiesQueue = deque(set(communitiesQueue))
            
            print('FINISHED COMMUNITY', 'We have', len(nodes), 'nodes')

        return (nodes, edges)

    g = Graph()

    (nodes, edges) = getNodesAndEdges()
    print('GOT NODES', len(nodes), 'AND EDGES', len(edges))

    g.add_vertices(nodes)
    g.add_edges(edges)

    # save graph
    pickle.dump(g, open(pickleFilePath, 'wb'))


def getLongestCommunityKeys(pickleFilePath):

    print('Load graph')

    ng = pickle.load(open(pickleFilePath, 'rb'))

    farthestPoints = ng.farthest_points()

    print('Farthest points in graph', farthestPoints, ng.vs[farthestPoints[0]]['name'], ng.vs[farthestPoints[1]]['name'])

    path = ng.get_shortest_paths(farthestPoints[0], farthestPoints[1])[0]

    return [ng.vs[nodeId]['name'] for nodeId in path]


def determineDynamicCommunities(alluvialData):

    dynamicCommunities = []

    alreadyParsedGlobal = []
    
    for community in alluvialData:

        dynamicCommunity = []
        communitiesQueue = deque()

        dynamicCommunity.append(community)
        dynamicCommunity.extend(alluvialData[community])

        alreadyParsedGlobal.append(community)
        alreadyParsedGlobal.extend(alluvialData[community])
        alreadyParsedGlobal = list(set(alreadyParsedGlobal))

        communitiesQueue.extend(alluvialData[community])
        communitiesQueue = deque(set(communitiesQueue))

        alreadyParsed = []

        while len(communitiesQueue) > 0:

            communityInQueue = communitiesQueue.popleft()

            if communityInQueue not in alluvialData:
                continue
            
            # filter duplicates
            communitiesToAddSet = set(alluvialData[communityInQueue])

            # check to see if any of the communities to add were previosly in the queue
            setDifferenceAddVsAlreadyParsed = communitiesToAddSet - set(alreadyParsed)
            listDifferenceAddVsAlradyParsed = list(setDifferenceAddVsAlreadyParsed)

            if len(listDifferenceAddVsAlradyParsed) == 0:
                continue

            alreadyParsed.extend(listDifferenceAddVsAlradyParsed)
            alreadyParsed = list(set(alreadyParsed))

            dynamicCommunity.extend(listDifferenceAddVsAlradyParsed)
            communitiesQueue.extend(listDifferenceAddVsAlradyParsed)

            dynamicCommunity = list(set(dynamicCommunity))
            communitiesQueue = deque(set(communitiesQueue))
        
        alreadyParsedGlobal.extend(alreadyParsed)
        alreadyParsedGlobal = list(set(alreadyParsedGlobal))

        communitiesQueueSize = sys.getsizeof(communitiesQueue)    
        print('SIZEOF communitiesQueue HUMAN', humanize.naturalsize(communitiesQueueSize))

        dynamicCommunities.append(dynamicCommunity)

        dynamicCommunitiesSize = sys.getsizeof(dynamicCommunities)
        print('SIZEOF dynamicCommunities HUMAN', humanize.naturalsize(dynamicCommunitiesSize))
    
    return dynamicCommunities

def determineSpecificDynamics(alluvialData, longest=True, shortest=False, nthCommunity=None):

    dynamicCommunities = determineDynamicCommunities(alluvialData)
    dynamicCommunities.sort(key=len)

    if (longest == True):
        finalSelection = dynamicCommunities[-1]
    elif (shortest == True):
        finalSelection = dynamicCommunities[0]
    elif (nthCommunity != None):
        finalSelection = dynamicCommunities[nthCommunity]

    return finalSelection[0]
    

def computeStats(filteredAlluvialData):

    ###### COMPUTE DYNAMIC COMMUNIY WIDTH

    communityWidths = [len(filteredAlluvialData[key]) for key in filteredAlluvialData]

    minCommunityWidth = min(communityWidths)
    maxCommunityWidth = max(communityWidths)
    meanCommunityWidth = np.mean(communityWidths)

    print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

    ###### COMPUTE DYNAMIC COMMUNIY DEPTH

    dynamicCommunities = determineDynamicCommunities(filteredAlluvialData)

    # print(dynamicCommunities)

    communityDepths = [len(dynamicCommunity) for dynamicCommunity in dynamicCommunities]

    minCommunityDepth = min(communityDepths)
    maxCommunityDepth = max(communityDepths)
    meanCommunityDepth = np.mean(communityDepths)

    print('Min depth', minCommunityDepth, 'Max depth', maxCommunityDepth, 'Mean depth', meanCommunityDepth)

def plotImageStatsClassic(dataset='biden'):

    alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, 50)

    # shortest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, False, True)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_classic_50_shortest_lived_1.png')

    print('PLOTTED ' + dataset + ' 50 SHORTEST')

    # longest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, True, False)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_classic_50_longest_lived_1.png')

    print('PLOTTED ' + dataset + ' 50 LONGEST')

    alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, 70)

    # shortest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, False, True)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_classic_70_shortest_lived_1.png')

    print('PLOTTED ' + dataset + ' 70 SHORTEST')

    # longest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, True, False)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_classic_70_longest_lived_1.png')

    print('PLOTTED ' + dataset + ' 70 LONGEST')

def plotImageStatsHybrid(dataset='biden', alpha=60):

    alluvialData = alluvialDataRetriever.getAlluvialDataHybrid('biden', 50, alpha)

    # shortest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, False, True)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_hybrid_50_alpha_' + str(alpha) + '_shortest_lived_1.png')

    print('PLOTTED ' + dataset + ' 50 SHORTEST')

    # longest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, True, False)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_hybrid_50_alpha_' + str(alpha) + '_longest_lived_1.png')

    print('PLOTTED ' + dataset + ' 50 LONGEST')

    alluvialData = alluvialDataRetriever.getAlluvialDataHybrid('biden', 70, alpha)

    # shortest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, False, True)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_hybrid_70_alpha_' + str(alpha) + '_shortest_lived_1.png')

    print('PLOTTED ' + dataset + ' 70 SHORTEST')

    # longest
    specificDynamicCommunityKey = determineSpecificDynamics(alluvialData, True, False)
    filteredAlluvialData = filterAlluvialData(alluvialData, [specificDynamicCommunityKey])
    plotAlluvial.plotAlluvialImage(filteredAlluvialData, dataset + '_hybrid_70_alpha_' + str(alpha) + '_longest_lived_1.png')

    print('PLOTTED ' + dataset + ' 70 LONGEST')

def plotImageFromGraph(alluvialData, pickleFilePath, outputImageFileName):

    nodes = getLongestCommunityKeys(pickleFilePath)

    filteredAlluvialData = filterAlluvialData(alluvialData, nodes)

    print('STARTED PLOTTING IMAGE FOR', outputImageFileName)

    plotAlluvial.plotAlluvialImage(filteredAlluvialData, outputImageFileName)

def generatePickleAndPlot(datasetType, dataset, similarity, alpha = None):

    print('STARTED GRAPH GENERATION FOR', datasetType, dataset, similarity, alpha)

    graphModelsDirPath = os.path.dirname(os.path.realpath(__file__)) + '/GRAPH_MODELS'
    pickleFileName = datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha) + '.pickle' if alpha != None else datasetType + '_' + dataset + '_' + str(similarity)  + '.pickle'
    pickleFilePath = graphModelsDirPath + '/' + pickleFileName
    outputImageFileName = 'GRAPH_' + datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha)  + '.png' if alpha != None else 'GRAPH_' + datasetType + '_' + dataset + '_' + str(similarity) + '.png'

    alluvialData = alluvialDataRetriever.getAlluvialDataHybrid(dataset, similarity, alpha) if datasetType == 'hybrid' else alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)

    print('STARTED GENERATION PICKLE')

    # determineDynamicCommunitiesAsGraph(alluvialData, pickleFilePath)
    
    print('FINISHED GENERATING PICKLE')
    
    plotImageFromGraph(alluvialData, pickleFilePath, outputImageFileName)

def generateGraphsAndImagesForDatasets():

    print('GENERATING GRAPHS FOR HYBRID')

    datasets = ['biden', 'protests']

    similarities = [50, 70]

    alphas = [50, 40]

    for dataset in datasets:
        for similarity in similarities:
            for alpha in alphas:
                generatePickleAndPlot('hybrid', dataset, similarity, alpha)

    print('GENERATING GRAPHS FOR CLASSIC')

    for dataset in datasets:
        for similarity in similarities:
            generatePickleAndPlot('classic', dataset, similarity)


generateGraphsAndImagesForDatasets()

# generatePickleAndPlot('hybrid', 'protests', 50, 50)


                
