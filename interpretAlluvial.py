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

    print('STARTED FILTERING ===')

    filteredAlluvial = {}
    alreadyParsedGlobal = []

    for community in communitiesToMonitor:

        if (community not in alluvialData) or (community in alreadyParsedGlobal):
            continue

        alreadyParsedGlobal.append(community)

        communitiesDeduplicated = list(set(alluvialData[community]))

        if community not in filteredAlluvial:
            filteredAlluvial[community] = communitiesDeduplicated

        communitiesQueue = deque()
        communitiesQueue.extend(communitiesDeduplicated)

        alreadyParsed = []

        while len(communitiesQueue) > 0:

            communityInQueue = communitiesQueue.popleft()

            if (communityInQueue not in alluvialData) or (communityInQueue in alreadyParsedGlobal):
                continue

            alreadyParsedGlobal.append(communityInQueue)
            
            # filter duplicates
            communitiesToAddSet = set(alluvialData[communityInQueue])

            if communityInQueue not in filteredAlluvial:
                filteredAlluvial[communityInQueue] = list(communitiesToAddSet)

            # check to see if any of the communities to add were previosly in the queue
            setDifferenceAddVsAlreadyParsed = communitiesToAddSet - set(alreadyParsed)
            listDifferenceAddVsAlredyParsed = list(setDifferenceAddVsAlreadyParsed)

            if len(listDifferenceAddVsAlredyParsed) == 0:
                continue

            alreadyParsed.extend(listDifferenceAddVsAlredyParsed)
            alreadyParsed = list(set(alreadyParsed))

            alreadyParsedGlobal.extend(listDifferenceAddVsAlredyParsed)
            alreadyParsedGlobal = list(set(alreadyParsedGlobal))
            
            communitiesQueue.extend(listDifferenceAddVsAlredyParsed)
            communitiesQueue = deque(set(communitiesQueue))

            # print('QUEUE LEN', len(communitiesQueue))

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

        if (community not in alluvialData) or (community in alreadyParsedGlobal):
            continue

        alreadyParsedGlobal.append(community)

        dynamicCommunity = []
        dynamicCommunity.append(community)

        communitiesQueue = deque()
        communitiesQueue.extend(alluvialData[community])
        communitiesQueue = deque(set(communitiesQueue))

        # already parsed for queue scope
        alreadyParsed = []

        while len(communitiesQueue) > 0:

            communityInQueue = communitiesQueue.popleft()

            if (communityInQueue not in alluvialData) or (communityInQueue in alreadyParsedGlobal):
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

            alreadyParsedGlobal.extend(listDifferenceAddVsAlradyParsed)
            alreadyParsedGlobal = list(set(alreadyParsedGlobal))

            dynamicCommunity.extend(listDifferenceAddVsAlradyParsed)
            dynamicCommunity = list(set(dynamicCommunity))

            communitiesQueue.extend(listDifferenceAddVsAlradyParsed)
            communitiesQueue = deque(set(communitiesQueue))

        # communitiesQueueSize = sys.getsizeof(communitiesQueue)    
        # print('SIZEOF communitiesQueue HUMAN', humanize.naturalsize(communitiesQueueSize))

        dynamicCommunities.append(dynamicCommunity)

        # dynamicCommunitiesSize = sys.getsizeof(dynamicCommunities)
        # print('SIZEOF dynamicCommunities HUMAN', humanize.naturalsize(dynamicCommunitiesSize))
    
    return dynamicCommunities

def determineDynamicCommunitiesDFS(alluvialData):

    # determine leafs

    nonLeaves = list(set(alluvialData.keys()))

    alreadyParsedGlobal = set()

    stack = []
    dynamicCommunities = []

    for communityId in alluvialData:

        if (communityId not in alluvialData) or (communityId in alreadyParsedGlobal):
            continue

        stack.append((communityId, [communityId]))

        while len(stack) > 0:

            (communityId, path) = stack.pop()

            if (communityId not in alreadyParsedGlobal) and (communityId not in nonLeaves):
                dynamicCommunities.append(path)

            if (communityId not in alluvialData) or (communityId in alreadyParsedGlobal):
                continue
            
            alreadyParsedGlobal.add(communityId)
            adjacentCommunities = alluvialData[communityId]

            for adjCommunity in adjacentCommunities:
                stack.append((adjCommunity, path + [adjCommunity]))

    return dynamicCommunities

def filterAlluvialDataDFS(alluvialData, communitiesToMonitor, maxWidth = None):

    filteredAlluvial = {}

    for communityId in communitiesToMonitor:
        if (communityId in alluvialData):

            allNeighs = alluvialData[communityId]

            if maxWidth == None:
                filteredAlluvial[communityId] = allNeighs
            else:
                justMaxWidth = allNeighs[0:maxWidth]
                mandatory = list(set(communitiesToMonitor) & set(allNeighs))
                filteredAlluvial[communityId] = list(set(justMaxWidth) - set(mandatory)) + mandatory

    return filteredAlluvial


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
    

def computeStats():

    print('GENERATING STATS FOR HYBRID')

    datasets = ['biden', 'protests']

    similarities = [70, 80, 85, 90]

    # for dataset in datasets:
    #     for similarity in similarities:
    #         print('')
    #         print('STATS FOR', 'dataset', dataset, 'similarity', similarity)

    #         alluvialData = alluvialDataRetriever.getAlluvialDataHybridText(dataset, similarity)

    #         print('--> width')
    #         communityWidths = [len(list(set(alluvialData[key]))) for key in alluvialData]
    #         minCommunityWidth = min(communityWidths)
    #         maxCommunityWidth = max(communityWidths)
    #         meanCommunityWidth = np.mean(communityWidths)
    #         print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

    #         print('--> depth')
    #         lenDynamicCommunities = [len(dynamicCommunity) for dynamicCommunity in determineDynamicCommunitiesDFS(alluvialData)]
    #         minCommunityDepth = min(lenDynamicCommunities)
    #         maxCommunityDepth = max(lenDynamicCommunities)
    #         meanCommunityDepth = np.mean(lenDynamicCommunities)
    #         print('Min depth', minCommunityDepth, 'Max depth', maxCommunityDepth, 'Mean depth', meanCommunityDepth)

    print('GENERATING STATS FOR CLASSIC')

    datasets = ['biden', 'protests']

    similarities = [70, 80, 85, 90]

    for dataset in datasets:
        for similarity in similarities:
            print('')
            print('STATS FOR', 'dataset', dataset, 'similarity', similarity)

            alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)

            print('--> width')
            communityWidths = [len(list(set(alluvialData[key]))) for key in alluvialData]
            minCommunityWidth = min(communityWidths)
            maxCommunityWidth = max(communityWidths)
            meanCommunityWidth = np.mean(communityWidths)
            print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

            print('--> depth')
            lenDynamicCommunities = [len(dynamicCommunity) for dynamicCommunity in determineDynamicCommunitiesDFS(alluvialData)]
            minCommunityDepth = min(lenDynamicCommunities)
            maxCommunityDepth = max(lenDynamicCommunities)
            meanCommunityDepth = np.mean(lenDynamicCommunities)
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

    print('PICKE', pickleFilePath)

    nodes = getLongestCommunityKeys(pickleFilePath)

    print('Nodes are', nodes)

    filteredAlluvialData = filterAlluvialData(alluvialData, nodes)

    print('STARTED PLOTTING IMAGE FOR', outputImageFileName)

    plotAlluvial.plotAlluvialImage(filteredAlluvialData, outputImageFileName)

def generatePickleAndPlot(datasetType, dataset, similarity, alpha = None):

    print('STARTED GRAPH GENERATION FOR', datasetType, dataset, similarity, alpha)

    graphModelsDirPath = os.path.dirname(os.path.realpath(__file__)) + '/GRAPH_MODELS'
    pickleFileName = datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha) + '.pickle' if alpha != None else datasetType + '_' + dataset + '_' + str(similarity)  + '.pickle'
    pickleFilePath = graphModelsDirPath + '/' + pickleFileName
    outputImageFileName = 'GRAPH_' + datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha)  + '.png' if alpha != None else 'GRAPH_' + datasetType + '_' + dataset + '_' + str(similarity) + '.png'

    if datasetType == 'hybrid':
        print('JUST HYBRID ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataHybrid(dataset, similarity, alpha) 
    elif datasetType == 'hybridText':
        print('HYBRID TEXT ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataHybridText(dataset, similarity)
    else:
        print('CLASSIC ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)

    print('STARTED GENERATION PICKLE')

    # determineDynamicCommunitiesAsGraph(alluvialData, pickleFilePath)
    
    print('FINISHED GENERATING PICKLE')
    
    plotImageFromGraph(alluvialData, pickleFilePath, outputImageFileName)

def generateDynamicAndPlot(datasetType, dataset, similarity, alpha = None):

    print('STARTED DYNAMIC COMM AND PLOT GENERATION FOR', datasetType, dataset, similarity, alpha)

    outputImageFileName = 'DYNAMIC_AVG_' + datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha)  + '.png' if alpha != None else 'DYNAMIC_AVG_' + datasetType + '_' + dataset + '_' + str(similarity) + '.png'

    if datasetType == 'hybrid':
        print('JUST HYBRID ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataHybrid(dataset, similarity, alpha) 
    elif datasetType == 'hybridText':
        print('HYBRID TEXT ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataHybridText(dataset, similarity)
    else:
        print('CLASSIC ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)

    print('STARTED GENERATION DYNAMIC')

    dynamicCommunities = determineDynamicCommunitiesDFS(alluvialData)

    # sort dynamicCommunities by lentgh of dynamic communities
    dynamicCommunities.sort(key=len)

    longestDynamicItems = dynamicCommunities[int(len(dynamicCommunities)/2)]

    print(dynamicCommunities[int(len(dynamicCommunities)/2)])

    print('Longest dynamic has length', len(longestDynamicItems))

    filteredAlluvialData = filterAlluvialDataDFS(alluvialData, longestDynamicItems, 10)

    print('STARTED PLOTTING IMAGE FOR', outputImageFileName)

    print('Filtered alluvial', filteredAlluvialData)

    # plotAlluvial.plotAlluvialImage(filteredAlluvialData, outputImageFileName)
    
    # print('STARTED PLOTTING IMAGE')

def generateGraphsAndImagesForDatasets():

    print('GENERATING GRAPHS FOR HYBRID')

    datasets = ['biden', 'protests']

    similarities = [30, 50, 70]

    # alphas = [50, 40]

    # for dataset in datasets:
    #     for similarity in similarities:
    #         for alpha in alphas:
    #             generatePickleAndPlot('hybrid', dataset, similarity, alpha)

    print('GENERATING GRAPHS FOR CLASSIC')

    for dataset in datasets:
        for similarity in similarities:
            generatePickleAndPlot('classic', dataset, similarity)

def generateGraphsAndImagesForDatasetsHybridText():

    print('GENERATING GRAPHS FOR HYBRID TEXT')

    datasets = ['biden', 'protests']

    similarities = [70, 80, 85, 90]

    for dataset in datasets:
        for similarity in similarities:
                generatePickleAndPlot('hybridText', dataset, similarity)

def comuputeDepthStatsUsingGraph(datasetType, dataset, similarity, alpha=None):

    graphModelsDirPath = os.path.dirname(os.path.realpath(__file__)) + '/GRAPH_MODELS'
    pickleFileName = datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha) + '.pickle' if alpha != None else datasetType + '_' + dataset + '_' + str(similarity)  + '.pickle'
    pickleFilePath = graphModelsDirPath + '/' + pickleFileName

    # print('Load graph')

    ng = pickle.load(open(pickleFilePath, 'rb'))
    maxCommunityDepth = ng.farthest_points()[2]
    avgCommunityDepth = ng.average_path_length(directed=False)

    print('Min depth', 1, 'Max depth', maxCommunityDepth, 'Mean depth', avgCommunityDepth)
    
def computeWidthStats(alluvialData):

    print('GENERATING STATS FOR HYBRID')

    datasets = ['biden', 'protests']

    similarities = [70, 80, 85, 90]

    for dataset in datasets:
        for similarity in similarities:
            print('')
            print('STATS FOR', 'dataset', dataset, 'similarity', similarity)
            alluvialData = alluvialDataRetriever.getAlluvialDataHybridText(dataset, similarity)
            print('--> width')
            computeWidthStats(alluvialData)
            print('--> depth')
            comuputeDepthStatsUsingGraph('hybridText', dataset, similarity)

    communityWidths = [len(list(set(alluvialData[key]))) for key in alluvialData]

    minCommunityWidth = min(communityWidths)
    maxCommunityWidth = max(communityWidths)
    meanCommunityWidth = np.mean(communityWidths)

    print('Min width', minCommunityWidth, 'Max width', maxCommunityWidth, 'Mean width', meanCommunityWidth)

def computeStatsForDatasets():

    print('GENERATING STATS FOR HYBRID')

    datasets = ['biden', 'protests']

    similarities = [70, 80, 85, 90]

    for dataset in datasets:
        for similarity in similarities:
            print('')
            print('STATS FOR', 'dataset', dataset, 'similarity', similarity)
            alluvialData = alluvialDataRetriever.getAlluvialDataHybridText(dataset, similarity)
            print('--> width')
            computeWidthStats(alluvialData)
            print('--> depth')
            comuputeDepthStatsUsingGraph('hybridText', dataset, similarity)

    # print('GENERATING STATS FOR CLASSIC')

    # for dataset in datasets:
    #     for similarity in similarities:
    #         print('STATS FOR', 'dataset', dataset, 'similarity', similarity)
    #         alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)
    #         computeWidthStats(alluvialData)
    #         comuputeDepthStatsUsingGraph('classic', dataset, similarity)

generateDynamicAndPlot('hybridText', 'biden', 70)
# generateDynamicAndPlot('hybridText', 'protests', 70)
# generateDynamicAndPlot('hybridText', 'biden', 80)
# generateDynamicAndPlot('hybridText', 'protests', 80)
# generateDynamicAndPlot('hybridText', 'biden', 85)
# generateDynamicAndPlot('hybridText', 'protests', 85)
# generateDynamicAndPlot('hybridText', 'biden', 90)
# generateDynamicAndPlot('hybridText', 'protests', 90)
# generateDynamicAndPlot('hybridText', 'protests', 95)
# generateDynamicAndPlot('hybridText', 'biden', 95)

# generateDynamicAndPlot('hybridText', 'biden', 85)
# generateDynamicAndPlot('hybridText', 'biden', 95)

# generateDynamicAndPlot('hybridText', 'protests', 85)
# generateDynamicAndPlot('hybridText', 'protests', 95)

# generateDynamicAndPlot('classic', 'biden', 70)
# generateDynamicAndPlot('classic', 'protests', 70)
# generateDynamicAndPlot('classic', 'biden', 80)
# generateDynamicAndPlot('classic', 'protests', 80)
# generateDynamicAndPlot('classic', 'biden', 85)
# generateDynamicAndPlot('classic', 'protests', 85)
# generateDynamicAndPlot('classic', 'biden', 90)
# generateDynamicAndPlot('classic', 'protests', 90)

# generatePickleAndPlot('classic', 'biden', 30)
# generatePickleAndPlot('classic', 'biden', 50)
# generatePickleAndPlot('classic', 'biden', 70)
# generatePickleAndPlot('classic', 'protests', 30)
# generatePickleAndPlot('classic', 'protests', 50)
# generatePickleAndPlot('classic', 'protests', 70)

# computeStats()
                
