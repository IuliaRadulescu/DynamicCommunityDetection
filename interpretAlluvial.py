import numpy as np
import alluvialDataRetriever
import plotAlluvial

def determineDynamicCommunitiesDFS(alluvialData):

    # determine leaves
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
    

def computeStats():

    print('GENERATING STATS FOR HYBRID')

    datasets = ['tennis']

    similarities = [75, 80, 85, 90, 95]

    for dataset in datasets:
        for similarity in similarities:
            print('')
            print('STATS FOR', 'dataset', dataset, 'similarity', similarity)

            alluvialData = alluvialDataRetriever.getAlluvialDataHybrid(dataset, similarity)

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

    print('GENERATING STATS FOR CLASSIC')

    similarities = [75, 80, 85, 90, 95]

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

def generateDynamicAndPlot(datasetType, dataset, similarity, alpha = None):

    print('STARTED DYNAMIC COMM AND PLOT GENERATION FOR', datasetType, dataset, similarity, alpha)

    outputFileName = 'DYNAMIC_AVG_' + datasetType + '_' + dataset + '_' + str(similarity) + '_' + str(alpha)  + '.json' if alpha != None else 'DYNAMIC_AVG_' + datasetType + '_' + dataset + '_' + str(similarity) + '.json'

    if datasetType == 'classic':
        print('CLASSIC ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataClassic(dataset, similarity)
    else:
        print('HYBRID TEXT ALLUVIAL')
        alluvialData = alluvialDataRetriever.getAlluvialDataHybrid(dataset, similarity)

    print('STARTED GENERATION DYNAMIC')

    dynamicCommunities = determineDynamicCommunitiesDFS(alluvialData)

    # sort dynamicCommunities by lentgh of dynamic communities
    dynamicCommunities.sort(key=len)

    if datasetType == 'classic':
        longestDynamicItems = dynamicCommunities[len(dynamicCommunities) - 1]
    else:
        longestDynamicItems = dynamicCommunities[int(len(dynamicCommunities)/2)]

    filteredAlluvialData = filterAlluvialDataDFS(alluvialData, longestDynamicItems, maxWidth=10)

    print('STARTED PLOTTING IMAGE FOR', outputFileName)

    if datasetType == 'classic':
        plotAlluvial.feedSankeyJsonClassic(filteredAlluvialData, outputFileName)
    else:
        plotAlluvial.feedSankeyJsonHybrid(filteredAlluvialData, outputFileName)

generateDynamicAndPlot('hybrid', 'tennis', 75)
generateDynamicAndPlot('hybrid', 'tennis', 80)
generateDynamicAndPlot('hybrid', 'tennis', 85)
generateDynamicAndPlot('hybrid', 'tennis', 90)
generateDynamicAndPlot('hybrid', 'tennis', 95)

generateDynamicAndPlot('classic', 'tennis', 75)
generateDynamicAndPlot('classic', 'tennis', 80)
generateDynamicAndPlot('classic', 'tennis', 85)
generateDynamicAndPlot('classic', 'tennis', 90)
generateDynamicAndPlot('classic', 'tennis', 95)

computeStats()
                
