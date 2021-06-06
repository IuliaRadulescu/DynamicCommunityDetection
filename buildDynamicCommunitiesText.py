import igraph
import pymongo
from igraph import Graph, VertexClustering
import numpy as np
from numpy import dot
from numpy.linalg import norm

# https://plotly.com/python/sankey-diagram/

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

def getAllSnapshots(prefix):

    allCollections = db.list_collection_names()

    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

def getCommunitiesForSnapshot(collectionName, timeStep, communityAttribute):

    allComments = list(db[collectionName].find())

    collectionCentroids = {}
    timeStepDict = {}

    for x in allComments:
        dictKey = str(x[communityAttribute]) + '_' + str(timeStep)

        if (x['clusterIdKMeans'] not in collectionCentroids):
            collectionCentroids[x['clusterIdKMeans']] = x['centroid']
        
        if dictKey in timeStepDict:
            timeStepDict[dictKey].append(x['clusterIdKMeans'])
        else:
            timeStepDict[dictKey] = [x['clusterIdKMeans']]

    for dictKey in timeStepDict:
        timeStepDict[dictKey] = set(timeStepDict[dictKey])
        timeStepDict[dictKey] = [collectionCentroids[clusterIdKMeans] for clusterIdKMeans in timeStepDict[dictKey]]

    return timeStepDict

'''
frontsEvents = {1: {}, 2: []}
'''
def updateFronts(fronts, frontEvents, frontId2CommunityId):

    for frontId in frontEvents[1]:
        # remove old front
        del fronts[frontId]
        # add replacements
        for item in frontEvents[1][frontId]:
            fronts += [item[1]]
            frontId2CommunityId[len(fronts)-1] = item[0]
    
    for frontId in frontEvents[2]:
        for item in frontEvents[2]:
            fronts += [item[1]]
            frontId2CommunityId[len(fronts)-1] = item[0]

    return (frontId2CommunityId, fronts)

def centroid(arr, timeStep):
    length = arr.shape[0]
    centroid = []
    for dim in range(arr.shape[1]):
        centroid.append(np.sum(arr[:, dim])/length)
    return np.array(centroid)

allSnapshots = getAllSnapshots('quarter')

'''
communitiesTimestepMapping[communityId_0_1] = [communityId_1_0, communityId_1_1, ...] 
'''
communitiesTimestepMapping = {}
fronts = []
'''
maps each front with its associated community at the associated specific timestep
'''
frontId2CommunityId = {}
timeStep = 0

snapshotCommunities0 = getCommunitiesForSnapshot(allSnapshots[0], 0, 'clusterIdSimple')

# the initial communities are the initial fronts
communitiesTimestepMapping = dict(zip(snapshotCommunities0.keys(), [[] for i in range(len(snapshotCommunities0))]))
fronts = [item[1] for item in snapshotCommunities0.items()]
frontId2CommunityId = dict(zip(range(len(fronts)), [communityId for communityId in snapshotCommunities0.keys()]))

for timeStep in range(1, len(allSnapshots)):

    print('TIME STEP', timeStep, 'comm', allSnapshots[timeStep])

    snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep, 'clusterIdSimple')

    '''
    frontsEvents[frontEvent][frontId] = [front1, front2, ...]
    1 = front x was replaced by fronts list
    2 = a new front must be added
    '''
    frontEvents = {1: {}, 2: []}

    # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
    for communityIdA in snapshotCommunities:

        centroidsA = snapshotCommunities[communityIdA]

        centroidOfCentroidsA = centroid(np.array(centroidsA), timeStep)
        
        bestCosine = 0.5
        bestFrontId = None

        for frontId in range(len(fronts)):

            centroidsB = fronts[frontId]

            centroidOfCentroidsB = centroid(np.array(centroidsB), timeStep)
            
            cosine = dot(centroidOfCentroidsA, centroidOfCentroidsB)/(norm(centroidOfCentroidsA)*norm(centroidOfCentroidsB))

            if (cosine > bestCosine):
                bestCosine = cosine
                bestFrontId = frontId
        
        if (bestFrontId != None):
            # front transformation event
            if (bestFrontId not in frontEvents[1]):
                frontEvents[1][bestFrontId] = []
            frontEvents[1][bestFrontId].append((communityIdA, snapshotCommunities[communityIdA]))
            if bestFrontId in frontId2CommunityId:
                bestFrontCommunityId = frontId2CommunityId[bestFrontId]
                communitiesTimestepMapping[bestFrontCommunityId].append(communityIdA)
        else:
            # front addition event
            frontEvents[2].append((communityIdA, snapshotCommunities[communityIdA]))

    # update mappings
    for key in snapshotCommunities.keys():
        communitiesTimestepMapping[key] = []

    (frontId2CommunityId, fronts) = updateFronts(fronts, frontEvents, frontId2CommunityId)

finalMappings = {}

for communityId in communitiesTimestepMapping:
    if (len(communitiesTimestepMapping[communityId]) > 0):
        if (communityId not in finalMappings):
            finalMappings[communityId] = []
        finalMappings[communityId] = communitiesTimestepMapping[communityId]
    

print(finalMappings)