import pymongo
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel

# https://plotly.com/python/sankey-diagram/
'''
@returns: a list of sorted strings representing the database collections
'''
def getAllSnapshots(prefix):

    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionUSABidenInauguration

    allCollections = db.list_collection_names()

    allCollections = list(filter(lambda x: prefix in x, allCollections))

    dbClient.close()

    return sorted(allCollections)

'''
@returns: dictionary, where keys are strings of the form: communityId_timeStep
            and the values are numpy arrays -> a numpy array is the centroid of distinct centroids for the objects, retrieved by communityAttribute
'''
def getCommunitiesForSnapshot(collectionName, timeStep, communityAttribute):

    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionUSABidenInauguration

    allComments = list(db[collectionName].find())

    dbClient.close()

    print('Finished reading comments from mongo!', collectionName)

    collectionCentroids = {}
    timeStepDict = {}

    for x in allComments:
        dictKey = str(x[communityAttribute]) + '_' + str(timeStep)

        if (x['clusterIdKMeans'] not in collectionCentroids):
            collectionCentroids[x['clusterIdKMeans']] = tuple(x['centroid'])
        
        if dictKey in timeStepDict:
            timeStepDict[dictKey].append(x['clusterIdKMeans'])
        else:
            timeStepDict[dictKey] = [x['clusterIdKMeans']]

    for dictKey in timeStepDict:
        # we need all centroids! so the mean will be accurate
        dictKeyCentroids = tuple((collectionCentroids[clusterIdKMeans] for clusterIdKMeans in timeStepDict[dictKey]))
        # compute centroid of centroids
        timeStepDict[dictKey] = tuple(centroid(np.array(dictKeyCentroids)))

    return timeStepDict

'''
frontsEvents = {1: {}, 2: []}
'''
def updateFronts(fronts, frontEvents, frontId2CommunityId):

    for frontId in frontEvents[1]:
        # remove old front
        del fronts[frontId]
        # add replacements
        for frontMergeEvent in frontEvents[1][frontId]:
            eventKey = frontMergeEvent[0]
            centroid = frontMergeEvent[1]

            fronts.append(centroid)
            frontId2CommunityId[len(fronts)-1] = eventKey
            
    for frontMergeEvent in frontEvents[2]:
        eventKey = frontMergeEvent[0]
        centroid = frontMergeEvent[1]

        fronts.append(centroid)
        frontId2CommunityId[len(fronts)-1] = eventKey

    fronts = list(set(fronts))

    return (frontId2CommunityId, fronts)

def centroid(arr):
    length = arr.shape[0]
    centroid = []
    for dim in range(arr.shape[1]):
        centroid.append(np.sum(arr[:, dim])/length)
    return np.array(centroid) * 10

def communityItemsToTuples(community):
    return [tuple(item) for item in community]

def averageEuclideanPairwise(arrA, arrB):

    distances = []

    for elemA in arrA:
        for elemB in arrB:
            distances.append(np.linalg.norm(elemA - elemB))

    return np.mean(np.array(distances))

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

'''
fronts = list of fronts; a front = a tuple of tuples, where a tuple represents a centroid
'''
fronts = list(set([centroid for _, centroid in snapshotCommunities0.items()]))
frontId2CommunityId = dict(zip(range(len(fronts)), [communityId for communityId in snapshotCommunities0.keys()]))

for timeStep in range(1, len(allSnapshots)):

    snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep, 'clusterIdSimple')

    '''
    frontsEvents[frontEvent][frontId] = [front1, front2, ...]
    1 = front x was replaced by fronts list
    2 = a new front must be added
    '''
    frontEvents = {1: {}, 2: []}

    # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
    for communityIdA in snapshotCommunities:
        centroidA = snapshotCommunities[communityIdA]
        processedCentroidsA = np.array([centroidA])
        
        # similarity ranges from 0 to 1
        bestSim = 0.5
        bestFrontId = None

        similarities = []
        distancesToFronts = {}

        processedCentroidsB = []

        for frontId in range(len(fronts)):
            centroidB = fronts[frontId]
            processedCentroidsB.append(centroidB)

        processedCentroidsB = np.array(processedCentroidsB)

        distances = rbf_kernel(processedCentroidsA, processedCentroidsB)[0]

        distancesSortedIndexes = np.argsort(distances)
        maxDistanceIndex = distancesSortedIndexes[-1]

        # print('Similarity:', distances[maxDistanceIndex])

        if (distances[maxDistanceIndex] > bestSim):
            bestFrontId = maxDistanceIndex

        if (distances[maxDistanceIndex] < 0.01):
            print('=================')
            print('Processed centroids A', processedCentroidsA)
            print('Processed centroids B', processedCentroidsB)
            print('Snapshot:', allSnapshots[timeStep])
        
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

    # update mappings so we have the new snapshot keys
    for key in snapshotCommunities.keys():
        communitiesTimestepMapping[key] = []

    (frontId2CommunityId, fronts) = updateFronts(fronts, frontEvents, frontId2CommunityId)

    print('We have', len(fronts), 'fronts')

finalMappings = {}

for communityId in communitiesTimestepMapping:
    if (len(communitiesTimestepMapping[communityId]) > 0):
        finalMappings[communityId] = communitiesTimestepMapping[communityId]
    
print(finalMappings)