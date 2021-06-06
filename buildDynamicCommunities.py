import igraph
import pymongo
from igraph import Graph, VertexClustering
import numpy as np

# https://plotly.com/python/sankey-diagram/

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSABidenInauguration

def getAllSnapshots(prefix):

    allCollections = db.list_collection_names()

    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

def getCommunitiesForSnapshot(collectionName, timeStep, communityAttribute):

    allComments = list(db[collectionName].find())

    timeStepDict = {}

    for x in allComments:
        dictKey = str(x[communityAttribute]) + '_' + str(timeStep)
        
        if dictKey in timeStepDict:
            timeStepDict[dictKey].append(x['author'])
        else:
            timeStepDict[dictKey] = [x['author']]

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

    snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep, 'clusterIdSimple')

    '''
    frontsEvents[frontEvent][frontId] = [front1, front2, ...]
    1 = front x was replaced by fronts list
    2 = a new front must be added
    '''
    frontEvents = {1: {}, 2: []}

    # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
    for communityIdA in snapshotCommunities:

        authorsA = list(set(snapshotCommunities[communityIdA]))
        
        bestJaccard = 0.2
        bestFrontId = None

        for frontId in range(len(fronts)):

            authorsB = list(set(fronts[frontId]))
            
            intersect = len(list(set(authorsA) & set(authorsB)))
            reunion = len(authorsA + authorsB)

            jaccard = intersect/reunion

            if (jaccard > bestJaccard):
                bestJaccard = jaccard
                bestFrontId = frontId
                # print('J', jaccard)
                # print('A', authorsA, 'B', authorsB)
        
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