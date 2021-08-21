import pymongo
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
import json

def doComputation(dbName, optimalSim, outputFileName):

    '''
    @returns: a list of sorted strings representing the database collections
    '''
    def getAllSnapshots(prefix):

        dbClient = pymongo.MongoClient('localhost', 27017)
        db = dbClient[dbName]

        allCollections = db.list_collection_names()

        allCollections = list(filter(lambda x: prefix in x, allCollections))

        dbClient.close()

        return sorted(allCollections)

    '''
    @returns: dictionary, where keys are strings of the form: communityId_timeStep
                value = a dict with two keys: TYPE_STRUCTURAL -> a list of authors 
                            and 
                        TYPE_TEXTUAL -> a list of tuples of tuples: ( (centroid1_1, centroid1_2), (centroid2_1), (centroid3_1, centroid3_2, centroid3_3) ) and centroidx_y = (i1, i2, ..., i24)
    '''
    def getCommunitiesForSnapshot(collectionName, timeStep):

        dbClient = pymongo.MongoClient('localhost', 27017)
        db = dbClient[dbName]

        allComments = list(db[collectionName].find())

        dbClient.close()

        print('Finished reading comments from mongo!', collectionName)

        author2Attributes = {}
        collectionCentroids = {}

        for comment in allComments:
            if comment['author'] not in author2Attributes:
                author2Attributes[comment['author']] = {
                    'structuralId': comment['clusterIdSimple'],
                    'textualIds': [comment['clusterIdKMeans']],
                }
            else:
                author2Attributes[comment['author']]['textualIds'].append(comment['clusterIdKMeans'])

            if (comment['clusterIdKMeans'] not in collectionCentroids):
                collectionCentroids[comment['clusterIdKMeans']] = tuple(comment['centroid'])
        
        timeStepDict = {}

        for author in author2Attributes:
            dictKey = str(author2Attributes[author]['structuralId']) + '_' + str(timeStep)

            if dictKey not in timeStepDict:
                timeStepDict[dictKey] = author2Attributes[author]['textualIds']
            else:
                timeStepDict[dictKey].extend(author2Attributes[author]['textualIds'])

        for dictKey in timeStepDict:
            timeStepDict[dictKey] = [collectionCentroids[clusterIdKMeans] for clusterIdKMeans in timeStepDict[dictKey]]
            timeStepDict[dictKey] = tuple(centeroidnp(np.array(timeStepDict[dictKey])))

        return timeStepDict

    def centeroidnp(arr):
        length, dim = arr.shape
        return np.array([np.sum(arr[:, i])/length for i in range(dim)])

    '''
    frontsEvents = {1: {}, 2: []}
    '''
    def updateFronts(fronts, frontEvents, frontId2CommunityId): 
        
        # remove things which should be removed if necessary

        indicesToRemove = frontEvents[1].keys()

        if (len(indicesToRemove) > 0):

            # !!! frontId2CommunityId needs to be updated; handle replacements and deletions

            # sort the indices to remove
            indicesToRemove = sorted(indicesToRemove)

            oldIdxToNewIdx = dict(zip(range(indicesToRemove[0]), range(indicesToRemove[0])))

            idIdx = 0
            step = 1

            while idIdx < (len(indicesToRemove) - 1):

                currentIdxToRemove = indicesToRemove[idIdx]
                nextIdxToRemove = indicesToRemove[idIdx + 1]
                
                oldIdxToNewIdx[currentIdxToRemove] = -1
                oldIdxToNewIdx[nextIdxToRemove] = -1

                k = currentIdxToRemove + 1

                while k < nextIdxToRemove:
                    oldIdxToNewIdx[k] = k - step
                    k += 1

                idIdx += 1
                step += 1

            # for the rest of the indices just decrement
            for idx in range(idIdx, len(frontId2CommunityId.keys())):
                oldIdxToNewIdx[idx] = idx - step

            newFrontId2CommunityId = {}

            for frontId in frontId2CommunityId:

                if (oldIdxToNewIdx[frontId] != -1):
                    newFrontId2CommunityId[oldIdxToNewIdx[frontId]] = frontId2CommunityId[frontId]

            frontId2CommunityId = newFrontId2CommunityId

            # !!! remove the fronts with specific indices; take care, this changes the fronts list indices
            fronts = [fronts[frontId] for frontId in range(len(fronts)) if frontId not in indicesToRemove]

            # add replacements
            for frontId in frontEvents[1]:
                for frontMergeEvent in frontEvents[1][frontId]:
                    eventKey = frontMergeEvent[0]
                    staticCommunityCentroids = frontMergeEvent[1]
                    newFront = staticCommunityCentroids
                    if newFront not in fronts:
                        fronts.append(newFront)
                        frontId2CommunityId[len(fronts)-1] = eventKey
                
        for frontCreateEvent in frontEvents[2]:

            eventKey = frontCreateEvent[0]
            staticCommunityCentroids = frontCreateEvent[1]
            newFront = staticCommunityCentroids

            if newFront not in fronts:
                fronts.append(newFront)
                frontId2CommunityId[len(fronts)-1] = eventKey

        return (frontId2CommunityId, fronts)

    allSnapshots = getAllSnapshots('quarter')

    snapshotCommunities0 = getCommunitiesForSnapshot(allSnapshots[0], 0)

    print('snapshot comms ====', snapshotCommunities0)

    '''
    communitiesTimestepMapping[communityId_0_1] = [communityId_1_0, communityId_1_1, ...] 
    '''
    # the initial communities are the initial fronts
    communitiesTimestepMapping = dict(zip(snapshotCommunities0.keys(), [[] for i in range(len(snapshotCommunities0))]))

    '''
    fronts = list of fronts; 
    a front = a dict with a list of authors 
        and 
            a list of tuples of tuples: ( (centroid1_1, centroid1_2), (centroid2_1), (centroid3_1, centroid3_2, centroid3_3) ) and centroidx_y = (i1, i2, ..., i24)
    '''
    fronts = []

    for staticCommunity0 in snapshotCommunities0:
        front = snapshotCommunities0[staticCommunity0]
        fronts.append(front)
    
    '''
        maps each front with its associated community at the associated specific timestep
    '''
    frontId2CommunityId = dict(zip(range(len(fronts)), [communityId for communityId in snapshotCommunities0.keys()]))

    centroidTuples2Distances = {}

    for timeStep in range(1, len(allSnapshots)):

        # print('timeStep', timeStep)

        snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep)

        '''
        frontsEvents[frontEvent][frontId] = [front1, front2, ...]
        1 = front x was replaced by fronts list
        2 = a new front must be added
        '''
        frontEvents = {1: {}, 2: []}

        # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
        for communityIdA in snapshotCommunities:

            centroidsTupleA = snapshotCommunities[communityIdA] # (centroid_1, centroid_2, ..., centroid_n)        
            
            bestFrontIds = []

            for frontId in range(len(fronts)):
                
                centroidsTupleB = fronts[frontId] # (centroid_1, centroid_2, ..., centroid_n) - a front is actually a static community

                # determine textual sim
                cosineSimilarities = []

                if ((centroidsTupleA, centroidsTupleB) not in centroidTuples2Distances) and ((centroidsTupleB, centroidsTupleA) not in centroidTuples2Distances):
                    for ca in centroidsTupleA:
                        for cb in centroidsTupleB:
                            cosineSimilarity = dot(ca, cb)/(norm(ca)*norm(cb))
                            cosineSimilarities.append(cosineSimilarity)

                    minSimilarity = min(cosineSimilarities)
                    centroidTuples2Distances[(centroidsTupleA, centroidsTupleB)] = minSimilarity

                elif (centroidsTupleA, centroidsTupleB) in centroidTuples2Distances:
                    minSimilarity = centroidTuples2Distances[(centroidsTupleA, centroidsTupleB)]
                    
                elif (centroidsTupleB, centroidsTupleA) in centroidTuples2Distances:
                    minSimilarity = centroidTuples2Distances[(centroidsTupleB, centroidsTupleA)]
                
                if (minSimilarity > optimalSim):
                    # print('SIM IS BIGGER', avgSimilarity)
                    bestFrontIds.append(frontId)

            # print('BEST FRONTS', bestFrontIds)
            if (len(bestFrontIds) > 0):
                for bestFrontId in bestFrontIds:
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
        
    with open(outputFileName, 'w') as outfile:
        json.dump(finalMappings, outfile)

parser = argparse.ArgumentParser()

parser.add_argument('-db', '--db', type=str, help='The database to read from')
parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities')
parser.add_argument('-o', '--o', type=str, help='The json output file')

args = parser.parse_args()

dbName = args.db
optimalSim = args.sim
outputFileName = args.o

doComputation(dbName, optimalSim, outputFileName)