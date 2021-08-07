import pymongo
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
import json

def doComputation(dbName, alpha, optimalSim, outputFileName):

    '''
    constants
    '''
    TYPE_STRUCTURAL = 1
    TYPE_TEXTUAL = 2

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
                timeStepDict[dictKey] = {TYPE_STRUCTURAL: [], TYPE_TEXTUAL: []}
                timeStepDict[dictKey][TYPE_STRUCTURAL] = [author]
                timeStepDict[dictKey][TYPE_TEXTUAL] = author2Attributes[author]['textualIds']
            else:
                timeStepDict[dictKey][TYPE_STRUCTURAL].append(author)
                timeStepDict[dictKey][TYPE_TEXTUAL].extend(author2Attributes[author]['textualIds'])

        # print('TIME STEP DICT', timeStepDict)

        for dictKey in timeStepDict:
            timeStepDict[dictKey][TYPE_STRUCTURAL] = list(set(timeStepDict[dictKey][TYPE_STRUCTURAL]))
            timeStepDict[dictKey][TYPE_TEXTUAL] = list(set(timeStepDict[dictKey][TYPE_TEXTUAL]))
            timeStepDict[dictKey][TYPE_TEXTUAL] = tuple([collectionCentroids[clusterIdKMeans] for clusterIdKMeans in timeStepDict[dictKey][TYPE_TEXTUAL]])

        return timeStepDict

    '''
    frontsEvents = {1: {}, 2: []}
    '''
    def updateFronts(fronts, frontEvents, frontId2CommunityId):
        
        # remove things which should be removed
        fronts = [fronts[frontId] for frontId in range(1, len(fronts)) if frontId not in frontEvents[1]]
        
        # add replacements
        for frontId in frontEvents[1]:
            for frontMergeEvent in frontEvents[1][frontId]:
                eventKey = frontMergeEvent[0]

                staticCommunityAuthors = frontMergeEvent[1][TYPE_STRUCTURAL]
                staticCommunityCentroids = frontMergeEvent[1][TYPE_TEXTUAL]

                newFront = {TYPE_STRUCTURAL: staticCommunityAuthors, TYPE_TEXTUAL: staticCommunityCentroids}
                
                if newFront not in fronts:
                    fronts.append(newFront)
                    frontId2CommunityId[len(fronts)-1] = eventKey
                
        for frontCreateEvent in frontEvents[2]:

            eventKey = frontCreateEvent[0]

            staticCommunityAuthors = frontCreateEvent[1][TYPE_STRUCTURAL]
            staticCommunityCentroids = frontCreateEvent[1][TYPE_TEXTUAL]

            newFront = {TYPE_STRUCTURAL: staticCommunityAuthors, TYPE_TEXTUAL: staticCommunityCentroids}

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
        front = {TYPE_STRUCTURAL: [], TYPE_TEXTUAL: []}

        front[TYPE_STRUCTURAL] = snapshotCommunities0[staticCommunity0][TYPE_STRUCTURAL]
        front[TYPE_TEXTUAL] = snapshotCommunities0[staticCommunity0][TYPE_TEXTUAL]

        fronts.append(front)
    
    '''
        maps each front with its associated community at the associated specific timestep
    '''
    frontId2CommunityId = dict(zip(range(len(fronts)), [communityId for communityId in snapshotCommunities0.keys()]))

    centroidTuples2Distances = {}

    for timeStep in range(1, len(allSnapshots)):

        snapshotCommunities = getCommunitiesForSnapshot(allSnapshots[timeStep], timeStep)

        '''
        frontsEvents[frontEvent][frontId] = [front1, front2, ...]
        1 = front x was replaced by fronts list
        2 = a new front must be added
        '''
        frontEvents = {1: {}, 2: []}

        # map communities from dynamicCommunities list (t-1) to the ones in snapshot (t)
        for communityIdA in snapshotCommunities:

            authorsA = snapshotCommunities[communityIdA][TYPE_STRUCTURAL]
            centroidsTupleA = snapshotCommunities[communityIdA][TYPE_TEXTUAL] # (centroid_1, centroid_2, ..., centroid_n)        
            
            bestFrontIds = []

            for frontId in range(len(fronts)):
                
                authorsB = list(set(fronts[frontId][TYPE_STRUCTURAL]))
                centroidsTupleB = fronts[frontId][TYPE_TEXTUAL] # (centroid_1, centroid_2, ..., centroid_n) - a front is actually a static community
                
                # print('FRONT', frontId)
                # print('==============')
                # print('CENTROIDSB', cb)getCommunitiesForSnapshot
                # determine structural sim
                intersect = len(list(set(authorsA) & set(authorsB)))
                reunion = len(list(set(authorsA + authorsB)))

                jaccard = intersect/reunion

                if (jaccard == 1):
                    print('authorsA', authorsA)
                    print('authorsB', authorsB)

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

                if (minSimilarity < 0):
                    minSimilarity = 0

                fullSimilarity = alpha * jaccard + (1-alpha) * minSimilarity

                if (fullSimilarity > 1):
                    print('Full similarity is greater than 1')
                
                if (fullSimilarity > optimalSim):
                    print('SIM IS BIGGER', fullSimilarity, jaccard, minSimilarity, alpha, ((1-alpha) * minSimilarity))
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
parser.add_argument('-alpha', '--alpha', type=float, help='Structural vs. textual weight')
parser.add_argument('-sim', '--sim', type=float, help='The minimum similarity to match communities')
parser.add_argument('-o', '--o', type=str, help='The json output file')

args = parser.parse_args()

dbName = args.db
alpha = args.alpha
optimalSim = args.sim
outputFileName = args.o

doComputation(dbName, alpha, optimalSim, outputFileName)