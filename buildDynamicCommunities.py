import pymongo
import numpy as np
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

        print('Finished reading all snapshots from mongo!')

        return sorted(allCollections)

    '''
    @returns: dictionary, where keys are strings of the form: communityId_timeStep
                and the values are lists of strings, where a list contains the author names in that community at the given timestep
    '''
    def getCommunitiesForSnapshot(collectionName, timeStep, communityAttribute):

        dbClient = pymongo.MongoClient('localhost', 27017)
        db = dbClient[dbName]

        allComments = list(db[collectionName].find())

        dbClient.close()

        print('Finished reading comments from mongo!', collectionName)

        author2Attributes = {}

        for comment in allComments:
            if comment['author'] not in author2Attributes:
                author2Attributes[comment['author']] = {
                    'structuralId': comment['clusterIdSimple']
                }
        
        timeStepDict = {}

        for author in author2Attributes:
            dictKey = str(author2Attributes[author]['structuralId']) + '_' + str(timeStep)

            if dictKey not in timeStepDict:
                timeStepDict[dictKey] = [author]
            else:
                timeStepDict[dictKey].append(author)

        # print('TIME STEP DICT', timeStepDict)

        for dictKey in timeStepDict:
            timeStepDict[dictKey] = list(set(timeStepDict[dictKey]))

        return timeStepDict

    '''
    frontsEvents = {1: {}, 2: []}
    '''
    def updateFronts(fronts, frontEvents, frontId2CommunityId):

        # remove things which should be removed
        fronts = [fronts[frontId] for frontId in range(1, len(fronts)) if frontId not in frontEvents[1]]

        # add replacements
        for frontId in frontEvents[1]:
            for item in frontEvents[1][frontId]:
                fronts += [item[1]]
                frontId2CommunityId[len(fronts)-1] = item[0]
        
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
            
            bestFrontIds = []

            for frontId in range(len(fronts)):

                authorsB = list(set(fronts[frontId]))
                
                intersect = len(list(set(authorsA) & set(authorsB)))
                reunion = len(list(set(authorsA + authorsB)))

                jaccard = intersect/reunion

                # if (jaccard > 0):
                #     print('jaccard', jaccard)

                if (jaccard > optimalSim):
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

        # update mappings
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