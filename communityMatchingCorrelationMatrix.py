import pymongo
import numpy as np

def getEvents(timeAData, timeBData):

    allData = timeAData + timeBData

    authorsA = list(set([x['author'] for x in timeAData]))
    authorsB = list(set([x['author'] for x in timeBData]))

    allAuthors = list(set(authorsA + authorsB))

    authors2Ids = dict(zip(allAuthors, range(0, len(allAuthors))))
    authors2Clusters = dict(zip([x['author'] for x in allData], [x['clusterIdSmooth'] for x in allData]))

    noLinesA = len(set([x['clusterIdSmooth'] for x in timeAData]))
    noLinesB = len(set([x['clusterIdSmooth'] for x in timeBData]))
    noCols = len(allAuthors)

    timeA = np.zeros((noLinesA, noCols), dtype=int)
    timeB = np.zeros((noLinesB, noCols), dtype=int)

    for author in authors2Ids:
        authorId = authors2Ids[author]
        clusterId = authors2Clusters[author]

        if (clusterId < len(timeA)):
            timeA[clusterId][authorId] = 1

        if (clusterId < len(timeB)):
            timeB[clusterId][authorId] = 1


    k1 = np.zeros((len(timeA), len(timeB)), dtype=int)
    k2 = np.zeros((len(timeA), len(timeB)), dtype=int)

    line = 0

    for clusterIdA in range(len(timeA)):
        column = 0
        for clusterIdB in range(len(timeB)):
            
            normA = np.linalg.norm(timeA[clusterIdA, :], ord = 1)
            normB = np.linalg.norm(timeB[clusterIdB, :], ord = 1)

            k1[line][column] = np.dot(timeA[clusterIdA, :], timeB[clusterIdB, :])/normA if normA != 0 else 0
            k2[line][column] = np.dot(timeA[clusterIdA, :], timeB[clusterIdB, :])/normB if normB != 0 else 0

            column += 1
        line += 1

    # process k1 and k2

    theta = 0.3
    k1[k1 <= theta] = 0
    k2[k2 <= theta] = 0

    for line in range(len(k1)):
        lineMax = np.max(k1[line])
        helper = k1[line]
        helper[helper < lineMax] = 0

    for col in range(np.shape(k2)[1]):
        colMax = np.max(k2[:, col])
        helper = k1[:, col]
        helper[helper < colMax] = 0

    return getMergesAndGrowths(k1, theta)

    # return getSplitsAndContractions(k2, theta)


def getSplitsAndContractions(k2, theta):

    splits = []
    contractions = []

    for line in range(len(k2)):
        postives = np.argwhere(k2[line] > 0)
        if (len(postives) == 1):
            contractions.append((line, postives[0]))

        if (len(postives) > 1):
            splits.append((line, postives))

    if (len(splits) > 0 or len(contractions) > 0):
        print('We have ', len(splits), ' splits and ', len(contractions), ' contractions')

    return (splits, contractions)


def getMergesAndGrowths(k1, theta):

    def printStats():

        growthsPerCommunities = {}

        for growthPair in growths:

            if (growthPair[1] not in growthsPerCommunities):
                growthsPerCommunities[growthPair[1]] = []
            
            growthsPerCommunities[growthPair[1]].append(growthPair[0])

        for growthCommunity in growthsPerCommunities:
            print('Community ', growthCommunity, ' received ', len(growthsPerCommunities[growthCommunity]), 'nodes')

    growths = []
    merges = []

    for line in range(len(k1)):
        for col in range(len(k1[line])):
            postives = np.argwhere(k1[:, col] > 0)
            if len(postives == 1):
                growths.append((postives[0], col))
            
            if len(postives) > 1:
                merges.append((postives, col))
                    
    if (len(merges) > 0 or len(growths) > 0):
        print('We have ', len(merges), ' merges and ', len(growths), ' growths')

    return (growths, merges)


dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

allCollections = db.list_collection_names()

prefix = 'quarter'
allCollections = list(filter(lambda x: prefix in x, allCollections))

allCollections = sorted(allCollections)

authors2Ids = {}
authors2Clusters = {}

finalDataG = []
finalDataM = []

for collectionId in range(len(allCollections) - 1):

    timeAData = list(db[allCollections[collectionId]].find())
    timeBData = list(db[allCollections[collectionId + 1]].find())

    print(allCollections[collectionId], ' - ', allCollections[collectionId + 1])

    (splits, contractions) = getEvents(timeAData, timeBData)

    finalDataG.append((allCollections[collectionId], allCollections[collectionId + 1], len(splits)))
    finalDataM.append((allCollections[collectionId], allCollections[collectionId + 1], len(contractions)))

print('FINAL_G!')

print(finalDataG)

print('FINAL_M!')

print(finalDataM)