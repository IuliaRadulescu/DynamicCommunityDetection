import pymongo
import numpy as np

'''
Get communities which merged, from time step matrix timeA to time step matrix timeB
timeA = time matrix for time step A (numpy array)
timeB = time matrix for time step B (numpy array)
'''
def getMerges(timeA, timeB, k):

    communitiesWhichMerged = []
    
    for clusterIdA1 in range(len(timeA)):
        for clusterIdA2 in range(len(timeA)):
            for clusterIdB in range(len(timeB)):

                if (clusterIdA2 <= clusterIdA1):
                    continue

                lineA1 = timeA[clusterIdA1, :]
                lineA2 = timeA[clusterIdA2, :]
                lineB = timeB[clusterIdB, :]

                termA1B = np.linalg.norm(np.logical_and(lineA1, lineB), ord = 1)
                termA2B = np.linalg.norm(np.logical_and(lineA2, lineB), ord = 1)
                
                # if (termA1B > 0 or termA2B > 0):
                #     print(termA1B, np.linalg.norm(lineA1, ord = 1)/2, termA2B, np.linalg.norm(lineA2, ord = 1)/2)

                if ((termA1B < (np.linalg.norm(lineA1, ord = 1)/2)) or (termA2B < (np.linalg.norm(lineA2, ord = 1)/2))):
                    continue

                frac1 = np.linalg.norm(np.logical_and(np.logical_or(lineA1, lineA2), lineB), ord = 1)

                termA1A2 = np.linalg.norm(np.logical_or(lineA1, lineA2), ord = 1)
                termB = np.linalg.norm(lineB, ord = 1)
                
                frac2 = max(termA1A2, termB)

                if (frac1 >= k * frac2):
                    communitiesWhichMerged.append((clusterIdA1, clusterIdA2, clusterIdB))

    return communitiesWhichMerged

def getJoins(timeA, timeB):

    joins = []

    for clusterIdA in range(len(timeA)):
        for clusterIdB in range(len(timeB)):

            lineA = timeA[clusterIdA, :]
            lineB = timeB[clusterIdB, :]

            if np.linalg.norm(np.logical_and(lineA, lineB), ord = 1) <= (np.linalg.norm(lineA, ord = 1) / 2):
                continue

            for vertexId in range(len(lineB)):
                if (lineB[vertexId] == 1 and lineA[vertexId] == 0):
                    joins.append((vertexId, clusterIdB))

    print(joins)
    print('We have ', len(joins), ' joins between snapshots')


dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

authors2Ids = {}
authors2Clusters = {}

timeAData = list(db['oneHour_7_19_7_20'].find())
timeBData = list(db['oneHour_7_20_7_21'].find())

# use a generic name for comments with no authors
genericAuthorName = 'JhonDoe25122020'

authorsA = list(set([x['author'] for x in timeAData]))
authorsB = list(set([x['author'] for x in timeBData]))

allAuthors = list(set(authorsA + authorsB))

authors2Ids = dict(zip(allAuthors, range(0, len(allAuthors))))
authors2Clusters = dict(zip([x['author'] for x in timeAData + timeBData], [x['authorClusterId'] for x in timeAData + timeBData]))

noLinesA = len(set([x['authorClusterId'] for x in timeAData]))
noLinesB = len(set([x['authorClusterId'] for x in timeBData]))
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


# print(getMerges(timeA, timeB, 0.1))
getJoins(timeA, timeB)