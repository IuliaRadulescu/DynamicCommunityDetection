import pymongo
import math
from datetime import datetime
from random_object_id import generate

def createTimeIntervals(startTimestamp, endTimestamp, intervalInSeconds):

    timeIntervals = []

    current = startTimestamp

    while (current < endTimestamp):
        
        nextTimestamp = current + intervalInSeconds

        timeIntervals.append(str(current) + '_' + str(nextTimestamp))

        current = nextTimestamp

    return timeIntervals

def convertTimestampIntervalToReadableInterval(timestampInterval):

    timestamps = timestampInterval.split('_')

    datetimeObj1 = datetime.fromtimestamp(int(timestamps[0]))
    datetimeObj2 = datetime.fromtimestamp(int(timestamps[1]))

    day1Padded0 = str(datetimeObj1.day) if int(datetimeObj1.day) > 9 else str(datetimeObj1.day).zfill(2)
    day2Padded0 = str(datetimeObj2.day) if int(datetimeObj2.day) > 9 else str(datetimeObj2.day).zfill(2)

    humanReadableInterval  = 'quarter_' \
        + day1Padded0 + '_' + str(datetimeObj1.strftime('%H')) + '_' + str(datetimeObj1.strftime('%M')) + '_' \
        + day2Padded0 + '_' + str(datetimeObj2.strftime('%H')) + '_' + str(datetimeObj2.strftime('%M'))

    return humanReadableInterval        
 
intervalInSeconds = 60 * 15
startTimestamp = 1604534436
endTimestamp = 1608158481

# 2 hours intervals
timeIntervals = createTimeIntervals(startTimestamp, endTimestamp, intervalInSeconds)

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionUSAElections

allComments = list(db.comments.find()) + list(db.submissions.find())

# use a generic name for comments with no authors
genericAuthorName = 'JhonDoe25122020'

comments2RedditIds = {}
interactionsDict = {}
toInsert = {}

for comment in allComments:

    if (comment['author'] == False):
        comment['author'] = genericAuthorName

    if (comment['redditId'] not in comments2RedditIds):
        comments2RedditIds[comment['redditId']] = []
    
    comments2RedditIds[comment['redditId']] = comment

'''
Get each author's intractions with other authors and generate a timestamp dependent key.
Add the direct parent only if it is in the interval
Do this recursively until there are no parents left
'''

for comment in allComments:

    if (comment['author'] == False):
        comment['author'] = genericAuthorName

    # skip parents, will be added anyway
    if ('parentAuthorId' not in comment or comment['parentAuthorId'] == False):
        continue

    interactionId = comment['author'] + '|*|' + str(int(comment['created']))

    if (interactionId not in interactionsDict):
        interactionsDict[interactionId] = []

    interactionsDict[interactionId] += [comment]

    # check if parent should also be added
    parentComment = comments2RedditIds[comment['parentRedditId'].split('_')[1]]

    commentInterval = round(int((int(comment['created'] - startTimestamp) / intervalInSeconds)))
    parentInterval = round(int((int(parentComment['created'] - startTimestamp) / intervalInSeconds)))

    if (commentInterval == parentInterval):
        interactionsDict[interactionId] += [parentComment]

del comments2RedditIds
del allComments

print(len(interactionsDict))

for interactionKey in interactionsDict:
    
    createdTimestamp = interactionKey.split('|*|')[1]

    associatedInterval = round(int((int(createdTimestamp) - startTimestamp) / intervalInSeconds))

    dbName = convertTimestampIntervalToReadableInterval(timeIntervals[associatedInterval])

    if (dbName not in toInsert):
        toInsert[dbName] = []

    toInsert[dbName] += interactionsDict[interactionKey]

del interactionsDict

print(len(toInsert))

for dbName in toInsert:
    print('Insert into ', dbName)
    noDuplicates = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in toInsert[dbName])]
    db[dbName].insert_many(noDuplicates)