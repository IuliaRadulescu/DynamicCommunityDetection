import plotly.graph_objects as go
import alluvialDataRetriever
import random
import os

def parseAlluvailData(alluvialData):

  label = []
  source = []
  target = []
  value = []

  for key in alluvialData:
    items = alluvialData[key]
    if key not in label:
        label.append(key)
    for item in items:
        if item not in label:
            label.append(item)

  for key in alluvialData:
    items = alluvialData[key]
    source.extend([label.index(key)]*len(items))
    itemsIndices = list(map(lambda x: label.index(x), items))
    target.extend(itemsIndices)
    value.extend([10]*len(items))
  
  return (label, source, target, value)

def determineNodesPositions(labels):

    x = []
    y = []

    xStep = round(1/(len(labels)), 2)
    yStep = round(1/(len(labels)), 2)
    stepSize = round(1/(len(labels)), 2)

    xLabelToStep = {}
    yLabelToStep = {}

    communityIds = []
    timeIntervalIds = []

    for label in labels:
        labelParts = label.split('_')
        timeIntervalIds.append(int(labelParts[1])) # x
        communityIds.append(int(labelParts[0])) # y

    communityIds.sort()
    timeIntervalIds.sort()
    
    for timeIntervalId in timeIntervalIds:
        if (timeIntervalId not in yLabelToStep):
            xLabelToStep[timeIntervalId] = round(xStep, 2)
            xStep += stepSize

    for communityId in communityIds:
        if (communityId not in yLabelToStep):
            yLabelToStep[communityId] = round(yStep, 2)
            yStep += stepSize
        
    x = [xLabelToStep[int(label.split('_')[1])] for label in labels]
    y = [yLabelToStep[int(label.split('_')[0])] for label in labels]

    return (x, y)

def plotAlluvialImage(alluvialData, imageTitle):

  (label, source, target, value) = parseAlluvailData(alluvialData)

  (x, y) = determineNodesPositions(label)

  generateRandomColor = lambda: 'rgba(' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ', 0.8)'

  def generateRandomColorList(listLen):
    return [generateRandomColor() for _ in range(listLen)]

  color = generateRandomColorList(len(source))

  fig = go.Figure(
      data=[go.Sankey(
        arrangement='snap',
        node = dict(
            pad = 15,
            thickness = 20,
            label = label,
            color = color,
            x = x,
            y = y
        ),
        link = dict(
          source = source, # indices correspond to labels
          target = target,
          value = value,
          color = color
        )
      )]
    )

  fig.show()

  # fig.write_image(os.path.dirname(os.path.realpath(__file__)) + '/images/' + imageTitle)