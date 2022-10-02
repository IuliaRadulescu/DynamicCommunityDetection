import plotly.graph_objects as go
import random
import os
import json

# diagrams made with https://plotly.com/python/sankey-diagram/

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
    value.extend([1]*len(items))
  
  return (label, source, target, value)

def feedSankeyJsonClassic(alluvialData, outputFileName):

  (label, source, target, value) = parseAlluvailData(alluvialData)

  generateRandomColor = lambda: 'rgba(' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ', 0.8)'

  def generateRandomColorList(listLen):
    return [generateRandomColor() for _ in range(listLen)]

  # color = generateRandomColorList(len(source))
  color = ['rgb(255,255,255)'] * (len(source) + 1)

  data = [dict(
            type = 'sankey',
            arrangement = 'freeform',
            node = dict(
                label = label,
                pad = 10,
                color = color
            ),
            link = dict(
                source = source,
                target = target,
                value = value
            )
          )
        ]

  with open(os.path.dirname(os.path.realpath(__file__)) + '/alluvailJsons/' + outputFileName, 'w') as outfile:
    json.dump(data, outfile)

def feedSankeyJsonHybrid(alluvialData, outputFileName):

  (label, source, target, value) = parseAlluvailData(alluvialData)

  generateRandomColor = lambda: 'rgba(' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ',' + str(random.randint(0,255)) + ', 0.8)'

  def generateRandomColorList(listLen):
    return [generateRandomColor() for _ in range(listLen)]

  # color = generateRandomColorList(len(source))
  color = ['rgb(255,255,255)'] * (len(source) + 1)

  data = [dict(
            type = 'sankey',
            arrangement = 'snap',
            node = dict(
                label = label,
                pad = 20,
                color = color
            ),
            link = dict(
                source = source,
                target = target,
                value = value
            )
          )
        ]

  with open(os.path.dirname(os.path.realpath(__file__)) + '/alluvailJsons/' + outputFileName, 'w') as outfile:
    json.dump(data, outfile)