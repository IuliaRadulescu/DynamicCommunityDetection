
import plotly.graph_objects as go
import alluvialData

alluvialData = alluvialData.getAlluvialData()

def filterAlluvialData(alluvialData, communitiesToMonitor):

  filteredAlluvial = {}

  for community in communitiesToMonitor:

    if community not in alluvialData:
      continue

    filteredAlluvial[community] = alluvialData[community]
    communitiesHeap = []
    communitiesHeap.extend(alluvialData[community])

    while len(communitiesHeap) > 0:

      communityInHeap = communitiesHeap.pop()
      
      if communityInHeap not in alluvialData:
        continue

      filteredAlluvial[communityInHeap] = alluvialData[communityInHeap]
      communitiesHeap.extend(alluvialData[communityInHeap])

  return filteredAlluvial

def parseAlluvailData(alluvialData):
    label = []
    source = []
    target = []
    value = []

    for key in alluvialData:
        items = alluvialData[key]
        label.append(key)
        label.extend(items)

    label = list(set(label))

    for key in alluvialData:
        items = alluvialData[key]
        source.extend([label.index(key)]*len(items))
        itemsIndices = list(map(lambda x: label.index(x), items))
        target.extend(itemsIndices)
        value.extend([int(100/len(items))]*len(items))
    
    return (label, source, target, value)

filteredAlluvialData = filterAlluvialData(alluvialData, list(alluvialData.keys())[0:20])

(label, source, target, value) = parseAlluvailData(filteredAlluvialData)

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = label,
      color = "blue"
    ),
    link = dict(
      source = source, # indices correspond to labels
      target = target,
      value = value
  ))])

fig.update_layout(title_text="Community evolution", font_size=10)
fig.show()
