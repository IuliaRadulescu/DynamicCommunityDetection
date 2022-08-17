import json
import os

def getAlluvialDataHybrid(dataset='tennis', similarity=70):

    if (dataset == 'tennis'):
        if (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim70.json',)
        elif (similarity == 75):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim75.json',)
        elif (similarity == 80):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim80.json',)
        elif (similarity == 85):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim85.json',)
        elif (similarity == 90):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim90.json',)
        elif (similarity == 95):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/hybridDynamicSim95.json',)

    return json.load(jsonFile)

def getAlluvialDataClassic(dataset='tennis', similarity=70):

    if (dataset == 'tennis'):
        if (similarity == 70):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim70.json',)
        elif (similarity == 75):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim75.json',)
        elif (similarity == 80):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim80.json',)
        elif (similarity == 85):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim85.json',)
        elif (similarity == 90):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim90.json',)
        elif (similarity == 95):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/RAW_OUTPUTS/classicDynamicSim95.json',)

    return json.load(jsonFile)