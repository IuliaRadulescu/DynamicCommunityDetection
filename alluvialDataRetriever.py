import json
import os

def getAlluvialDataHybridText(dataset='biden', similarity=50):

    if (dataset == 'biden'):
        if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY05CommunitiesText.json',)
        elif (similarity == 60):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY06CommunitiesText.json',)
        elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY07CommunitiesText.json',)
        elif (similarity == 80):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY08CommunitiesText.json',)
        elif (similarity == 85):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY085CommunitiesText.json',)
        elif (similarity == 90):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY09CommunitiesText.json',)
        elif (similarity == 95):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/SIMILARITY095CommunitiesText.json',)
    elif (dataset == 'protests'):
        if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY05CommunitiesText.json',)
        elif (similarity == 60):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY06CommunitiesText.json',)
        elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY07CommunitiesText.json',)
        elif (similarity == 80):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY08CommunitiesText.json',)
        elif (similarity == 85):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY085CommunitiesText.json',)
        elif (similarity == 90):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY09CommunitiesText.json',)
        elif (similarity == 95):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/SIMILARITY095CommunitiesText.json',)

    return json.load(jsonFile)

def getAlluvialDataClassic(dataset='biden', similarity=50):

    if (dataset == 'biden'):
        if (similarity == 30):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY03ClassicCommunities.json',)
        elif (similarity == 50):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY05ClassicCommunities.json',)
        elif (similarity == 70):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY07ClassicCommunities.json',)
        elif (similarity == 80):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY08ClassicCommunities.json',)
        elif (similarity == 85):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY085ClassicCommunities.json',)
        elif (similarity == 90):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY09ClassicCommunities.json',)
    elif (dataset == 'protests'):
        if (similarity == 30):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY03ClassicCommunities.json',)
        elif (similarity == 50):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY05ClassicCommunities.json',)
        elif (similarity == 70):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY07ClassicCommunities.json',)
        elif (similarity == 80):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY08ClassicCommunities.json',)
        elif (similarity == 85):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY085ClassicCommunities.json',)
        elif (similarity == 90):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY09ClassicCommunities.json',)

    return json.load(jsonFile)