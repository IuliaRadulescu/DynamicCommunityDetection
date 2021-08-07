import json
import os

def getAlluvialDataHybrid(dataset='biden', similarity=50, alpha=60):

    if (dataset == 'biden'):
        if (alpha == 60):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA60/SIMILARITY05CommunitiesText.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA60/SIMILARITY07CommunitiesHybrid.json',)
        elif (alpha == 50):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA50/SIMILARITY05CommunitiesHybrid.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA50/SIMILARITY07CommunitiesHybrid.json',)
        elif (alpha == 40):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA40/SIMILARITY05CommunitiesHybrid.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_HYBRID/ALPHA40/SIMILARITY07CommunitiesHybrid.json',)
    elif (dataset == 'protests'):
        if (alpha == 60):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA60/SIMILARITY05CommunitiesHybrid.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA60/SIMILARITY07CommunitiesHybrid.json',)
        elif (alpha == 50):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA50/SIMILARITY05CommunitiesHybrid.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA50/SIMILARITY07CommunitiesHybrid.json',)
        elif (alpha == 40):
            if (similarity == 50):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA40/SIMILARITY05CommunitiesHybrid.json',)
            elif (similarity == 70):
                jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_HYBRID/ALPHA40/SIMILARITY07CommunitiesHybrid.json',)

    return json.load(jsonFile)

def getAlluvialDataClassic(dataset='biden', similarity=50):

    if (dataset == 'biden'):
        if (similarity == 50):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY05ClassicCommunities.json',)
        elif (similarity == 70):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/BIDEN_CLASSIC/SIMILARITY07ClassicCommunities.json',)
    elif (dataset == 'protests'):
        if (similarity == 50):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY05ClassicCommunities.json',)
        elif (similarity == 70):
            jsonFile = open(os.path.dirname(os.path.realpath(__file__)) + '/PROTESTS_CLASSIC/SIMILARITY07ClassicCommunities.json',)

    return json.load(jsonFile)