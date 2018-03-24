import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getNofNull(df,attribute):
    return pd.isnull(df[attribute]).sum()

def getPercentNull(df,attribute):
    return getNofNull(df,attribute)/len(df)

def getPercentValues(df,attribute):
    return df.groupby(attribute).size().apply(lambda x: float(x) / df.groupby(attribute).size().sum()*100).sort_values(ascending=False)

def getMissingValues(df):
    miss = {}
    #mi stampa tutti gli attributi del @dataframe in input aventi missing values
    #con il relativo numero di missing values
    for a in df.columns:
        miss[a] = getNofNull(df,a)
    return sorted(miss.items(), key=lambda x:x[1], reverse=True)

def getMostFrequent(series):
    # restituisce il valore più frequente per quella series
    try:
        return series.value_counts().idxmax()
    except ValueError:
            return np.nan


def getGroupedDescription(attributeToDescribe, attributeToGroupBy,df):
    # restituisce la descrizione di @attributeToDescribe, raggruppati per i diversi valori
    # di @attributeToGroupBy
    return df[attributeToDescribe].groupby(df[attributeToGroupBy]).describe()


def getSingleAttributeStats(attribute, df, toOthers=False, kindRepresentation='bar', **kwargs):
    from preprocessing_utils import fillToOthers

    tmpDf = df.copy()
    if toOthers:
        try:
            topN = kwargs['topN']
            tmpDf = fillToOthers(attribute,tmpDf, topN=topN)
        except KeyError:
            try:
                valueToCut = kwargs['valueToCut']
                tmpDf = fillToOthers(attribute,tmpDf,valueToCut=valueToCut)
            except KeyError:
                tmpDf = fillToOthers(attribute,tmpDf)

    # DA UTILIZZARE PER ATTRIBUTI NON CONTINUI!
    print (tmpDf[attribute].describe())
    print ("Distribuzione valori di " + attribute +": ")
    print (tmpDf[attribute].value_counts())

    dfToPercent = tmpDf.groupby(attribute).size().apply\
    (lambda x: float(x) / tmpDf.groupby(attribute).size().sum())
    print ("Valori di " + attribute + " in percentuale: ")
    print (dfToPercent)

    # Set up a grid of plots
    fig = plt.figure(figsize=(10, 10))
    fig_dims = (2, 2)

    plt.subplot2grid(fig_dims, (0, 0))
    tmpDf[attribute].value_counts().plot(kind=kindRepresentation,\
                                                 title=attribute + ' values distribution')

    plt.subplot2grid(fig_dims, (0, 1))
    dfToPercent.plot(kind = kindRepresentation, title=attribute + ' values percentage')

    plt.subplot2grid(fig_dims, (1, 0))
    tmpDf[attribute].value_counts().plot(kind='pie', autopct='%1.1f%%')


def getSingleContinueAttributeStats(attribute,df):
    # DA UTILIZZARE PER ATTRIBUTI CONTINUI!
    print (df[attribute].describe())

    fig = plt.figure(figsize=(10, 4))
    fig_dims = (1, 2)
    plt.subplot2grid(fig_dims, (0, 0))
    df[attribute].plot(kind='hist')
    plt.subplot2grid(fig_dims, (0, 1))
    df[attribute].plot(kind='box')


def getTwoAttributesStats(attribute1, attribute2, \
                          df, kindRepresentation='bar'):
    # DA UTILIZZARE SE ENTRAMBI GLI ATTRIBUTI SONO NON CONTINUI!
    tab = pd.crosstab(df[attribute1], df[attribute2])
    print(tab)
    tabpct = tab.div(tab.sum(1).astype(float), axis=0)
    print(tabpct)
    tabpct.plot(kind=kindRepresentation, stacked=True, title=attribute2 + ' Rate by ' + attribute1)
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)
    print ("\n\n\n")
    tab2 = pd.crosstab(df[attribute2], df[attribute1])
    print(tab2)
    tabpct2 = tab2.div(tab2.sum(1).astype(float), axis=0)
    print(tabpct2)
    tabpct2.plot(kind=kindRepresentation, stacked=True, title=attribute1 + ' Rate by ' + attribute2)
    plt.xlabel(attribute2)
    plt.ylabel(attribute1)


def getTwoContinuesAttributesStats(attribute1, attribute2, \
                                   df, kindRepresentation='bar'):
    # DA UTILIZZARE SE UNO(O ENTRAMBI) DEGLI ATTRIBUTI SONO CONTINUI!

    fig, axes = plt.subplots(1, 1, figsize=(10,5))
    axes.scatter(df[attribute1], df[attribute2])
    axes.set_title('Survivors by Age Plot')
    axes.set_xlabel(attribute1)
    axes.set_ylabel(attribute2)
