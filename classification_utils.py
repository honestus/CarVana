def fillToBinary(attribute, dataframe=originalDf,toOthers=True,valueToCut=0.1):
    if toOthers:
        fillToOthers(attribute, dataframe,valueToCut)
    dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[attribute], prefix=attribute+'_Val')], axis=1)
    dataframe = dataframe.drop(attribute,axis=1)
    return dataframe
