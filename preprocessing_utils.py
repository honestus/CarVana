import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from common_utils import getMostFrequent

def removeAttributes(df, removeBadBuy=False, *args):
    #restituisce un dataframe ottenuto, a partire dal dataframe contenente tutti gli attributi
    # rimuovendo tutti gli attributi che abbiamo deciso di rimuovere
    # se @removeBadBuy =True, rimuove anche IsBadBuy dagli attributi del dataframe
    try:
        attributesToRemove = args[0]
    except IndexError:
        attributesToRemove = ['RefId','VehYear','Model','SubModel','WheelTypeID','TopThreeAmericanName',\
        'MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',\
        'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',\
        'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',\
        'MMRCurrentRetailCleanPrice','MMRCurrentRetailAveragePrice',
        'PRIMEUNIT','AUCGUART','BYRNO','IsOnlineSale']
    if removeBadBuy:
        attributesToRemove.append('IsBadBuy')
    return df.drop(attributesToRemove, axis=1)


def fillToOthers(attribute, df, valueToCut=0.1):
    numValues = sum(df[attribute].value_counts())
    numDistinct = len(df[attribute].unique())
    valuesToRepl=[]
    for val in df[attribute].unique():
        if float (len(df[df[attribute]==val]))/numValues < valueToCut:
            valuesToRepl.append(val)
    df[attribute][df[attribute].isin(valuesToRepl)]='Others'
    return df



def mapAttToInteger(attribute, df, removeOriginalAttribute=True):
    attrValues = sorted(df[attribute].unique())
    genders_mapping = dict(zip(attrValues, range(0, len(attrValues) + 1)))
    df[str(attribute)+"Int"] = df[attribute].map(genders_mapping).astype(int)
    if(removeOriginalAttribute):
        del df[attribute]
    return df


def discretizeAttribute(attribute, number_intervals, df, \
                        method='kmeans', removeOriginalAttribute=False):
    # a partire dall'attributo @attribute del dataframe @dataframe, crea un nuovo attributo
    # nominato "discretized@attribute" per il dataframe in input, sfruttando uno dei 3 possibili metodi:
    # kmeans(default), equalFrequency o equalWidth. Tale attributo avrà n valori, dove n=@number_intervals
    # in input; inoltre, se @removeOriginalAttribute =True, rimuove l'attributo originale, una volta creato
    # quello discretizzato
    discretizedAttributeName = 'discretized'+attribute[0].upper()+attribute[1:]

    if method=='frequency':
        df[discretizedAttributeName]=pd.qcut(df[attribute],number_intervals)
    elif method=='width':
        df[discretizedAttributeName]=pd.cut(df[attribute],number_intervals)
    else:
        dfTemp=pd.DataFrame(df[attribute])
        attributeValues = dfTemp.values
        kmeans = KMeans(init='k-means++', n_clusters=number_intervals, n_init=10, max_iter=100)
        kmeans.fit(attributeValues)
        df[discretizedAttributeName+"label"] = kmeans.labels_

        df[discretizedAttributeName] = df[attribute]
        df[discretizedAttributeName] = df[discretizedAttributeName].\
        groupby(df[discretizedAttributeName+"label"]).apply\
        (lambda x: x.replace (to_replace=x.unique(), \
                              value="[" + str(x.min()) + ", " + str(x.max()) +"]"))
        del df[discretizedAttributeName+"label"]
    if removeOriginalAttribute:
        del df[attribute]

    return df



def removeDays(date):
    #restituisce la data ottenuta solo con mese e anno, rimuovendo quindi il giorno
    splitted = str(date).split("/")
    try:
        month = splitted[0]
        year = splitted[2]
        newDate = month + "/" + year
        return newDate
    except:
        return date

def getTrimester(date):
    #a partire dalla data iniziale, restituisce il trimestre(1, 2, 3 o 4) e l'anno di appartenenza
    splitted = str(date).split("/")
    try:
        month = splitted[0]
        trimester = (int(month)-1)//3 + 1
        year = splitted[2]
        dateToTrimester = str(trimester) + "/" + year
        return dateToTrimester
    except:
        return date


def sizeMapping(df):
    sizes = df['Size']
    sizes [df['Size'].isin(['MEDIUM SUV','CROSSOVER'])]='MEDIUM'
    sizes [df['Size'].isin(['SMALL TRUCK','LARGE TRUCK','VAN'])]='LARGE'
    sizes [df['Size'].isin(['SPECIALTY'])]='SPORTS'
    sizes [df['Size'].isin(['SMALL SUV'])]='COMPACT'
    return sizes


def replaceMissingValues(df, replaceSize=True, **kwargs):

    #RISOLUZIONE MISSING VALUES TRIM PER IL DATAFRAME IN INPUT
    """ SOSTITUISCO I VALORI DEI MISSING VALUES DI TRIM CON I VALORI
        DI TRIM PIÙ FREQUENTI AVENTI COME MODELLO E SOTTOMODELLO QUELLO DELL'OGGETTO
        PER CUI TRIM E' MISSING VALUES: ad es.:
        Ferrari(Make) Enzo(Model) 5000GT(SubModel), trim come miss.value.
        Vedo i trim associati agli oggetti aventi Enzo come Model e
        5000GT come SubModel e quindi prendo il più frequente di questi
        e lo sostituisco all'oggetto avente Trim come missing value """

    df ["Trim"] = df ["Trim"].groupby([df["Model"],df["SubModel"]]).apply \
    (lambda x: x.fillna(getMostFrequent(x)))
    # Se dopo aver sostituito i valori più frequenti dei Sottomodelli,
    # ci sono ancora missing values, allora:
    # SOSTITUISCO I VALORI DEI MISSING VALUES DI TRIM CON I VALORI
    # DI TRIM PIÙ FREQUENTI AVENTI COME MODELLO QUELLO DELL'OGGETTO
    # PER CUI TRIM E' MISSING VALUES: ad es.:
    # Ferrari(Make) Enzo(Model), trim e SubModel come miss.value
    # (per questo motivo devo analizzare Model e non posso analizzare SubModel).
    # Vedo i trim associati agli oggetti aventi Enzo come Model
    # e quindi prendo il più frequente di questi e lo sostituisco
    # all'oggetto avente Trim come missing value
    if len(df [df.Trim.isnull()])>0:
        df ["Trim"] = df ["Trim"].groupby(df["Model"]).apply \
        (lambda x: x.fillna(getMostFrequent(x)))
    # Se ancora ci sono missing values per trim, li sostituisco
    #con il valore più frequente per l'intero dataset.
    if len(df [df.Trim.isnull()])>0:
        df.Trim = df.Trim.fillna(getMostFrequent(df.Trim))


    #RISOLUZIONE MISSING VALUES COLOR PER IL DATAFRAME IN INPUT
    """ Rimpiazzo i NOT AVAIL con i NAN, quindi rimpiazzo tutti i missing values(NAN)
    con il valore più frequente di color nell'intero dataset """
    df ["Color"] [df["Color"]=='NOT AVAIL'] = df["Color"].map(lambda x: np.nan)
    df["Color"] = df["Color"].fillna(getMostFrequent(df["Color"]))



    #RISOLUZIONE MISSING VALUES SIZE PER IL DATAFRAME IN INPUT
    """ Sostituisco i miss values di Size, sfruttando i valori di Model.
    Per ogni model, infatti, c'è quasi sempre una sola size associata, quindi Model
    è un ottimo modo per discriminare le diverse size. Quindi sostituisco i missing values
    di size con il valore più frequente di size associato ad un certo model.
    Se non vi sono valori per Size di un certo Model nel dataset,
    prendo il valore più frequente di Size nell'intero dataset."""
    df["Size"] = df["Size"].groupby(df["Model"]).apply\
    (lambda x: x.fillna(getMostFrequent(x)))

    if len(df [df["Size"].isnull()]) > 0:
        df["Size"] = df["Size"].fillna(getMostFrequent(df["Size"]))


    #RISOLUZIONE MISSING VALUES NATIONALITY PER IL DATAFRAME IN INPUT
    """ Rimpiazzo i missing values di Nationality sfruttando il valore di
    make degli oggetti: ad ogni make è associato sostanzialmente
    sempre un solo valore per nationality. In ogni caso, se per un make
    nel dataset ci sono più valori per nationality, prendo quello più
    frequente e uso quello """
    df["Nationality"] = df["Nationality"].groupby(df["Make"]).apply \
    (lambda x: x.fillna(getMostFrequent(x)))

    if len(df [df["Nationality"].isnull()]) > 0:
        df["Nationality"] = df["Nationality"].fillna\
        (getMostFrequent(df["Nationality"]))

    if replaceSize:
        df['Sizes'] = sizeMapping(df)

    #RISOLUZIONE MISSING VALUES WHEELTYPE PER IL DATAFRAME IN INPUT
    """ Rimpiazzo i missing values di WheelType sfruttando il valore di Size.
    Infatti ho notato una netta differenza di bilanciamento dei valori di WheelType, per le
    diverse size nel dataset. Per alcune Size, la grande maggioranza sono Covers, per
    un'altra Size, la grande maggioranza sono Special, e per altre ancorad
    troviamo una maggioranza di Alloy. Ciò è anche abbastanza comprensibile, dato che per un
    suv difficilmente troveremo "cerchi comuni", per una sportiva molto più probabilmente
    troveremo dei cerchi "Special", mentre per un'utilitaria troveremo spesso
    dei cerchi "comuni"(Alloy). """
    df["WheelType"] = df["WheelType"].groupby(df["Size"]).apply \
    (lambda x: x.fillna(getMostFrequent(x)))

    if len(df [df["WheelType"].isnull()]) > 0:
        df["WheelType"] = df["WheelType"].fillna\
        (getMostFrequent(df["WheelType"]))


    #RISOLUZIONE MISSING VALUES TRANSMISSION PER IL DATAFRAME IN INPUT
    """ Sostituisco tutti i valori di Transmission con il loro valore in maiuscolo
    nel dataset infatti ci sono oggetti con 'Manual' ed oggetti con 'MANUAL' """
    df['Transmission'] = df["Transmission"][df["Transmission"].notnull()]\
    .map(lambda x: str(x).upper())
    df["Transmission"] = df["Transmission"].fillna(getMostFrequent(df["Transmission"]))


    #RIMPIAZZO PREZZI UGUALI A 0 O A 1 CON NAN, PER IL DATAFRAME IN INPUT
    """Sostituisco tutti gli 0 per i vari prezzi con nan
    inoltre sostituisco tutti gli 1 con nan per MMRCurrentRetailAveragePrice. """
    df['MMRCurrentRetailAveragePrice'] [df['MMRCurrentRetailAveragePrice']==0.0] = df['MMRCurrentRetailAveragePrice'].map(lambda x: np.nan)
    df ['MMRAcquisitionAuctionCleanPrice'] [df.MMRAcquisitionAuctionCleanPrice.isin ([0.0,1.0])] = df.MMRAcquisitionAuctionCleanPrice.map(lambda x: np.nan)
    df ['MMRAcquisitionRetailAveragePrice'] [df.MMRAcquisitionRetailAveragePrice==0] = df.MMRAcquisitionRetailAveragePrice.map(lambda x: np.nan)
    df ['MMRAcquisitonRetailCleanPrice'] [df.MMRAcquisitonRetailCleanPrice==0] = df.MMRAcquisitonRetailCleanPrice.map(lambda x: np.nan)

    try:
        dates = kwargs['dates']
        #RISOLUZIONE MISSING VALUES MMRCurrentRetailAveragePrice PER IL DATAFRAME IN INPUT
        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePrice.groupby\
        ([dates, df.Model,df.SubModel]).apply\
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePriceFill.groupby \
        ([dates, df.Model]).apply \
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePriceFill.groupby \
        ([dates, df.Make]).apply \
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

        df['PurchDate'] = dates

    except (KeyError, ValueError):
        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePrice.groupby\
        ([df['PurchDate'], df.Model,df.SubModel]).apply\
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePriceFill.groupby \
        ([df['PurchDate'], df.Model]).apply \
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

        df['MMRCurrentRetailAveragePriceFill'] = \
        df.MMRCurrentRetailAveragePriceFill.groupby \
        ([df['PurchDate'], df.Make]).apply \
        (lambda x: x.replace(to_replace=np.nan, value=x.median()))

    return df
