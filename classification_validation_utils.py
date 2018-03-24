def fillToBinary(attribute, dataframe=originalDf,toOthers=True,valueToCut=0.1):
    if toOthers:
        fillToOthers(attribute, dataframe,valueToCut)
    dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[attribute], prefix=attribute+'_Val')], axis=1)
    dataframe = dataframe.drop(attribute,axis=1)
    return dataframe


def getDecisionTree(dataframe, train_features=[], train_target=[], crit='entropy', weightClass=None,\
                    minSamples=10, minSamplesLeaf=5, maxDepth=8, howSplit='random'):
    # restituisce un decision tree istruito sul dataframe in input, con le impostazioni in input
    if train_features == [] or train_target== []:
        train_features = getFeaturesValues(dataframe)
        train_target = getTargetValues(dataframe)

    # Genera il decision tree con le scelte effettuate in input per lo split, profondità etc.
    clf = tree.DecisionTreeClassifier(criterion=crit, splitter=howSplit, max_depth=maxDepth,
                                      min_samples_split=minSamples, min_samples_leaf=minSamplesLeaf,\
                                     class_weight = weightClass)

    return clf.fit(train_features, train_target)


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def getImportances(classifier, dataframe):
    importances = {}
    for index in range(1, len (dataframe.columns)):
        importances[dataframe.columns[index]] = (float(classifier.feature_importances_[index-1])*100)

    return importances



def splitTrainTest(dataframe, testPercent=0.3):
    featuresValues = getFeaturesValues(dataframe)
    targetValues = getTargetValues(dataframe)
    return train_test_split(featuresValues, targetValues,
                                test_size=testPercent, random_state=0)
