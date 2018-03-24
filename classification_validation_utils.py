def fillToBinary(attribute, df,toOthers=True,valueToCut=0.1):
    if toOthers:
        fillToOthers(attribute, df,valueToCut)
    df = pd.concat([df, pd.get_dummies(df[attribute], prefix=attribute+'_Val')], axis=1)
    df = df.drop(attribute,axis=1)
    return df


def getDecisionTree(df, train_features=[], train_target=[], crit='entropy', weightClass=None,\
                    minSamples=10, minSamplesLeaf=5, maxDepth=8, howSplit='random'):
    # restituisce un decision tree istruito sul dataframe in input, con le impostazioni in input
    if train_features == [] or train_target== []:
        train_features = getFeaturesValues(df)
        train_target = getTargetValues(df)

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


def getImportances(classifier, df):
    importances = {}
    for index in range(1, len (df.columns)):
        importances[df.columns[index]] = (float(classifier.feature_importances_[index-1])*100)

    return importances



def splitTrainTest(df, testPercent=0.3):
    featuresValues = getFeaturesValues(df)
    targetValues = getTargetValues(df)
    return train_test_split(featuresValues, targetValues,
                                test_size=testPercent, random_state=0)
