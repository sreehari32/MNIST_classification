import pandas as pd
import numpy as np
import config
import os
import joblib
import model_dispatcher
from sklearn.metrics import accuracy_score


def run(fold, model):
    # Load MNIST data
    df = pd.read_csv(config.training_file)

    # training data is where k fold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # cross validation data is where kfold =fold
    df_cv = df[df.kfold == fold].reset_index(drop=True)

    # extracting X_train values
    X_train = df_train.drop("class", axis=1).values
    # extracting y_train values
    y_train = df_train["class"].values

    # extracting X_cv values
    X_cv = df_cv.drop("class", axis=1).values
    # extracting y_cv values
    y_cv = df_cv["class"].values

    # initialising models
    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_cv)

    # calculating the accuracy
    accuracy = accuracy_score(y_cv, pred)
    print(f"For fold={fold} the accuracy is = {accuracy}")

    run(fold=0, model="random_forest")
    # save the model
    # joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


run(fold=0, model="decision_tree_gini")
run(fold=0, model="decision_tree_entropy")
run(fold=0, model="random_forest")


