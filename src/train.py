# src/train.py
import os
import config
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    #df = pd.read_csv("/home/ben/programmes/Gittuto/input/train_folds.csv")
    df =pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to the provided fold
    df_train = df[df.kfold !=fold].reset_index(drop=True)

    # validation data is where kfold is equal to the provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from  dataframe and convert it to a 
    # numpy array by using .values
    # target is the label column in the dataframe
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # for validation
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()

    # fit the model on the training data
    clf.fit(x_train,y_train)

    # create predictions for the validation sample
    preds = clf.predict(x_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    #joblib.dump(clf, f"/home/ben/programmes/Gittuto/models/dt_{fold}.bin")
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )



if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
    print("done")
    