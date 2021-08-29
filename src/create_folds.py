# src/create_folds.py
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # training data is in csv file called train.csv
    df = pd.read_csv("/home/ben/programmes/Gittuto/input/train.csv")

    # we create a new column called kfold and initialize to -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from the model selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    # save the new csv with kfold column
    df.to_csv("/home/ben/programmes/Gittuto/input/train_folds.csv", index=False)