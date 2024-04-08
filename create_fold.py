import pandas as pd
from sklearn import model_selection
from sklearn.datasets import fetch_openml

# Load MNIST data
mnist = fetch_openml("mnist_784")

if __name__ == "__main__":

    # Extract features and labels
    X, y = mnist.data, mnist.target
    df = pd.concat([X, y], axis="columns")

    # a new column is created and inserted with value -1
    df["kfold"] = -1
    # the dataset is shuffled
    df = df.sample(frac=1).reset_index(drop=True)
    # defining Kfold object to split the 70,000 examples to 7 folds
    kf = model_selection.KFold(n_splits=7)
    for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, "kfold"] = f
    # saving the document  
    df.to_csv("inputs/train_folds.csv", index=False)
