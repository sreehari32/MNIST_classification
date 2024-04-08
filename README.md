
Here we will discuss the classification model applied to the mnist datasets

1. # Exploratory Analysis

   All the necessary libraries are imported
   >import numpy as np
   >
   >import pandas as pd
   >
   >from sklearn.datasets import fetch_openml
   >
   >import seaborn as sns
   >
   >import matplotlib.pyplot as plt

   Loaded the MNIST dataset from sklearn.datasets
   >mnist = fetch_openml("mnist_784")
   >
   > Extract features and labels
   >   
   >X, y = mnist.data, mnist.target
   >
   >df = pd.concat([X, y], axis="columns")
   >

   ### Dataset distribution plotted
   >sns.countplot(data=df,x='class')
   >
   <img width="445" alt="image" src="https://github.com/sreehari32/MNIST_classification/assets/51872549/7ec235c7-27c1-4fb0-82ea-3092c53fdd45">

   ### plotted the first element
   >print(df.loc[0,'class'])
   >plt.imshow(df.drop('class',axis=1).loc[0].values.reshape(28,28),cmap='gray')


   <img width="332" alt="image" src="https://github.com/sreehari32/MNIST_classification/assets/51872549/7200b1d8-716c-483d-946c-1e08677771d0">




   2. # create_fold.py
      A file named create_fold.py is created to split the samples into k folds
      >
      > a new column is created and inserted with value -1
      >
      > df["kfold"] = -1
      >
      > the dataset is shuffled
      >
      > df = df.sample(frac=1).reset_index(drop=True)
      >
      > defining Kfold object to split the 70,000 examples to 7 folds
      >
      >kf = model_selection.KFold(n_splits=7)
      >
      >for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, "kfold"] = f
      > saving the document
      >
      >df.to_csv("inputs/train_folds.csv", index=False)
      >
      The file creates a csv file which has the 70,000 samples being split to 7 groups of 10,000 each

   3. # model_dispatcher.py
  
      A dictionary is created with model names as keys and models as values

      We started with DecisionTreeClassifier and eventually added the other models after checking the performance

      Usage of model_dispatcher file enables us to add new models to the algorithm without making any changes in the main program

      >>>models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "random_forest": ensemble.RandomForestClassifier(),
}
 >>>   







