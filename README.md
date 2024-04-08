# MNIST_classification
Here we will discuss the classification model applied to the mnist datasets

1. Exploratory Analysis

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
   <img width="451" alt="image" src="https://github.com/sreehari32/MNIST_classification/assets/51872549/eba3c19c-c0da-4eed-bb19-   ceeff3af5bd0">






