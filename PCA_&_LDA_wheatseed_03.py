# loading the dataset and splitting it into the feature matrix X and the target vector y
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
y = data.target

"""standardize the data using StandardScaler to make sure that 
all features are on the same scale"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""perform PCA to see if we can reduce the dimensionality of the data
while retaining most of its variance. We'll start by plotting the
explained variance ratio for each principal component"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(X_scaled)

plt.bar(range(1, 14), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

"""From the plot, we can see that the first two principal components 
explain almost 60% of the variance, while the first three principal 
components explain over 70% of the variance. Let's plot the data in the 
first two principal components to see if there is any separation between 
the three classe"""


X_pca = pca.transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
"""We can see that there is some separation between the three classes, 
but it is not very clear. Let's see if LDA can do better"""

"""LDA is a supervised dimensionality reduction technique that tries to 
maximize the separation between the classes while minimizing the variance 
within each class. We'll start by fitting an LDA model:"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_scaled, y)

#let's transform the data into the two-dimensional LDA space and plot it
X_lda = lda.transform(X_scaled)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()

"""We can see that LDA has done a much better job of separating the three classes. 
The first linear discriminant separates the first class from the other two, while 
the second linear discriminant separates the second class from the third"""
