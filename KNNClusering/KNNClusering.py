import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

#Loading the dataset
credit_data = pd.read_csv("C:\Code\Datasets\credit_data.csv")
x = np.array(credit_data[["income", "age", "loan"]]).reshape(-1, 3)
y = np.array(credit_data.default)

#Preprossing the data
x = preprocessing.MinMaxScaler().fit_transform(x)

#A list for all the cross validation scores KNN with K = 1 to 100
cross_valid_scores = []

for i in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())

#The best K for the KNN
best_k_value = np.argmax(cross_valid_scores)
print("Best k value: ", best_k_value)

#Creating KNN model with best K
best_model = KNeighborsClassifier(n_neighbors=best_k_value)
best_score = cross_val_score(best_model, x, y, cv=10, scoring='accuracy')
print("Accuracy: ", best_score.mean())