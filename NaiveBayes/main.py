import pandas as pd, numpy as np, sklearn.preprocessing as prepros
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score

credit_data = pd.read_csv("C:\Code\Datasets\credit_data.csv")
x = np.array(credit_data[["income", "age", "loan"]]).reshape(-1, 3)
y = np.array(credit_data.default)

x = prepros.MinMaxScaler.fit_transform(x)

model = GaussianNB()
scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
print("Accuracy Score: ", scores.mean())