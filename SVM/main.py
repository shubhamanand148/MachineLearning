import numpy as np, matplotlib.pyplot as plt, sklearn.svm as svm
from mlxtend.plotting import plot_decision_regions

x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([0.3, 3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([4, 7, 1.5, 6.3, 1.9, 2.9, 7.1])

#Training Data
x = np.array([[0.3, 4], [0.3, 1], [0.5, 4.5], [1, 2.3], [1.4, 1.9], [1.7, 8.9], [2, 4.1], [3.3, 7],
              [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]])

#0: Blue, 1: Red
y = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

plt.plot(x_blue, y_blue, 'ro', color='blue')
plt.plot(x_red, y_red, 'ro', color='red')
plt.plot(4.5, 4.5, 'ro', color='green')

# C: This decides the margin for error in training.
# Low value of C: Smooth curve (Underfitting).
# Hih value of C: Complex curve (Overfitting).
classifier = svm.SVC(C=1000)
classifier.fit(x, y)
print(classifier.predict([[4.5, 4.5]]))

#Plotting Decision Boundry
plot_decision_regions(x, y, clf=classifier, legend=2)

plt.show()