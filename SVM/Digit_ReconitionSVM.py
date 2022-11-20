import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.metrics import accuracy_score

#Loading the dataset
digit_data = datasets.load_digits()

#Joining the images along with target values
images_and_labels = list(zip(digit_data.images, digit_data.target))

'''
Printing the dataset along with target value.
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' %label)

plt.show()
'''

#The classifier works in 1D array.
#Transforming/Flattening the 8x8 image to 1D array with 64 elements.
data = digit_data.images.reshape((len(digit_data.images), -1))

classifier = svm.SVC(gamma=0.001)

train_test_split = int(len(digit_data.images) * 0.8)