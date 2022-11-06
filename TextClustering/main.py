#This is a file for text classification from 20newsgroup dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

#Categories to select from the articles depending on topic
categories = ['sci.electronics', 'alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med',
              'rec.motorcycles', 'sci.space']

#Getting training data for the above categories
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=1)

# Print the first article from the training data and its target
#print("\n".join(training_data.data[0].split("\n")[:100]))
#print("Target/Topic is: ", training_data.target_names[training_data.target[0]])

#Creating a Document Term Matrix (A list of all the words in the document with the frequency of the word)
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
#print(count_vector.vocabulary_)

#Transforming the word occurrences to TF-IDF
#TF-IDF Vectorizer = CountVectorizer + Tfidf_transformer
tfidf_tranformer = TfidfTransformer()
x_train_tfidf= tfidf_tranformer.fit_transform(x_train_counts)
#print(x_train_tfidf)

#Training a Multinomial Model
model = MultinomialNB().fit(x_train_tfidf, training_data.target)

#The below code is for prediction.

#New sentence for prediction.
sentence_for_pred = ['This has nothing to do with church or religion',
                     'Software engineering is getting hotter',
                     'This has nice brakes']

x_pred_counts = count_vector.transform(sentence_for_pred)
x_pred_tfidf = tfidf_tranformer.transform(x_pred_counts)

prediction = model.predict(x_pred_tfidf)

print(prediction, "\n\n\n")
for doc, category in zip(sentence_for_pred, prediction):
    print('%r ----------> %s' %(doc, training_data.target_names[category]))