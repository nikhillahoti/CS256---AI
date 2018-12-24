
from sklearn.datasets import fetch_20newsgroups
categories = ['rec.autos', 'rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

# Selecting the data for the above mentioned categories only
news_group_train =  fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'), categories=categories,shuffle=True, random_state=42)

# Term Frequency - Inverse Document Frequency gives us the revelance score for each word. If the word occurs many times in a document, it is an important word
# in that document. But, if it occurs in other documents as well, it might be a very generic word like the stop word like a, the.
# Hence, the word should be penalised as it is not important. The If idf takes care of the stop words and so there is not need to remove these
# words from the word list
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(news_group_train.data)
#print(vectors.shape)

# Importing the Naive Bayes module and the metrics module for calculating the accuracy
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Importing the Testing set
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# the text data is to be converted into numerical form in order to find some statistical importance.
vectors_test = vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha=0.01)
clf.fit(vectors, news_group_train.target)

pred = clf.predict(vectors_test)
print("Accuracy is: ", metrics.f1_score(newsgroups_test.target, pred, average='macro'))

# Accuracy obtained is about 98%