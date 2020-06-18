import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('bbc-text.csv')
#data = pd.read_csv('test.csv')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 2225):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print(round(i*100/2225))
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
dataset['category']= le.fit_transform(dataset['category'])
y = dataset.iloc[:, 0].values
# Splitting the dataset into the Training set and Test set




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred)*100)


from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train) 
y_preds = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cms = confusion_matrix(y_test, y_preds)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_preds)*100)




from sklearn.tree import DecisionTreeClassifier
declf = DecisionTreeClassifier()
declf = declf.fit(X_train,y_train)
predictedDT= declf.predict(X_test)
conDT = confusion_matrix(y_test,predictedDT)
print(metrics.accuracy_score(y_test,predictedDT)*100)



from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predictedknn= model.predict(X_test) 
from sklearn.metrics import confusion_matrix
conknn = confusion_matrix(y_test,predictedknn)
print(metrics.accuracy_score(y_test,predictedknn)*100)

label=['Navie Bayes','Supprt Vector Classifier','decision Tree','KNN']

acc= [
    metrics.accuracy_score(y_test,y_pred)*100,
    metrics.accuracy_score(y_test,y_preds)*100,
    metrics.accuracy_score(y_test,predictedDT)*100,
    metrics.accuracy_score(y_test,predictedknn)*100
      ]
#plt.figure(figsize=(15,15))
print(acc)
index = np.arange(len(label))
plt.bar(index,acc, color=['cyan','green','red','black'])
plt.xlabel('Accuracy', fontsize=10)
plt.ylabel('In Percentage', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=30)
plt.title('Classification Algorithm')
plt.ylim(0,100)
plt.savefig('reportacc.png')
plt.show()

