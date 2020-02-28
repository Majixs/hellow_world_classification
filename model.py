import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics


data_y = []
data_i = []
data_a = []

with open('yelp.txt') as yelp:
    for i in yelp:
        data_y.append(i.rstrip())
    
with open('imdb.txt') as imdb:
    for i in imdb:
        data_i.append(i.rstrip())
        
with open('amazon.txt') as amazon:
    for i in amazon:
        data_a.append(i.rstrip())

for i in range(len(data_y)):
    data_y[i] = data_y[i].split("\t")
    data_i[i] = data_i[i].split("\t")
    data_a[i] = data_a[i].split("\t")

label_y = []
label_a = []
label_i = []

for i in range(len(data_y)):
    label_y.append(data_y[i][1])
    label_a.append(data_a[i][1])
    label_i.append(data_i[i][1])
    
for i in range(len(data_y)):
    data_y[i] = data_y[i][0]
    data_a[i] = data_a[i][0]
    data_i[i] = data_i[i][0]
    
all_data = data_y + data_a + data_i
all_label = label_y + label_a + label_i

train_d = all_data[:2700]
test_d = all_data[2700:]
train_l = all_label[:2700]
test_l = all_label[2700:]

clf = Pipeline([
        ('vect', CountVectorizer()), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultinomialNB())])
    
clf.fit(train_d, train_l)
predicted = clf.predict(test_d)

#вычисляем среднее
mean = np.mean(predicted == test_l)

#report
report = metrics.classification_report(test_l, predicted, target_names = ['0', '1'])

#confusion_matrix
c_m = metrics.confusion_matrix(test_l, predicted)

print("Confusion Matrix")
print(c_m)
print("-  -  -")
print("Classification repotr")
print(report)
print("-  -  -")
print("Mean")
print(mean)

    