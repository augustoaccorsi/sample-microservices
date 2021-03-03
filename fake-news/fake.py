#https://www.kaggle.com/asurajsubramanian/fake-news-classification-easiest-99-accur-b21d78

from keras.models import Sequential
from keras import layers

import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

df = pd.read_csv('./data_set/train.csv')
test = pd.read_csv('./data_set/test.csv')
df.head()

df = df.fillna('')
test = test.fillna('')
df['total'] = df['author']+' '+df['title']
test['total']=test['author']+' '+test['title']

X = df.drop('label',axis=1)
y = df['label']
print(X.shape)
print(y.shape)

def stemmer(input_):
    review = re.sub('[^a-zA-Z]',' ',input_)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

df['total'] = df['total'].apply(stemmer)

sentences = df['total'].values
labels = df['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# len(vectorizer.vocabulary_) = 15031

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=6,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)
                
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save('.')