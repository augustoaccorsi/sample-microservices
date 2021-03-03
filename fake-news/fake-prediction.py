# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./data_set/test.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
#%matplotlib inline

true_df = pd.read_csv('./data_set/True.csv')
fake_df = pd.read_csv('./data_set/Fake.csv')
print(f'Shape True New articles: {true_df.shape}')
print(f'Shape Fake New articles: {fake_df.shape}')

true_df['Label'] = 'True'
fake_df['Label'] = 'Fake'
true_df.head()

df_final = pd.concat([true_df,fake_df])
print(f'Shape of Combined DATA SET: {df_final.shape}')

df_final['Label'].shape

TRAIN_SET,TEST_SET, TRAIN_label,TEST_label = train_test_split(df_final['text'], df_final['Label'],random_state = 10,
                                                               test_size = 0.20)
print(f'TRAIN FEATURE SET: {TRAIN_SET.shape}')
print(f'TRAIN LABEL SET: {TRAIN_label.shape}')
print(f'TEST FEATURE SET: {TEST_SET.shape}')

# Building Pipeline for raw text transformation
clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words= "english")),
    ('classifier', MultinomialNB()),
    ])

model_NB = clf.fit(TRAIN_SET, TRAIN_label)

score = model_NB.score(TEST_SET, TEST_label)
print(f'Accuracy of Model is: {score}')

#Confusion Matrix
# Compute confusion matrix
y_predicted_labels = model_NB.predict(TEST_SET)
cnf_matrix = confusion_matrix(TEST_label, y_predicted_labels)
np.set_printoptions(precision=2)
cnf_matrix

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['True', 'Fake'],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['True', 'Fake'], normalize=True,
                      title='Normalized confusion matrix')

#plt.show()

from sklearn.linear_model import LogisticRegression

# Building Pipeline for raw text transformation
clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words= "english")),
    ('classifier', LogisticRegression()),
    ])

model_LogReg = clf.fit(TRAIN_SET, TRAIN_label)

score = model_LogReg.score(TEST_SET, TEST_label)
print(f'Accuracy of Model is: {score}')

#Confusion Matrix
# Compute confusion matrix
y_predicted_labels = model_LogReg.predict(TEST_SET)
cnf_matrix = confusion_matrix(TEST_label, y_predicted_labels)
np.set_printoptions(precision=2)
cnf_matrix

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#With Normalization
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['True', 'Fake'],
                      title='Confusion matrix, without normalization')
# With normalization
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['True', 'Fake'], normalize=True,
                      title='Normalized confusion matrix')

#plt.show()

joblib.dump(model_LogReg,'Log_Reg.joblib')

sample_example = ['Deaths were reported in several cities including Yangon, Dawei and Mandalay as police used live rounds and tear gas. Security forces began the violent crackdown on Saturday, after weeks of largely peaceful protests against the 1 February military takeover. Government leaders, including Aung San Suu Kyi, were overthrown and detained.',
                  'Donald Trump says he has no plans to launch a new political party, telling a conservative conference in Florida that it would split the Republican vote.',
                  'Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year!  Bishop Talbert Swan (@TalbertSwan) December 31, 2017no one likes you  Calvin (@calvinstowell) December 31, 2017Your impeachment would make 2018 a great year for America, but I ll also accept regaining control of Congress.  Miranda Yaver (@mirandayaver) December 31, 2017Do you hear yourself talk? When you have to include that many people that hate you you have to wonder? Why do the they all hate me?  Alan Sandoval (@AlanSandoval13) December 31, 2017Who uses the word Haters in a New Years wish??  Marlene (@marlene399) December 31, 2017You can t just say happy new year?  Koren pollitt (@Korencarpenter) December 31, 2017Here s Trump s New Year s Eve tweet from 2016.Happy New Year to all, including to my many enemies and those who have fought me and lost so badly they just don t know what to do. Love!  Donald J. Trump (@realDonaldTrump) December 31, 2016This is nothing new for Trump. He s been doing this for years.Trump has directed messages to his  enemies  and  haters  for New Year s, Easter, Thanksgiving, and the anniversary of 9/11. pic.twitter.com/4FPAe2KypA  Daniel Dale (@ddale8) December 31, 2017Trump s holiday tweets are clearly not presidential.How long did he work at Hallmark before becoming President?  Steven Goodine (@SGoodine) December 31, 2017He s always been like this . . . the only difference is that in the last few years, his filter has been breaking down.  Roy Schulze (@thbthttt) December 31, 2017Who, apart from a teenager uses the term haters?  Wendy (@WendyWhistles) December 31, 2017he s a fucking 5 year old  Who Knows (@rainyday80) December 31, 2017So, to all the people who voted for this a hole thinking he would change once he got into power, you were wrong! 70-year-old men don t change and now he s a year older.Photo by Andrew Burton/Getty Images.',
                  'The US Congress has certified Joe Biden\s victory in the presidential election, hours after supporters of Donald Trump stormed the building in an attack that saw four people die.']

model_loaded= joblib.load('Log_Reg.joblib')

print()
print(model_loaded)

for text in sample_example:
    print(f'Predicted label for news:{model_loaded.predict([text])[0]}')

    