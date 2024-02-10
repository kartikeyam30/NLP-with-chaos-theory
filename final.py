# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 04:52:44 2021

@author: Kartikeya Mehrotra
"""

# %%

import num2words
import re
import warnings
import gensim
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import cycle
import scipy
from sklearn.metrics import roc_auc_score
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.integrate import odeint
# import winsound
import math

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from autocorrect import spell
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD, Adam, Adamax, Adagrad
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Imb_Pipe

# %% Main

content = []
label = []
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
mlb = MultiLabelBinarizer()


# Code to convert a binary sequence to its decimal equivalent
def binary_to_decimal(binary_number):
    num = 0
    for i in range(0, len(binary_number)):
        num = num + binary_number[i] * (2 ** i)
    return num


# Code to clean the text of unwanted objects
def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()


# Code to read data from specified PATH and extract the messages along with labels from it
def read_data(Path):
    data = pd.read_csv(Path)

    content = pd.DataFrame(data['Text'])
    labels = pd.DataFrame(data['Label'])

    return data, content, labels


# Code to remove special charcatres along with "Stop Words" from the text and perform other cleaing operations
def pre_processing(content):
    for x in range(0, len(content)):
        text = content.iloc[x][0]
        text = text.replace('\par', '')
        tokens = (word_tokenize(text))
        text = TreebankWordDetokenizer().detokenize([w for w in tokens[:] if not w in stop_words]) + "\n"
        text = (ps.stem((text.lower())));
        text = re.sub('[^A-Za-z0-9" "]+', '', text)
        text = lemmatizer.lemmatize(text)
        content.iloc[x][0] = text.lower()
        x += 1

    return content


# Code to convert string labels to enumerations
def map_str_to_enum(content, label):
    for i in range(0, len(content)):
        if label[i] == "conclusion":
            label[i] = 0
        elif str(label[i][0]) == "exclusivity":
            label[i] = 1
        elif str(label[i][0]) == "friendship_forming":
            label[i] = 2
        elif str(label[i][0]) == "relationship_forming":
            label[i] = 3
        elif str(label[i][0]) == "other":
            label[i] = 4
        elif str(label[i][0]) == "sexual":
            label[i] = 5
        else:
            label[i] = 6


# %% ****************************** ORIGINAL DATA ****************************************
# %% Naive Bayes
Path = 'final.json'                         # INPUT FILE PATH HERE ########
number_of_max_features = 500                # SET MAX NUM OF FEATURES HERE ####
data, content, labels = read_data(Path)
content = pre_processing(content)

X_train, X_test, y_train, y_test = train_test_split(content['Text'], labels['Label'], test_size=0.2, shuffle=True,
                                                    stratify=labels)
text_clf = Pipeline([('vect', CountVectorizer(max_features=number_of_max_features)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=0.1)),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
nb_wo_chaos_original = metrics.classification_report(y_test, predicted)
print(nb_wo_chaos_original)

#%% KNN without Lorenz

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
knn_wo_chaos_original = metrics.classification_report(y_test, predicted)
print(knn_wo_chaos_original)

# %% Linear SVC Without Lorenz

text_clf = Pipeline([('vect', CountVectorizer(max_features=number_of_max_features)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC())
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
svc_wo_chaos_original = metrics.classification_report(y_test, predicted)
print(svc_wo_chaos_original)

# %% **************************************UNDERSAMPLED DATA*******************************

# %% Naive Bayes without Lorenz
data, content, labels = read_data(Path)
content = pre_processing(content)

train, label = make_imbalance(content, labels,
                              sampling_strategy={0: 261, 1: 473, 2: 532, 3: 800, 4: 1000, 5: 650, 6: 275})

X_train, X_test, y_train, y_test = train_test_split(train['Text'], label['Label'], test_size=0.2, shuffle=True,
                                                    stratify=label)
text_clf = Pipeline([('vect', CountVectorizer(max_features=number_of_max_features)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=0.1)),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
nb_wo_chaos_undersampled = metrics.classification_report(y_test, predicted)
print(nb_wo_chaos_undersampled)

# %% KNN without Lorenz
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
knn_wo_chaos_undersampled = metrics.classification_report(y_test, predicted)
print(knn_wo_chaos_undersampled)


#%% Linear SVC Without Lorenz

text_clf = Pipeline([('vect', CountVectorizer(max_features=number_of_max_features)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC())
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
svc_wo_chaos_undersampled = metrics.classification_report(y_test, predicted)
print(svc_wo_chaos_undersampled)


#%% **************************************OVERSAMPLED DATA*******************************


#%% Naive Bayes without Lorenz
data, content, labels = read_data(Path)
content = pre_processing(content)

train, label = make_imbalance(content, labels,
                              sampling_strategy={0: 261, 1: 473, 2: 532, 3: 800, 4: 1000, 5: 650, 6: 275})

vect = CountVectorizer(max_features=number_of_max_features)
train = vect.fit_transform(train['Text'])
tran = TfidfTransformer()
train = tran.fit_transform(train)

sm = SMOTE()
train, label = sm.fit_resample(train, label)

X_train, X_test, y_train, y_test = train_test_split(train, label['Label'], test_size=0.2, shuffle=True,
                                                    stratify=label)
text_clf = MultinomialNB(alpha=0.1)
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
nb_wo_chaos_oversampled = metrics.classification_report(y_test, predicted)
print(nb_wo_chaos_oversampled)

# %% KNN without Lorenz
# X_train, X_test, y_train, y_test = train_test_split(content['Text'], labels['Label'], test_size=0.2, shuffle=True,
#                                                    stratify=labels)

vect = CountVectorizer(max_features=number_of_max_features)
train = vect.fit_transform(train['Text'])
tran = TfidfTransformer()
train = tran.fit_transform(train)

sm = SMOTE()
train, label = sm.fit_resample(train, label)

X_train, X_test, y_train, y_test = train_test_split(train, label['Label'], test_size=0.2, shuffle=True,
                                                    stratify=label)
text_clf = KNeighborsClassifier()
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
knn_wo_chaos_oversampled = metrics.classification_report(y_test, predicted)
print(knn_wo_chaos_oversampled)

# %% Linear SVC Without Lorenz

vect = CountVectorizer(max_features=number_of_max_features)
train = vect.fit_transform(train['Text'])
tran = TfidfTransformer()
train = tran.fit_transform(train)

sm = SMOTE()
train, label = sm.fit_resample(train, label)

X_train, X_test, y_train, y_test = train_test_split(train, label['Label'], test_size=0.2, shuffle=True,
                                                    stratify=label)
text_clf = LinearSVC()
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
svc_wo_chaos_oversampled = metrics.classification_report(y_test, predicted)
print(svc_wo_chaos_oversampled)
# %%


number_of_max_features = 500
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(content['Text'])

df1 = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
df1['mean_tfidf'] = df1.mean(axis=1)
df1.head()

df2 = pd.DataFrame()
df2['TFIDF_Mean'] = df1['mean_tfidf']
df2['annotations'] = content
df2.head()

# %%
number_of_max_features = 50000
vectorizer = TfidfVectorizer(max_features=number_of_max_features)
X = vectorizer.fit_transform(content['Text'])

dfg = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dfg = dfg.apply(np.ceil)
counts = pd.DataFrame(dfg.sum(axis=0))
counts = counts.rename(columns={0: 'Counts'})
counts.sort_values('Counts', ascending=False).head(20).plot.bar(title='Highest word counts in the dataset')

sexuallist = labels[labels['Label'] == 5].index.values.tolist()
X = vectorizer.fit_transform(content.loc[sexuallist]['Text'])
dfg1 = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

dfg1 = dfg1.apply(np.ceil)
counts2 = pd.DataFrame(dfg1.sum(axis=0))
counts2 = counts2.rename(columns={0: 'Counts'})
counts2.sort_values('Counts', ascending=False).head(20).plot.bar(title='Highest word counts in sexual category')

allvals = counts.sort_values('Counts', ascending=False).head(100).index
sexualvals = counts2.sort_values('Counts', ascending=False).head(100).index

commons = allvals[allvals[:].isin(list(sexualvals[:]))]

allvals = counts[counts.index.isin(commons)]
sexualvals = counts2[counts2.index.isin(commons)]

allvals = allvals.reset_index()
sexualvals = sexualvals.reset_index()
allvals = allvals.merge(sexualvals, how='left', on='index')

# %%
# Since Sexual-labelled texts are 879 out of a total of 10212 entries(approximately 8.5%),
# we will compare the sexual count of the words to 1/10th of that in the total count to see the dominance

X_axis = np.arange(len(allvals))
plt.figure(figsize=(20, 10))
plt.bar(X_axis - 0.2, allvals['Counts_y'], 0.4, label='Sexual')
plt.bar(X_axis + 0.2, allvals['Counts_x'] / 10, 0.4, label='Total')

plt.xticks(X_axis, allvals['index'], rotation=90)
plt.xlabel("Words")
plt.ylabel("Counts")
plt.title("Behaviour of most used sexual words compared to 1/10th of the total count")
plt.legend()
plt.show()

# As we can see, most used words in sexual texts, are also the words mostly used in sexual references.
# To get a clearer picture of majority, we will compare to 1/8th of the total to see the dominance.

X_axis = np.arange(len(allvals))
plt.figure(figsize=(20, 10))
plt.bar(X_axis - 0.2, allvals['Counts_y'], 0.4, label='Sexual')
plt.bar(X_axis + 0.2, allvals['Counts_x'] / 8, 0.4, label='Total')

plt.xticks(X_axis, allvals['index'], rotation=90)
plt.xlabel("Words")
plt.ylabel("Counts")
plt.title("Behaviour of most used sexual words compared to 1/8th of the total count")
plt.legend()
plt.show()

# Still, most words are dominant.

# Since there are 7 total categories, let us see the impact of these words by category by
# comparing to 1/7th of the total word count

X_axis = np.arange(len(allvals))
plt.figure(figsize=(20, 10))
plt.bar(X_axis - 0.2, allvals['Counts_y'], 0.4, label='Sexual')
plt.bar(X_axis + 0.2, allvals['Counts_x'] / 7, 0.4, label='Total')

plt.xticks(X_axis, allvals['index'], rotation=90)
plt.xlabel("Words")
plt.ylabel("Counts")
plt.title("Behaviour of most used sexual words as a category")
plt.legend()
plt.show()

# Even now most words are dominant.

# This shows these words are not only those that have the highest count in sexual texts,
# they are also not occuring elsewhere as much, hence providing an almost direct relation to the category.

# Let us have a look at these words

allvals = counts.sort_values('Counts', ascending=False).head(200).index
sexualvals = counts2.sort_values('Counts', ascending=False).head(200).index

commons = allvals[allvals[:].isin(list(sexualvals[:]))]

allvals = counts[counts.index.isin(commons)]
sexualvals = counts2[counts2.index.isin(commons)]

allvals = allvals.reset_index()
sexualvals = sexualvals.reset_index()
allvals = allvals.merge(sexualvals, how='left', on='index')

allvals.Counts_x = (allvals.Counts_x / 7).round(0)
allvals = allvals.where((allvals['Counts_x']) <= allvals['Counts_y'])

allvals = allvals.dropna()
# %%

d = {}
allvals = allvals.reset_index().drop('level_0', axis=1)
for i in range(len(allvals)):
    d[allvals['index'][i]] = allvals['Counts_y'][i]

wordcloud = WordCloud().generate_from_frequencies(frequencies=d)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

# %%

le = LabelEncoder()
df2['annotations'] = le.fit_transform(df2['annotations'])
df2['Labels'] = labels
df2.head()

df2_temp = df2[:]

# %%

columns = df2_temp.columns
x = df2_temp.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2_temp = pd.DataFrame(x_scaled, columns=columns)

df2_temp['Labels'] = labels
df2_temp.head()
# df2_temp.iloc[0,0]
df2_temp['TFIDF_Mean'] = df2_temp['TFIDF_Mean'] * 10
# define the initial system state (aka x, y, z positions in space)
initial_state = [0.1, 0, 0]

# define the system parameters sigma, rho, and beta
sigma = 10.
rho = 28.
beta = 8 / 3.
p = 0


# define the lorenz system
# x, y, and z make up the system state, t is time, and sigma, rho, beta are the system parameters
def lorenz_system(current_state, t):
    # positions of x, y, z in space at the current time point
    x, y, z = current_state

    # define the 3 ordinary differential equations known as the lorenz equations
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    # return a list of the equations that describe the system
    return [dx_dt, dy_dt, dz_dt]


# plot the lorenz attractor in three-dimensional phase space


# %%

conc = df2_temp[df2_temp['Labels'] == 1]
exclusive = df2_temp[df2_temp['Labels'] == 2]
friendship = df2_temp[df2_temp['Labels'] == 3]
relationship = df2_temp[df2_temp['Labels'] == 4]
sexual = df2_temp[df2_temp['Labels'] == 5]
rest = df2_temp[df2_temp['Labels'] != 5]
other = df2_temp[df2_temp['Labels'] == 6]

data = [sexual, conc, exclusive, friendship, relationship, other]

frames = ['s', 'c', 'e', 'f', 'r', 'o']

start_time = 0
end_time = 100
steps = 3000
time_points = np.linspace(start_time, end_time, 3000)

# %%

for (i, d) in zip(data, frames):
    array = np.zeros((len(i), 3000, 3), dtype=np.float32)
    for j in range(0, len(i)):
        initial_state = [i.iloc[j, 2], i.iloc[j, 1], i.iloc[j, 0]]
        xyz = odeint(lorenz_system, initial_state, time_points)
        array[j] = xyz
        print(j)
    locals()[d] = array

# winsound.Beep(freq, duration)

# %%

for d in frames:
    val = locals()[d][0]
    x = pd.DataFrame(val)
    distance = [math.sqrt(a ** 2 + b ** 2 + c ** 2) for a, b, c in zip(x[0], x[1], x[2])]

    fig = plt.figure(figsize=(22, 12))
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.plot(x[0], x[1], x[2], color='g', alpha=1.0, linewidth=1.6)
    ax.set_title('Lorenz attractor phase diagram' + d)
    ax.text2D(0.05, 0.95, str(steps) + " time steps", transform=ax.transAxes)
    plt.savefig('Phase Attractor- ' + str(steps) + '-' + d + '.jpg')
    plt.show()

# %% Lorenz System

start_time = 0
end_time = 100
n = 500
time_points = np.linspace(start_time, end_time, 500)
dist_list = []

for i in range(0, len(df2)):
    initial_state = [df2_temp.iloc[i, 2], df2_temp.iloc[i, 1], df2_temp.iloc[i, 0]]
    xyz = odeint(lorenz_system, initial_state, time_points)
    # extract the individual arrays of x, y, and z values from the array of arrays
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    distance = [math.sqrt((a) ** 2 + (b) ** 2 + c ** 2) for a, b, c in zip(x, y, z)]
    dist_list.append(distance)
    print(i)

    if i == (len(df2) - 1):
        time_points = np.linspace(start_time, end_time, end_time)
        initial_state = [df2_temp.iloc[i, 2], df2_temp.iloc[i, 1], df2_temp.iloc[i, 0]]
        xyz = odeint(lorenz_system, initial_state, time_points)
        # extract the individual arrays of x, y, and z values from the array of arrays
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        distance = [math.sqrt((a) ** 2 + (b) ** 2 + c ** 2) for a, b, c in zip(x, y, z)]
# winsound.Beep(freq, duration)

len(df2)
df2.head()
train = pd.DataFrame(dist_list)
train = train.replace([np.inf, -np.inf], np.nan)
train.head()

train = train.dropna()
len(label[0:1000])
train.to_csv('Label_Unscaled_Data.csv')

# %%

dist_list2 = dist_list[:]

dist_list2 = pd.DataFrame(dist_list2)

dist_list2.loc[:, 'Labels'] = labels.Label

conc = dist_list2[dist_list2['Labels'] == 1]
exclusive = dist_list2[dist_list2['Labels'] == 2]
friendship = dist_list2[dist_list2['Labels'] == 3]
relationship = dist_list2[dist_list2['Labels'] == 4]
sexual = dist_list2[dist_list2['Labels'] == 5]
rest = dist_list2[dist_list2['Labels'] != 5]
other = dist_list2[dist_list2['Labels'] == 6]

sexualmean = sexual.drop('Labels', axis=1).mean(axis=0)
restmean = rest.drop('Labels', axis=1).mean(axis=0)
concmean = conc.drop('Labels', axis=1).mean(axis=0)
excmean = exclusive.drop('Labels', axis=1).mean(axis=0)
frdmean = friendship.drop('Labels', axis=1).mean(axis=0)
relmean = relationship.drop('Labels', axis=1).mean(axis=0)
othermean = other.drop('Labels', axis=1).mean(axis=0)

data = [sexualmean, restmean]

# %% ************************ ORIGINAL DATA ********************************
columns = train.columns
x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled, columns=columns)


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train, labels, test_size=0.2, shuffle=True,
                                                            stratify=labels)

text_clf = Pipeline([('clf', MultinomialNB(alpha=0.01)),
                     ])
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
nb_chaos_original = metrics.classification_report(y_test_2, predicted)
print(nb_chaos_original)

text_clf = KNeighborsClassifier()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
knn_chaos_original = metrics.classification_report(y_test_2, predicted)
print(knn_chaos_original)

text_clf = LinearSVC()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
svc_chaos_original = metrics.classification_report(y_test_2, predicted)
print(svc_chaos_original)


onehot = pd.get_dummies(labels['Label'])
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train, onehot, test_size=0.2, shuffle=True,
                                                            stratify=onehot)

n_inputs, n_features, n_outputs = X_train_2.shape[0], X_train_2.shape[1], len(y_train)

X_train = np.expand_dims(X_train_2, axis=2)
X_test = np.expand_dims(X_test_2, axis=2)

y_train = y_train_2.astype('category')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])

model.summary()

model.fit(X_train, y_train_2, epochs=10, batch_size=20, verbose=1)

pred = model.predict(X_test)
model.evaluate(X_test, y_test_2)
pred_labels = np.argmax(pred, axis=1)
y_test_labels = y_test_2.idxmax(axis=1)
cnn_chaos_original = classification_report(y_test_labels, pred_labels)
print(cnn_chaos_original)

# %% ************************ UNDER SAMPLED DATA ********************************
# %% NB with Leibniz

columns = train.columns
x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled, columns=columns)


train_1, label_1 = make_imbalance(train, labels,
                                  sampling_strategy={0: 261, 1: 473, 2: 532, 3: 800, 4: 1000, 5: 650, 6: 275})


onehot = pd.get_dummies(label_1['Label'])

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_1, label_1, test_size=0.2, shuffle=True,
                                                            stratify=label_1)

text_clf = Pipeline([('clf', MultinomialNB(alpha=0.01)),
                     ])
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
nb_chaos_undersampled = metrics.classification_report(y_test_2, predicted)
print(nb_chaos_undersampled)

text_clf = KNeighborsClassifier()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
knn_chaos_undersampled = metrics.classification_report(y_test_2, predicted)
print(knn_chaos_undersampled)

text_clf = LinearSVC()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
svc_chaos_undersampled = metrics.classification_report(y_test_2, predicted)
print(svc_chaos_undersampled)


# %%
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_1, onehot, test_size=0.2, shuffle=True,
                                                            stratify=onehot)

n_inputs, n_features, n_outputs = X_train_2.shape[0], X_train_2.shape[1], len(y_train)

X_train = np.expand_dims(X_train_2, axis=2)
X_test = np.expand_dims(X_test_2, axis=2)

y_train = y_train_2.astype('category')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])

model.summary()

model.fit(X_train, y_train_2, epochs=10, batch_size=20, verbose=1)

pred = model.predict(X_test)
model.evaluate(X_test, y_test_2)
pred_labels = np.argmax(pred, axis=1)
y_test_labels = y_test_2.idxmax(axis=1)
cnn_chaos_undersampled = classification_report(y_test_labels, pred_labels)
print(cnn_chaos_undersampled)


# %% ************************ OVER SAMPLED DATA ********************************
# %% NB with Leibniz

columns = train.columns
x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled, columns=columns)

sm = SMOTE()

train_1, label_1 = make_imbalance(train, labels,
                                  sampling_strategy={0: 261, 1: 473, 2: 532, 3: 800, 4: 1000, 5: 650, 6: 275})

train_1, label_1 = sm.fit_resample(train_1.fillna(0), label_1)

onehot = pd.get_dummies(label_1['Label'])

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_1, label_1, test_size=0.2, shuffle=True,
                                                            stratify=label_1)

text_clf = Pipeline([('clf', MultinomialNB(alpha=0.01)),
                     ])
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
nb_chaos_oversampled = metrics.classification_report(y_test_2, predicted)
print(nb_chaos_oversampled)

text_clf = KNeighborsClassifier()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
knn_chaos_oversampled = metrics.classification_report(y_test_2, predicted)
print(knn_chaos_oversampled)

text_clf = LinearSVC()
text_clf.fit(X_train_2, y_train_2)
predicted = text_clf.predict(X_test_2)
svc_chaos_oversampled = metrics.classification_report(y_test_2, predicted)
print(svc_chaos_oversampled)


# %%
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_1, onehot, test_size=0.2, shuffle=True,
                                                            stratify=onehot)

n_inputs, n_features, n_outputs = X_train_2.shape[0], X_train_2.shape[1], len(y_train)

X_train = np.expand_dims(X_train_2, axis=2)
X_test = np.expand_dims(X_test_2, axis=2)

y_train = y_train_2.astype('category')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])

model.summary()

model.fit(X_train, y_train_2, epochs=10, batch_size=20, verbose=1)

pred = model.predict(X_test)
model.evaluate(X_test, y_test_2)
pred_labels = np.argmax(pred, axis=1)
y_test_labels = y_test_2.idxmax(axis=1)
cnn_chaos_oversampled = classification_report(y_test_labels, pred_labels)
print(cnn_chaos_oversampled)
