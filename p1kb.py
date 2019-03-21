
import shelve
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

from string import punctuation
translator = str.maketrans(' ', ' ', punctuation)
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def normalize_text(doc):
    tokens = []
    for sent in doc.sents:
        sent = str(sent)
        sent = sent.replace('\r', ' ').replace('\n', ' ')
        lower = sent.lower()
        nopunc = lower.translate(translator)
        words = nopunc.split()
        nostop = [w for w in words if w not in stoplist]
        no_numbers = [w if not w.isdigit() else '#' for w in nostop]
        stemmed = [stemmer.stem(w) for w in no_numbers]
        tokens += stemmed
    return tokens

df = pd.DataFrame(index = range(0,0), columns=['jahr',  'rev', 'nsents', 'nwords', 'nlets', 'nnouns', 'nverbs', 'nadjes', 'tgs'], dtype = int)
df['tgs'] = df['tgs'].astype('object')

fpath = '/home/xhta/Robot/cases'

lrev = []	#list of reversed cases
lnrev = []	#list of not reversed cases

print ("reading in cases_reversed.txt")
with open ("/home/xhta/Robot/cases_reversed.txt") as fcrev:
    readCSV = csv.reader(fcrev, delimiter = ',')
    next(readCSV, None)   # skip 1 line   don't skip in case of 96  only skip in the orig
    for Zei in readCSV:
        if (Zei[1] == '0'): lnrev.append(Zei[0])
        else: lrev.append(Zei[0])

print ("done reading in cases_reversed.txt")

n_samples = 1000	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

n_rev = len(lrev)
n_nrev = len(lnrev)
n_all = n_rev + n_nrev

n_samples_rev = int(np.ceil(n_samples * n_rev/n_all))
n_samples_nrev = n_samples - n_samples_rev

samples_rev_caseid = np.random.choice(lrev, n_samples_rev, replace = False)
samples_nrev_caseid = np.random.choice(lnrev, n_samples_nrev, replace = False)

for caseid_nrev in samples_nrev_caseid:
    df.loc[caseid_nrev] = [0, 0, 0, 0, 0, 0, 0, 0, [ ]]

for caseid_rev in samples_rev_caseid:
    df.loc[caseid_rev] = [0, 1, 0, 0, 0, 0, 0, 0, [ ]]

print ("done inserting caseid and rev into dataframe")

from os import listdir
from os.path import isfile, join

fnames = [f for f in listdir(fpath) if isfile(f)]

ldir = listdir(fpath)

import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

nlp2 = spacy.load('en_core_web_sm')

from nltk import ngrams
from collections import Counter
term_frequencies = Counter()

totalgrams = []

i = 0
for fname in ldir:
    lae = len(fname)
    cname = fname[5:(lae-4)]
    year = fname[0:4]
    if (not (cname in df.index)): continue
    #if(len(fname) > 16): print(fname , cname, year)
    if ( i% 199 ==0):
        #print(df.loc[cname])
        print(datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    df.at[cname, 'jahr'] =  year
    if ( i% 199 ==0):
        #print(cname, year)
        print(df.loc[cname, 'jahr'])
    i = i + 1
    fna2 = join(fpath, year + '_' + cname + '.txt')
    rawtext = open(fna2).read()
    doc = nlp(rawtext)
    df.at[cname, 'nlets'] =  len(rawtext)
    sentences = [sent.string.strip() for sent in doc.sents]
    df.at[cname, 'nsents'] =  len(sentences)
    df.at[cname, 'nwords'] = len([token for token in doc if not token.is_punct])
    doc2 = nlp2(rawtext)
    df.at[cname, 'nnouns'] = len([w for w in list(doc2) if w.tag_.startswith('N')])
    df.at[cname, 'nverbs'] = len([w for w in list(doc2) if w.tag_.startswith('V')])
    df.at[cname, 'nadjes'] = len([w for w in list(doc2) if w.tag_.startswith('J')])
    cltoks = normalize_text(doc2)
    ntoks = [str(token).lower() for token in list(doc2) if (token.tag_.startswith('N')) & (not token.is_punct) & (not token.is_space) & (not token.is_stop) & (str(token) in cltoks)]
    trigrams = ngrams(cltoks, 3)
    grams = []
    for t in trigrams:
        lt = list(t)
        if(lt[2] in ntoks): grams += ['_'.join(t)]
    df.at[cname, 'tgs'] = grams
    totalgrams += grams
    term_frequencies.update(cltoks)

je = datetime.now()
pkl_fname = 'df_1k.' + datetime.now().strftime('%Y%m%d_%H%M%S' + ".pkl")
#df.to_pickle(pkl_fname)

df['nsents'].hist(bins=50)	# histogram over number of sentences
df['nsents'].plot(kind='kde', style = 'k--', color = 'g', alpha = 0.3)
df['nwords'].hist(bins=50)	# histogram over number of words
df['nwords'].plot(kind='kde', style = 'k--')
df['nlets'].hist(bins=50)       # histogram over number of letters
df['nlets'].plot(kind='kde', style = 'k--')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax = df.groupby('jahr')['nnouns'].sum().plot()
ax.set_ylabel('Total number of nouns')
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)
ax2 = df.groupby('jahr')['nverbs'].sum().plot()
ax2.set_ylabel('Total number of verbs')
ax2.set_xlim([1900, 2015])
plt.show()

ax = df.groupby('jahr')['nadjes'].sum().plot()
ax.set_ylabel('Total number of adjectives')
ax.set_xlim([1900, 2015])
plt.show()


n_feats = 1000 		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from collections import Counter
tgc = Counter(totalgrams)
mc = tgc.most_common()[0:n_feats]
lmc = len(mc)

feats = ['rev']
for j in range(lmc):
    tu = tgc.most_common()[j]
    fea = tu[0]
    feats.append(fea)

df3 = pd.DataFrame(index = df.index, columns=feats, dtype = int)
df3['rev'] = df['rev']
df3 = df3.astype({"rev": int})
pkl_fname = 'df3_1k.' + datetime.now().strftime('%Y%m%d_%H%M%S' + ".pkl")
df3.to_pickle(pkl_fname)

for loca in df.index:
    tgs = df.loc[loca, 'tgs']
    for feat in feats:
        if (feat == 'rev'): continue
        df3.at[loca, feat] = tgs.count(feat)
        
 # elapse time  starttime = time.time()     elapsed time = time.time() - starttime


from sklearn import preprocessing

target = df3['rev']
features = df3.loc[:,df3.columns != 'rev']

features_scaled = preprocessing.scale(features, with_mean=False)
msk = np.random.rand(len(df3)) < 0.8

# training
targeta = target[msk]
feata = features_scaled[msk]

# test
targete = target[~msk] 
feate = features_scaled[~msk]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 123, solver = 'liblinear').fit(feata, targeta)

ypreda = clf.predict(feata)
yprede = clf.predict(feate)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
accuracy_score(targeta, ypreda)  # accuracy on training data
accuracy_score(targete, yprede)  # accuracy on test data
f1_score(targeta, ypreda)  # F1 on training data
f1_score(targete, yprede)  # F1 on test data
confusion_matrix(targeta, ypreda)  # cm on training data
confusion_matrix(targete, yprede)  # cm on test data

#from sklearn import grid_search, # deprecated
from sklearn import model_selection
parameters = {'penalty':('l1', 'l2'), 'C':[1, 2, 5, 10]}

#clf2 = grid_search.GridSearchCV(clf, parameters)
#clf2.fit(feata, targeta)
#clf2.best_params_

clf3 = model_selection.GridSearchCV(clf, parameters)
clf3.fit(feata, targeta)
clf3.best_params_

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#for fname in ldir[0:10]:
for fname in ldir:
    lae = len(fname)
    cname = fname[5:(lae-4)]
    year = fname[0:4]
    if ( i% 59 ==0):
        print(datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    i = i + 1
    fna2 = join(fpath, year + '_' + cname + '.txt')
    rawtext = open(fna2).read()
    #doc = nlp(rawtext)
    #doc2 = nlp2(rawtext)
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(rawtext)
#    for k in sorted(ss):
#        print('{0}:{1}, '.format(k, ss[k]), end='')
#    print()


from sklearn.feature_extraction.text import TfidfVectorizer

dfvec = pd.DataFrame(index = range(0,0), columns=['jahr',  'rev', 'text', 'tgs'], dtype = int)
dfvec['jahr'] = df['jahr']
dfvec['rev'] = df['rev']
dfvec['tgs'] = df['tgs']
dfvec['text'] = dfvec['text'].astype('object')
dfvec['rev'] = dfvec['rev'].astype('int')

i = 0
for fname in ldir:
    lae = len(fname)
    cname = fname[5:(lae-4)]
    year = fname[0:4]
    if (not (cname in df.index)): continue
    if ( i% 179 ==0):
        print(datetime.now().strftime('%Y%m%d_%H:%M:%S'))
    i = i + 1
    fna2 = join(fpath, year + '_' + cname + '.txt')
    rawtext = open(fna2).read()
    #nlp3 = spacy.load('en')
    #doc = nlp3(rawtext)
    #sentences = [sent.string.strip() for sent in doc.sents]
    #doc2 = nlp2(rawtext)
    dfvec.at[cname, 'text'] = rawtext

vec = TfidfVectorizer(min_df = 0.001, max_df = 0.8, max_features = 4000, stop_words = 'english', use_idf = True, ngram_range = (1,3))
vec = TfidfVectorizer(min_df = 0.0001, max_df = 0.4, max_features = 40000, stop_words = 'english')
trainedVec = vec.fit_transform (dfvec['text'])

from sklearn.metrics.pairwise import cosine_similarity

cs = cosine_similarity(trainedVec[0:1000], trainedVec)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 10, n_jobs = -1) 

km.fit(cs)


