# -*- coding: utf-8 -*-
"""TestRank_Text_Summarization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Eqt_mk69qh5XfM20qqlGDZK8CzsOtTat

# Text Summarization using text rank algorithm

# Importing the libraries
"""

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
import re

"""# Uploading Data"""

# Upload the CSV file
from google.colab import files
uploaded = files.upload()

# Read the CSV file
import io
df = pd.read_csv(io.StringIO(uploaded['tennis_articles_v4.csv'].decode("utf-8")))

df.head()

"""# Preprocessing the Data"""

# split the the text in the articles into sentences
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

# flatten the list
sentences = [y for x in sentences for y in x]
# for x in sentences:
#   for y in x:
#     sentences.append(y)

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

nltk.download('stopwords')# one time execution

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

# for i in sen:
#   if i not in stop_words:
#     sen_new += i

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

"""# Downloading the word embeddings"""

# download pretrained GloVe word embeddings
! wget http://nlp.stanford.edu/data/glove.6B.zip

! unzip glove*.zip

"""# Extracting word vectors"""

# Extract word vectors
word_embeddings = {} # example {fox:.822455, .....}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

"""# Formation of sentence vectors"""

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

len(sentence_vectors)

"""# Finding similarities by using cosine similarity
The next step is to find similarities among the sentences. We will use cosine similarity to find similarity between a pair of sentences. Let's create an empty similarity matrix for this task and populate it with cosine similarities of the sentences.
"""

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

"""# Forming Graph from similarity Matrix"""

import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

"""# Sorting and printing Summary"""

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Specify number of sentences to form the summary
sn = 10

# Generate summary
for i in range(sn):
  print(f"{i}.{ranked_sentences[i][1]}")

"""# End"""