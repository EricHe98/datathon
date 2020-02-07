import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import Word2Vec
import warnings 
import sys
warnings.filterwarnings(action = 'ignore') 

# accept filenames as arguments
pageviews_file = sys.argv[1]
census_file = sys.argv[2]
content_file = sys.argv[3]

print('Reading data')
pageview = pd.read_csv(pageviews_file, sep=',', error_bad_lines=False)
census = pd.read_csv(census_file, sep=',', error_bad_lines=False)
content = pd.read_csv(content_file, sep=',', error_bad_lines=False)

# select features from original dataset
print('Processing features')
import string
content_texts = content[['title', 'article_content', 'meta_title', 'meta_description', 'meta_keywords']]
content['merged'] = content_texts.astype(str).apply(' '.join, 1).str.lower()
content['merged'] = content['merged'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
content['merged_words'] = content['merged'].apply(lambda x: x.split(' '))

# train model
print('Training model')
word2vec = Word2Vec(content['merged_words'], min_count=3, size=32, workers=3, window=3, sg=0) # CBOW

# extract embeddings
print('Extracting embeddings')
word_list = list(word2vec.wv.vocab.keys())
word_embeddings = pd.Series(word_list, index=word_list).apply(lambda x: word2vec[x])
df = pd.DataFrame.from_dict(word_embeddings.to_dict()).T
colnames = ['embed_' + str(i) for i in range(32)]
df.columns = colnames

print('Writing embeddings to csv')
df.to_csv('data/w2v_df.csv')