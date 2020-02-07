import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords 
from gensim.models import Word2Vec
import warnings 
import sys
warnings.filterwarnings(action = 'ignore') 

# accept filenames as arguments
# only the content file is required
content_file = sys.argv[1]

print('Reading data from file {}'.format(content_file))
content = pd.read_csv(content_file, sep=',', error_bad_lines=False)

# select features from original dataset
print('Processing features')
import string
content_texts = content[['title', 'article_content', 'meta_title', 'meta_description', 'meta_keywords']]
content['merged'] = content_texts.astype(str).apply(' '.join, 1).str.lower()
content['merged'] = content['merged'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# remove stopwords
stop_words = stopwords.words('english')
content['merged_words'] = content['merged'].apply(lambda x: [word for word in x.split(' ') if not word in stop_words])

content['merged_words'] = content['merged'].apply(lambda x: x.split(' '))

# train model
print('Training model')
word2vec = Word2Vec(content['merged_words'], min_count=3, size=32, workers=3, window=3, sg=0) # CBOW

# extract embeddings
print('Extracting word embeddings')
word_list = list(word2vec.wv.vocab.keys())
word_embeddings = pd.Series(word_list, index=word_list).apply(lambda x: word2vec[x])
df = pd.DataFrame.from_dict(word_embeddings.to_dict()).T
colnames = ['embed_' + str(i) for i in range(32)]
df.columns = colnames

# extract content embeddings
print('Computing content embeddings')
content_embeddings = content.apply(lambda row: df.loc[row['merged_words']].mean(axis=0), axis=1)
content_embeddings = content_embeddings.set_index(content['url'])

words_output_file = 'data/w2v_df.csv'
content_output_file = 'data/content_embeddings.csv'
print('Writing word embeddings to csv at location {}'.format(words_output_file))
df.to_csv(words_output_file)
print('Writing content embeddings to csv at location {}'.format(content_output_file))
content_embeddings.to_csv(content_output_file) 