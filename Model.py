import pandas as pd
import numpy as np
from recommendation import AbstractRecommender

class DotProductRecommender(AbstractRecommender):
    def __init__(self, content_embed):
        self.count = {}
        self.censusdict = {}
        self.sumembedding = {}
        self.avgembedding = {} 
        self.content_embed = pd.read_csv(content_embed, index_col = 0)
        self.last_page_viewed = {}

    def observe(self, user_interaction):
        user_id = user_interaction["USER_ID"]
        url_path = user_interaction["URL_PATH"]
        new_url = url_path.replace("/en", "").replace("/es-mx", "")
        census_key = user_interaction["CENSUS_KEY"]
        
        self.last_page_viewed[user_id] = new_url
        
        if new_url in self.content_embed.index:
            content_embedding = self.content_embed.loc[new_url]
        else: 
            content_embedding = np.zeros(32)
        
        if user_id in self.count.keys(): 
            self.count[user_id] += 1
        else: self.count[user_id] = 1
        
        self.censusdict[user_id] = census_key
        
        if user_id in self.sumembedding.keys(): 
            self.sumembedding[user_id] += content_embedding
        else: self.sumembedding[user_id] = content_embedding
 
        self.avgembedding[user_id] = self.sumembedding[user_id] / self.count[user_id]
    
    def recommend(self, user_id, n):
        try:
            # dont recommend most recent viewed page
            non_self = self.content_embed[self.content_embed.index != self.last_page_viewed[user_id]]
            # take the dot product of average user content embeddings to all possible content embeddings
            dot_products = non_self @ self.avgembedding[user_id]
            # return top n results
            recs = dot_products.sort_values(ascending=False).index[:n].tolist()
            return recs
        except KeyError:
            return n * ["/en"]
        
if __name__ == 'main':
    pageview = pd.read_csv('data/pageview.csv', error_bad_lines=False)
    dpr = DotProductRecommender('data/content_embeddings.csv')
    # record all historical pageview observations
    for i in range(len(pageview)):
        if (i % 10000) == 0:
            print(i)
        dpr.observe(pageview.iloc[i])
    # save model
    import pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(dpr, f)