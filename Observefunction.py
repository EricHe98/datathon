import pandas as pd
from recommendation import AbstractRecommender

class ExampleRecommender(AbstractRecommender):
    def __init__(self, content_embed):
        self.last_page_viewed = {}
        self.count = {}
        self.censusdict = {}
        self.sumembedding = {}
        self.avgembedding = {} 
        self.content_embed = pd.read_csv(content_embed, index_col = 0)

    def observe(self, user_interaction):
        user_id = user_interaction["USER_ID"]
        url_path = user_interaction["URL_PATH"]
        new_url = url_path.replace("/en", "").replace("/es-mx", "")
        census_key = user_interaction["CENSUS_KEY"]

        self.last_page_viewed[user_id] = url_path
        
        if user_id in self.count.keys(): self.count[user_id] += 1
        else: self.count[user_id] = 1
        
        self.censusdict[user_id] = census_key
        
        if user_id in self.sumembedding.keys(): 
            self.sumembedding[user_id] += self.content_embed.loc[new_url]
        else: self.sumembedding[user_id] = self.content_embed.loc[new_url]
 
        self.avgembedding[user_id] = self.sumembedding[user_id] / self.count[user_id]
    
    def recommend(self, user_id, n):
        try:
            return n * [self.last_page_viewed[user_id]]
        except KeyError:
            return n * ["/en"]