# -*- mode: python; coding: utf-8; -*-
from recommendation import AbstractRecommender

import pandas as pd


class ExampleRecommender(AbstractRecommender):
    def __init__(self):
        self.last_page_viewed = {}

    def observe(self, user_interaction):
        user_id = user_interaction["user_id"]
        url_path = user_interaction["url_path"]
        self.last_page_viewed[user_id] = url_path

    def recommend(self, user_id, n):
        try:
            return n * [self.last_page_viewed[user_id]]
        except KeyError:
            return n * ["/en"]


if __name__ == "__main__":
    user_interactions = pd.DataFrame(
        dict(
            row_num=range(4),
            user_id=["a", "b", "c", "a"],
            url_path=["/en/z", "/en/x", "/en", "/en/x"],
            census_key=["p", "q", "r", "p"],
        )
    )
    print("Test user interaction data:")
    print(user_interactions)
    print()

    n = 3
    print(f"Example site-visit and recommendation sequence:")
    recommender = ExampleRecommender()
    for _, s in user_interactions.iterrows():
        recommendations = recommender.recommend(s["user_id"], n)
        print(f"- User `{s['user_id']}` visits at time {s['row_num']}:")
        print(f"  - Recommend {recommendations}.")
        print(f"  - User visits {s['url_path']}.")
        recommender.observe(s)