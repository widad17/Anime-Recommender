import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

anime = pd.read_parquet('anime/anime.parquet')
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(anime['Mod_name'])


def search(query, vectorizer):
    processed_query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed_query])
    cosine_sim = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(cosine_sim, -10)[-10:]
    result = anime.iloc[indices].sort_values("Popularity", ascending=True)
    return result


if __name__ == "__main__":
    print(search("One Piece", vectorizer))



