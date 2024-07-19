import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD


# Load MovieLens data (replace with your data path)
ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
movies_df = pd.read_csv('./ml-latest-small/movies.csv')

# Merge ratings and movies data
data = pd.merge(ratings_df, movies_df, on='movieId')

# Create Surprise Dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Create TF-IDF matrix for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(movie_title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


# Content-based recommendation
recommendations = content_based_recommendations('Toy Story')
print(recommendations)
