from collab_based_rec import *

fav_anime_list = [1, 21, 40748, 10629, 16498, 18397]

if __name__ == "__main__":
    print(fav_anime_list)
    rating_matrix, interactions = create_recommendation_matrix(liked_anime_list, anime_percentage=0.8)
    anime_recs = create_anime_recommendation_df(rating_matrix, interactions)
    print(recommend_anime(anime_recs))
