import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv( "tmdb_5000_movies.csv")

credits_df.columns = ['id','tittle','cast','crew']
movies_df = movies_df.merge(credits_df, on="id")

# Demographic Filtering

C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

print("C: ", C)
print("m: ", m)

new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print(new_movies_df.shape)

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)

new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)

new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)

    
tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
# print("\ntfidf matrix:\n",tfidf_matrix,"\n----\n")

# Compute similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("cosine similarity:\n",cosine_sim,"\n---\n")

indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
print("indices are:\n")
print(indices)
print("---")

def get_recommendations(title, cosine_sim=cosine_sim):
    #print("title:", title)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    print("Movies are: ", movies)
    return movies.to_json()





