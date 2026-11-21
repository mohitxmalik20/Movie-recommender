from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Movie Recommender API")

# 2. Load the precomputed data into memory when the server starts
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Ensure 'movie_dict.pkl' and 'similarity.pkl' are in the same folder.")


# 3. Create the recommendation endpoint
@app.get("/recommend")
def recommend_movie(title: str):
    try:
        # Find the index of the movie in the DataFrame
        #movie_index = movies_df[movies_df['title'] == title].index[0]
        movie_index = movies_df[movies_df['title'].str.lower() == title.strip().lower()].index[0]
        # Fetch the similarity distances for that specific movie
        distances = similarity[movie_index]

        # Sort and grab the top 5 most similar movies (ignoring index 0, which is the movie itself)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        # Extract just the titles
        recommended_movies = []
        for i in movie_list:
            recommended_movies.append(movies_df.iloc[i[0]].title)

        return {
            "success": True,
            "query": title,
            "recommendations": recommended_movies
        }


    except IndexError:
        raise HTTPException(
            status_code=404,
            detail="No movies to recommend from this title."
        )