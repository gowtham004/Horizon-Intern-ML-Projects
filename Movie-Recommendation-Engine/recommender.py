import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load & Merge
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

# 2. Data Cleaning Functions
def convert_generic(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'].replace(" ", ""))
    return L

def get_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'].replace(" ", ""))
            break 
    return L

# 3. Apply Cleaning
movies['genres'] = movies['genres'].apply(convert_generic)
movies['keywords'] = movies['keywords'].apply(convert_generic)
movies['cast'] = movies['cast'].apply(lambda x: convert_generic(x)[:3]) # Top 3 actors
movies['director'] = movies['crew'].apply(get_director) # Get director from crew

# 4. Create "Tags"
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# 5. Vectorization (The Math)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vector = tfidf.fit_transform(movies['tags']).toarray() 
cosine_sim = cosine_similarity(vector)

# 6. Save for Streamlit
pickle.dump(movies, open('movie_list.pkl', 'wb'))
pickle.dump(cosine_sim, open('similarity.pkl', 'wb'))

print("Setup Complete! Run your Streamlit app now.")