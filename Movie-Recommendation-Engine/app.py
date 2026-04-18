import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.set_page_config(page_title="Pro Movie Engine", page_icon="🔍")
st.title('Multi-Feature Movie Search')

# 1. Search Mode Selection
option = st.radio(
    "How would you like to search?",
    ('Movie Title', 'Actor/Director Name'),
    horizontal=True
)

def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    recs = []
    for i in distances[1:7]:
        recs.append(movies.iloc[i[0]].title)
    return recs

# 2. Logic for Movie Title Search
if option == 'Movie Title':
    selected_movie = st.selectbox('Select a movie:', movies['title'].values)
    if st.button('Find Similar Movies'):
        results = recommend(selected_movie)
        for r in results:
            st.write(f"🎬 {r}")

# 3. Logic for Actor/Director Search
else:
    search_query = st.text_input("Enter Actor or Director Name (e.g., Christian Bale or Christopher Nolan)").lower().replace(" ", "")
    if st.button('Search'):
        # We look for movies where the 'tags' contain the name
        mask = movies['tags'].str.contains(search_query)
        results = movies[mask]['title'].values
        
        if len(results) > 0:
            st.success(f"Found {len(results)} movies:")
            for r in results[:10]: # Show top 10 matches
                st.write(f"🎥 {r}")
        else:
            st.error("No matches found. Try checking the spelling or remove spaces.")