# Multi-Feature Movie Recommender System 🎬

A Streamlit-based web application that recommends movies using Natural Language Processing (NLP) and Cosine Similarity. This project demonstrates data engineering with **Pandas** and mathematical vectorization with **Scikit-Learn**.

##  Features
* **Multi-Way Search:** Search by Movie Title or Actor/Director name.
* **Feature Engineering:** Combines Overview, Cast (Top 3), and Director into a unified "Tag" system for better accuracy.
* **Vectorization:** Utilizes `TfidfVectorizer` to convert text into mathematical vectors.
* **Cosine Similarity:** Calculates the angular distance between movie vectors to find the most similar content.

##  Tech Stack
* **Language:** Python 3.11
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (TF-IDF, Cosine Similarity)
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

##  Learning Outcomes
* Implemented the **Top-Down** approach to learning NLP.
* Mastered **Data Cleaning** techniques like removing spaces from tags to prevent "Christian Wood" vs "Christian Bale" confusion.
* Utilized **Pickle** for model serialization to ensure high-performance backend delivery.
