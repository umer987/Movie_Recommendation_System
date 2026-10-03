import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS FOR PROFILE CARD
# ==========================================
st.markdown("""
<style>
    /* Main profile card container */
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        display: flex;
        align-items: center;
        gap: 30px;
    }
    
    /* Profile picture */
    .profile-pic-container {
        flex-shrink: 0;
    }
    
    .profile-pic {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        border: 5px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    
    .profile-pic:hover {
        transform: scale(1.05);
    }
    
    /* Right side content */
    .profile-info {
        flex-grow: 1;
        color: white;
    }
    
    .profile-name {
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 5px 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 0.5px;
    }
    
    .profile-title {
        font-size: 16px;
        font-weight: 500;
        margin: 0 0 20px 0;
        opacity: 0.95;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Link buttons */
    .link-buttons {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    
    .link-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        color: white !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .link-btn:hover {
        background: rgba(255, 255, 255, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .profile-card {
            flex-direction: column;
            text-align: center;
            padding: 20px;
        }
        .link-buttons {
            justify-content: center;
        }
        .profile-name {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PROFILE CARD (TOP OF DASHBOARD)
# ==========================================
st.markdown("""
<div class="profile-card">
    <div class="profile-pic-container">
        <img src="https://github.com/umer987.png" alt="Profile" class="profile-pic">
    </div>
    <div class="profile-info">
        <h1 class="profile-name">Syed Muhammad Umer</h1>
        <p class="profile-title">Full Stack Developer (MERN) | AI/ML Engineer | React | Node.js | Python | TensorFlow</p>
        <div class="link-buttons">
            <a href="https://github.com/umer987" target="_blank" class="link-btn">
                🐙 GitHub
            </a>
            <a href="https://www.linkedin.com/in/syed-m-umer" target="_blank" class="link-btn">
                💼 LinkedIn
            </a>
            <a href="https://github.com/umer987/SYED_MUHAMMAD_UMER_RESUME" target="_blank" class="link-btn">
                📄 Resume
            </a>
            <a href="https://syed-m-umer.vercel.app/" target="_blank" class="link-btn">
                🌐 Portfolio
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA (cached for performance)
# ==========================================
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

@st.cache_resource
def build_content_model(movies):
    """Build TF-IDF matrix and cosine similarity once."""
    movies = movies.copy()
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Title -> index mapping
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    return movies, cosine_sim, indices

# ==========================================
# RECOMMENDER FUNCTIONS
# ==========================================
def get_content_recommendations(title, movies, cosine_sim, indices, n=10):
    try:
        idx = indices[title]
    except KeyError:
        return pd.DataFrame()
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    result = movies.iloc[movie_indices][['title', 'genres']].copy()
    result['similarity'] = scores
    return result

def get_popular_movies(movies, ratings, min_ratings=50, n=10):
    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    popular = movies.merge(stats, on='movieId')
    popular = popular[popular['num_ratings'] >= min_ratings]
    popular = popular.sort_values('avg_rating', ascending=False)
    
    return popular[['title', 'genres', 'avg_rating', 'num_ratings']].head(n)

# ==========================================
# LOAD DATA
# ==========================================
with st.spinner("Loading movie data..."):
    movies, ratings = load_data()
    movies, cosine_sim, indices = build_content_model(movies)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Choose a recommender:",
    ["🏠 Home", "🎭 Content-Based", "🔥 Popular Movies", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset Stats**")
st.sidebar.markdown(f"- 🎥 Movies: `{len(movies):,}`")
st.sidebar.markdown(f"- ⭐ Ratings: `{len(ratings):,}`")
st.sidebar.markdown(f"- 👥 Users: `{ratings['userId'].nunique():,}`")

# ==========================================
# PAGE: HOME
# ==========================================
if page == "🏠 Home":
    st.title("🎬 Movie Recommendation System")
    st.markdown("""
    Welcome! This app recommends movies using **machine learning**.
    
    ### 🚀 How to use
    1. **Content-Based** → Pick a movie you like, get similar movies by genre.
    2. **Popular Movies** → See the highest-rated movies in the dataset.
    
    ### 🛠️ Built with
    - **Python** & **Pandas** for data handling
    - **Scikit-learn** (TF-IDF + Cosine Similarity) for recommendations
    - **Streamlit** for the interactive web UI
    """)
    
    st.markdown("---")
    st.subheader("🔥 Top 5 Popular Movies Right Now")
    top5 = get_popular_movies(movies, ratings, n=5)
    for i, row in top5.iterrows():
        st.markdown(f"**{row['title']}**  \n`{row['genres']}` — ⭐ {row['avg_rating']:.2f} ({row['num_ratings']} ratings)")

# ==========================================
# PAGE: CONTENT-BASED
# ==========================================
elif page == "🎭 Content-Based":
    st.title("🎭 Content-Based Recommender")
    st.markdown("Pick a movie and we'll find similar ones based on **genre similarity**.")
    
    # Movie selector
    movie_list = sorted(movies['title'].dropna().unique().tolist())
    selected_movie = st.selectbox(
        "Choose a movie:",
        movie_list,
        index=movie_list.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_list else 0
    )
    
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("🎯 Recommend", type="primary"):
        with st.spinner("Finding similar movies..."):
            recs = get_content_recommendations(
                selected_movie, movies, cosine_sim, indices, n=num_recs
            )
        
        if recs.empty:
            st.error("Movie not found. Try another one.")
        else:
            st.success(f"Found {len(recs)} movies similar to **{selected_movie}**")
            
            # Display as cards
            for _, row in recs.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {row['title']}")
                        st.caption(f"🎭 {row['genres'].replace('|', ', ')}")
                    with col2:
                        st.metric("Similarity", f"{row['similarity']*100:.1f}%")

# ==========================================
# PAGE: POPULAR MOVIES
# ==========================================
elif page == "🔥 Popular Movies":
    st.title("🔥 Popular Movies")
    st.markdown("Highest-rated movies, filtered by minimum number of ratings.")
    
    min_ratings = st.slider("Minimum number of ratings:", 10, 200, 50, step=10)
    n_movies = st.slider("How many to show:", 5, 50, 15)
    
    popular = get_popular_movies(movies, ratings, min_ratings=min_ratings, n=n_movies)
    
    st.markdown(f"### Top {n_movies} Movies (min {min_ratings} ratings)")
    
    # Show as a styled dataframe
    display_df = popular.copy()
    display_df['avg_rating'] = display_df['avg_rating'].round(2)
    display_df.columns = ['Title', 'Genres', 'Avg Rating', 'Num Ratings']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Bar chart
    st.markdown("### 📊 Ratings Comparison")
    chart_data = popular.set_index('title')['avg_rating']
    st.bar_chart(chart_data)

# ==========================================
# PAGE: ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### 🎯 What is a Recommendation System?
    A system that predicts what a user might like based on data.
    
    ### 🧠 Techniques Used
    
    **1. Content-Based Filtering**
    - Uses **TF-IDF** to convert genres into numerical vectors.
    - Uses **Cosine Similarity** to find movies with similar genres.
    - No user data needed — works on item features only.
    
    **2. Popularity-Based Filtering**
    - Aggregates ratings to find highly-rated movies.
    - Filters out movies with few ratings (avoids "1 rating = 5.0 stars" bias).
    
    ### 📊 Dataset
    **MovieLens Latest Small** by GroupLens Research.
    - 100,000 ratings from 600 users on 9,000 movies.
    
    ### 🔮 Future Improvements
    - Add **Collaborative Filtering** (user-based & item-based)
    - Use movie **tags** for richer content features
    - Add **movie posters** using the `links.csv` (TMDB API)
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("🎬 Movie Recommender | Built with Streamlit & Scikit-learn")