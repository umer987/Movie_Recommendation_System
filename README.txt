<div align="center">

# 🎬 Movie Recommendation System

### An Interactive ML-Powered Movie Recommender built with Streamlit & Scikit-learn

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<img src="https://github.com/umer987.png" alt="Profile" width="120" style="border-radius: 50%;" />

**Built by [Syed Muhammad Umer](https://github.com/umer987)**

Full Stack Developer (MERN) | AI/ML Engineer | React | Node.js | Python | TensorFlow

[![GitHub](https://img.shields.io/badge/GitHub-umer987-181717?style=for-the-badge&logo=github)](https://github.com/umer987)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed_M_Umer-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/syed-m-umer)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-000000?style=for-the-badge&logo=vercel)](https://syed-m-umer.vercel.app/)
[![Resume](https://img.shields.io/badge/Resume-Download-2ea44f?style=for-the-badge&logo=github)](https://github.com/umer987/SYED_MUHAMMAD_UMER_RESUME)

</div>

---

## 📖 Overview

A **Movie Recommendation System** that suggests movies using Machine Learning. It uses the **MovieLens Latest Small** dataset and implements both **Content-Based Filtering** and **Popularity-Based Filtering** techniques.

The app is built as an interactive **Streamlit web dashboard** with a modern gradient UI, allowing users to:
- 🎭 Find movies similar to one they like
- 🔥 Discover the highest-rated movies
- 📊 Visualize ratings with interactive charts

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 🎭 **Content-Based Recommender** | Recommends movies by computing genre similarity using TF-IDF and Cosine Similarity |
| 🔥 **Popularity-Based Recommender** | Shows top-rated movies filtered by a minimum number of ratings |
| 📊 **Interactive Charts** | Bar charts comparing average ratings of popular movies |
| 🎨 **Beautiful UI** | Gradient profile card, card-based movie display, glassmorphism buttons |
| ⚡ **Fast Performance** | Streamlit caching prevents recomputation of similarity matrices |
| 📱 **Responsive Design** | Works seamlessly on desktop, tablet, and mobile |

---

## 🧠 How It Works

### 1. Content-Based Filtering
```
Movie Genres → TF-IDF Vectorization → Cosine Similarity Matrix → Top-N Similar Movies
```
- **TF-IDF** converts each movie's genres into a numerical vector
- **Cosine Similarity** measures how similar two movies are (0 to 1)
- Returns the top-N most similar movies to the one you selected

### 2. Popularity-Based Filtering
```
Ratings.csv → Group by movieId → Average rating + Count → Filter + Sort → Top-N Popular
```
- Aggregates all user ratings per movie
- Filters out movies with too few ratings (avoids bias)
- Sorts by average rating to find the best movies

---

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (TF-IDF, Cosine Similarity) |
| **Web Framework** | Streamlit |
| **Dataset** | MovieLens Latest Small (GroupLens) |

---

## 📊 Dataset

The project uses the **MovieLens Latest Small** dataset by [GroupLens Research](https://grouplens.org/datasets/movielens/).

| File | Size | Description |
| :--- | :--- | :--- |
| `movies.csv` | 483 KB | Movie IDs, titles, and genres |
| `ratings.csv` | 2.4 MB | User ratings with timestamps |
| `tags.csv` | 116 KB | User-generated tags |
| `links.csv` | 194 KB | IMDB/TMDB links |

**Stats:**
- 🎥 **9,000+** Movies
- ⭐ **100,000** Ratings
- 👥 **600+** Users

---

## 📁 Project Structure

```
movie_recommender/
│
├── app.py                  # Main Streamlit application
├── movies.csv              # Movie data
├── ratings.csv             # User ratings
├── tags.csv                # User tags
├── links.csv               # External links
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/umer987/movie-recommender.git
cd movie-recommender
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas scikit-learn
```

### 4. Download the Dataset

Download **MovieLens Latest Small** from [GroupLens](https://grouplens.org/datasets/movielens/) and place the CSV files in the project root folder.

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` 🎉

---

## 📸 Screenshots

### 🏠 Home Page
> Gradient profile card with clickable GitHub, LinkedIn, Resume, and Portfolio links.

### 🎭 Content-Based Recommender
> Select a movie → get similar movies with similarity scores.

### 🔥 Popular Movies
> Interactive table + bar chart of top-rated movies.

---

## 📋 Requirements

Create a `requirements.txt` file with the following:

```txt
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

## 🔮 Future Improvements

- [ ] **Collaborative Filtering** (User-based & Item-based)
- [ ] **Hybrid Recommender** (Content + Collaborative)
- [ ] **Movie Posters** using TMDB API via `links.csv`
- [ ] **User Authentication** to save personal watchlists
- [ ] **Search Bar** for easier movie selection
- [ ] **Deploy on Streamlit Cloud**

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Connect With Me

<div align="center">

| Platform | Link |
| :---: | :--- |
| 🐙 **GitHub** | [github.com/umer987](https://github.com/umer987) |
| 💼 **LinkedIn** | [linkedin.com/in/syed-m-umer](https://www.linkedin.com/in/syed-m-umer) |
| 🌐 **Portfolio** | [syed-m-umer.vercel.app](https://syed-m-umer.vercel.app/) |
| 📄 **Resume** | [Download](https://github.com/umer987/SYED_MUHAMMAD_UMER_RESUME) |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for the ML tools
- [Shields.io](https://shields.io/) for the badges

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

**Made with ❤️ by [Syed Muhammad Umer](https://github.com/umer987)**

</div>