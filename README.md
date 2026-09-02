
# Movie Recommendation System

This project implements a movie recommendation system using data from Rotten Tomatoes. The system utilizes a K-Nearest Neighbors (KNN) algorithm based on cosine similarity to provide personalized movie recommendations based on user reviews and ratings.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Functionality](#functionality)
- [License](#license)

## Project Description

The movie recommendation system processes movie and review data from Rotten Tomatoes, normalizes the scores from various formats, and constructs a user-item review matrix. It employs a KNN model to suggest movies similar to a specified title based on critic reviews.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```

2. **Install required packages**:
   Make sure you have Python installed. Then, you can install the necessary libraries using pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. **Download the datasets**:
   Ensure you have the following CSV files in the project directory:
   - `rotten_tomatoes_movies.csv`
   - `rotten_tomatoes_movie_reviews.csv`

## Usage

You can run the movie recommendation engine by executing the script. Here’s an example of how to use the `movie_recommender_engine` function:

```python
from movie_recommender import movie_recommender_engine, movie_dataset, review_matrix, cf_knn_model

# Get recommendations for the movie "Twilight"
recommendations = movie_recommender_engine("Twilight", movie_dataset, review_matrix, cf_knn_model, 10)
print(recommendations)
```

## Data Sources

- [Rotten Tomatoes Movies Dataset](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews)
- [Rotten Tomatoes Movie Reviews Dataset](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews)

## Functionality

1. **Data Cleaning**: 
   - Removes entries with missing movie titles or scores.
   - Normalizes various scoring formats into a consistent scale (0-10).

2. **Data Merging**: 
   - Merges movie details with ratings based on the movie ID.
   - Removes duplicate entries from the merged dataset.

3. **Recommendation Engine**: 
   - Constructs a review matrix with critics as rows and movie IDs as columns.
   - Implements a KNN model to find similar movies based on critic ratings.
   - Returns the top N recommended movies based on a specified movie title.
