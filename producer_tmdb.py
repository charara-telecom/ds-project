# producer_tmdb_all.py
import requests
from kafka import KafkaProducer
import json
import time

# --- Config Kafka ---
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_movie_reviews = 'movie-reviews'
topic_movie_details = 'movie-details'

API_KEY = '8c6e215fd08d2ce11b83b7762749b724'

TMDB_ENDPOINTS = [
    f"https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}&language=en-US&page=1",
    f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1",
    f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1",
    f"https://api.themoviedb.org/3/movie/upcoming?api_key={API_KEY}&language=en-US&page=1"
]

TMDB_REVIEW_URL = "https://api.themoviedb.org/3/movie/{}/reviews?api_key={}&language=en-US"
TMDB_MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US"

# --- Eviter les doublons de films ---
sent_movie_ids = set()

# --- Boucle pour récupérer les films et leurs reviews ---
while True:
    for endpoint in TMDB_ENDPOINTS:
        response = requests.get(endpoint)
        if response.status_code == 200:
            movies = response.json().get("results", [])
            # Limite le nombre de films pour ne pas surcharger le flux
            for movie in movies[:50]:
                print("movie: ", movie)

                movie_id = movie['id']
                movie_title = movie['title']

                # --- Movie details ---
                if movie_id not in sent_movie_ids:
                    details_url = TMDB_MOVIE_DETAILS_URL.format(movie_id, API_KEY)
                    details_response = requests.get(details_url)

                    if details_response.status_code == 200:
                        movie_details = details_response.json()
                        print("###############################")
                        movie_details_payload = {
                            "movie_id": movie_id,
                            "title": movie_details.get("title"),
                            "overview": movie_details.get("overview"),
                            "release_date": movie_details.get("release_date"),
                            "runtime": movie_details.get("runtime"),
                            "genres": [g["name"] for g in movie_details.get("genres", [])],
                            "vote_average": movie_details.get("vote_average"),
                            "popularity": movie_details.get("popularity"),
                        }
                        print(movie_details_payload)

                        producer.send(topic_movie_details, movie_details_payload)
                        sent_movie_ids.add(movie_id)

                        print(f"Sent details for movie '{movie_title}'")
                    else:
                        print(f"Error TMDB details: {details_response.status_code} for {movie_title}")

                # --- Movie reviews ---
                review_url = TMDB_REVIEW_URL.format(movie_id, API_KEY)
                review_response = requests.get(review_url)

                if review_response.status_code == 200:
                    reviews = review_response.json().get("results", [])
                    print(len(reviews), " reviews found for ", movie_title)
                    print("###############################")

                    for review in reviews[:100]:
                        review["type"] = "review"
                        review["movie_id"] = movie_id
                        review["movie_title"] = movie_title
                        review["timestamp"] = time.time()

                        producer.send(topic_movie_reviews, review)
                        print(f"Sent review by {review['author']} for movie '{movie_title}'")
                else:
                    print(f"Error TMDB reviews: {review_response.status_code} for {movie_title}")
        else:
            print(f"Error TMDB endpoint: {response.status_code} for {endpoint}")

    producer.flush()
    time.sleep(60)