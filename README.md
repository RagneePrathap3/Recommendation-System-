# Music Recommendation-System-

This is a Music Recommendation System project that recommends a song based on user input. If a user inputs a favorite song, the system identifies similar songs based on the artist and genre. It applies Machine Learning (ML) to study the song patterns and makes informed recommendations. It aims to provide a basic yet user-friendly music discovery tool.

The backend is developed with Flask (Python) and takes care of the recommendation logic. It operates on a dataset of songs, cleanses the data, and identifies key features such as artists and genres. The system employs TF-IDF (Text-Based Similarity) and K-Nearest Neighbors (KNN) to identify songs most similar to the one input. The backend takes the user-supplied song name, looks for similar songs in the data set, and returns a list of suggested tracks in JSON format.

The frontend is a basic HTML, CSS, and JavaScript interface that enables users to input a song title and receive recommendations in real-time. Once a user inputs a song and clicks the "Get Recommendations" button, the Flask backend receives a request. The results are dynamically rendered on the page, listing the recommended songs and their artists. The system provides an interactive and seamless experience for users seeking to discover new music.  

