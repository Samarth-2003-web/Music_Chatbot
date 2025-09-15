from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import random

app = Flask(__name__)

# --- Data Loading and Preparation ---
# Load and prepare your music data once at startup
try:
    # Attempt to load the dataset
    df = pd.read_csv('music_recommendation_big_dataset.csv')
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    # If the file doesn't exist, set df to None
    print("Warning: 'music_recommendation_big_dataset.csv' not found. The app will run in a fallback mode.")
    df = None

def build_knowledge_base(data):
    """
    Processes the raw DataFrame to build a structured knowledge base.
    """
    if data is None:
        return None
        
    data = data.copy()
    # Drop rows with missing values to ensure data quality
    data = data.dropna().reset_index(drop=True)
    
    # Clean object columns by converting to lowercase
    object_cols = data.select_dtypes(include='object').columns
    for col in object_cols:
        data[col] = data[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        
    # Convert numerical columns to string for consistent keying
    data['Year'] = data['Year'].astype(str)
    data['Rating'] = data['Rating'].astype(str)

    # --- Categorization Functions ---
    def categorize_danceability(danceability):
        if danceability >= 0.7: return 'high danceability'
        elif danceability > 0.4: return 'medium danceability'
        else: return 'low danceability'

    def categorize_energy(energy):
        if energy >= 0.7: return 'high energy'
        elif energy > 0.4: return 'medium energy'
        else: return 'low energy'

    def categorize_valence(valence):
        if valence >= 0.6: return 'happy'
        elif valence > 0.4: return 'neutral'
        else: return 'sad'

    def categorize_tempo(tempo):
        if tempo < 90: return 'slow tempo'
        elif tempo < 120: return 'medium tempo'
        else: return 'fast tempo'

    # Apply categorization to create new features
    data['danceability_category'] = data['Danceability'].apply(categorize_danceability)
    data['energy_category'] = data['Energy'].apply(categorize_energy)
    data['valence_category'] = data['Valence'].apply(categorize_valence)
    data['tempo_category'] = data['Tempo'].apply(categorize_tempo)
    
    def aggregate_songs(df, column_name):
        """Helper function to group songs by a specific category."""
        # Groups by the column and creates a set of (Song Name, Artist) tuples
        return df.groupby(column_name)[['Song Name', 'Artist']].apply(lambda x: set(map(tuple, x.values))).to_dict()

    # Build the final knowledge base dictionary
    return {
        "year": aggregate_songs(data, 'Year'),
        "artist": aggregate_songs(data, 'Artist'),
        "genre": aggregate_songs(data, 'Genre'),
        "rating": aggregate_songs(data, 'Rating'),
        "mood": aggregate_songs(data, 'Mood'),
        "language": aggregate_songs(data, 'Language'),
        "tempo": aggregate_songs(data, 'tempo_category'),
        "danceability": aggregate_songs(data, 'danceability_category'),
        "energy": aggregate_songs(data, 'energy_category'),
        "valence": aggregate_songs(data, 'valence_category'),
    }

def recommend_song(knowledge_base, preferences):
    """
    Recommends songs based on user preferences by finding the intersection
    of songs matching all specified criteria.
    """
    if not preferences:
        return []

    sets_to_intersect = []
    for category, value in preferences.items():
        if category in knowledge_base and value in knowledge_base[category]:
            sets_to_intersect.append(knowledge_base[category][value])
        else:
            # If any preference doesn't match, no songs can satisfy all criteria
            return []

    if not sets_to_intersect:
        return []

    # Find the common songs across all the selected preference sets
    final_matches = set.intersection(*sets_to_intersect)
    return list(final_matches)

# --- Initialize Knowledge Base ---
knowledge_base = build_knowledge_base(df)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat logic for song recommendations."""
    if knowledge_base is None:
        return jsonify({'reply': {'text': "I'm sorry, the music dataset could not be loaded on the server.", 'songs': []}})

    user_message = request.json.get('message', '').lower()
    preferences = {}
    
    # Parse user message to extract preferences
    for category_name, category_dict in knowledge_base.items():
        for attribute_value in category_dict.keys():
            # Check if an attribute (like 'happy', 'sad', '2001') is in the message
            if str(attribute_value) in user_message:
                preferences[category_name] = attribute_value
    
    if not preferences:
        return jsonify({'reply': {'text': "I couldn't quite understand your preferences. Could you be more specific? For example, try 'show me some happy songs from 2001'.", 'songs': []}})

    recommendations = recommend_song(knowledge_base, preferences)
    
    response_data = {}
    if recommendations:
        random.shuffle(recommendations)
        
        # Prepare data for a structured JSON response
        song_list = []
        for song, artist in recommendations[:10]:
            song_list.append({'song': song.title(), 'artist': artist.title()})
            
        response_data = {
            'text': "Here are some songs you might like:",
            'songs': song_list
        }
    else:
        response_data = {
            'text': "I couldn't find any songs matching all your criteria. Please try a different combination.",
            'songs': []
        }
        
    return jsonify({'reply': response_data})

if __name__ == '__main__':
    # Make sure to place your CSV file in the same directory as this script
    app.run(debug=True)

