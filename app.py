from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import random

app = Flask(__name__)

# Load and prepare your music data once at startup
try:
    df = pd.read_csv('music_recommendation_big_dataset.csv')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    df = None

def Music(data):
    data = data.copy()
    data = data.dropna().reset_index(drop=True)
    object_cols = data.select_dtypes(include='object').columns
    for col in object_cols:
        data[col] = data[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    data['Year'] = data['Year'].astype(str)
    data['Rating'] = data['Rating'].astype(str)
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
    data['danceability_category'] = data['Danceability'].apply(categorize_danceability)
    data['energy_category'] = data['Energy'].apply(categorize_energy)
    data['valence_category'] = data['Valence'].apply(categorize_valence)
    data['tempo_category'] = data['Tempo'].apply(categorize_tempo)
    def aggregate_songs(df, column_name):
        return df.groupby(column_name)[['Song Name', 'Artist']].apply(lambda x: set(map(tuple, x.values))).to_dict()
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
    if not preferences:
        return []
    sets_to_intersect = []
    for category, value in preferences.items():
        if category in knowledge_base and value in knowledge_base[category]:
            sets_to_intersect.append(knowledge_base[category][value])
        else:
            return []
    if not sets_to_intersect:
        return []
    final_matches = set.intersection(*sets_to_intersect)
    return list(final_matches)

if df is not None:
    knowledge_base = Music(df)
else:
    knowledge_base = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if knowledge_base is None:
        return jsonify({'reply': "Music dataset not found on server."})
    user_message = request.json.get('message', '').lower()
    preferences = {}
    for category_name, category_dict in knowledge_base.items():
        for attribute_value in category_dict.keys():
            if str(attribute_value) in user_message:
                preferences[category_name] = attribute_value
    if not preferences:
        return jsonify({'reply': "I'm sorry, I couldn't understand that. Try being more specific."})
    recommendations = recommend_song(knowledge_base, preferences)
    if recommendations:
        random.shuffle(recommendations)
        reply = "Here are some songs you might like:\n"
        for song, artist in recommendations[:10]:
            reply += f"- '{song.title()}' by {artist.title()}\n"
    else:
        reply = "Couldn't find any songs matching all your criteria. Please try a different combination."
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
