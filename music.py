import pandas as pd
import numpy as np
import random

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
    """
    Recommends songs by finding the intersection of all user preferences.
    """
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



print(" Loading and analyzing your music library, please wait...")
try:
    df = pd.read_csv('music_recommendation_big_dataset.csv')

    df.columns = df.columns.str.strip()

    knowledge_base = Music(df)
    print("Done! Your chatbot is ready.")

except FileNotFoundError:
    print(" Error: 'music_recommendation_big_dataset.csv' not found. Make sure it's in the same folder!")
    exit()

# 2. Start the interactive chat loop
print("\n--- 🎵 Music Recommender Chatbot 🎵 ---")
print("I can find songs for you based on mood, genre, artist, year, energy, etc.")
print("For example: 'Suggest a happy rock song from 2000' or 'find a slow tempo sad song'")
print(" Type 'exit' to quit.")

while True:
    user_input = input("\n> ").lower()
    if user_input == 'exit':
        print(" Goodbye!")
        break

    # 3. Parse user input to build a preferences dictionary
    preferences = {}
    # This simple parser checks every known attribute against the user's input text
    for category_name, category_dict in knowledge_base.items():
        for attribute_value in category_dict.keys():
            # Check if the attribute (e.g., 'happy', 'rock', '2005') is in the user's text
            if str(attribute_value) in user_input:
                preferences[category_name] = attribute_value

    if not preferences:
        print("I'm sorry, I couldn't understand that. Try being more specific.")
        continue

    print(f"Searching with preferences: {preferences}...")
    recommendations = recommend_song(knowledge_base, preferences)

    # 4. Display the results
    if recommendations:
        print("\nI found some songs you might like!")
        random.shuffle(recommendations)
        # Display up to 5 recommendations
        for i, (song, artist) in enumerate(recommendations[:10]):
            print(f"  - '{song.title()}' by {artist.title()}")
    else:
        print("\n Couldn't find any songs matching all your criteria. Please try a different combination.")