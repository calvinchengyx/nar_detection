# validate the variance of narrative descriptions
# the goal is to report average simiarity of three different generated CT narratives, my hypothesis is that 
# the similarity should be high, changing the number of candidates should not change overal interpretation of the topic summary 
# INOUT: 6_topic2nar_{DATE}_v{VERSION}.json
# OUTPUT: 6_1_topic2nar_vali_{DATE}_v{VERSION}.json

import pandas as pd
import json
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


### load the data 
SERVER = "ct_platform/ct_nar_platform_git"
INPUT = f"{SERVER}/pipeline_data/6_topic2nar_20250703_v1.json"
DATE = time.strftime("%Y%m%d")
VERSION = 1
OUTPUT_FILE = f"{SERVER}/pipeline_data/6_1_topic2nar_vali_{DATE}_v{VERSION}.json"

with open(INPUT, 'r') as file:
    data_topic = json.load(file)
df = pd.DataFrame(data_topic)
df['topic'] = df['topic'].astype(str)

# create a dictionary with topic_id as keys, mmr_10_doc, mmr_20_doc, mmr_30_doc as values
topic_dict = {}
for index, row in df.iterrows():
    topic_id = row['topic']
    mmr_10_doc = row['mmr_10_nar']
    mmr_20_doc = row['mmr_20_nar']
    mmr_30_doc = row['mmr_30_nar']
    
    if topic_id not in topic_dict:
        topic_dict[topic_id] = {
            'mmr_10_nar': mmr_10_doc.replace('topic: ', '').strip(),
            'mmr_20_nar': mmr_20_doc.replace('topic: ', '').strip(),
            'mmr_30_nar': mmr_30_doc.replace('topic: ', '').strip()
        }

def calculate_word_overlap_similarity(doc1, doc2, doc3):
    """Calculate simple word overlap similarity using one-hot encoding"""
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, binary=True)
    onehot_matrix = vectorizer.fit_transform([doc1, doc2, doc3])
    similarities = cosine_similarity(onehot_matrix)
    # Average of the three pairwise similarities
    avg_similarity = (similarities[0,1] + similarities[0,2] + similarities[1,2]) / 3
    return avg_similarity

def calculate_tfidf_similarity(doc1, doc2, doc3):
    """Calculate TF-IDF weighted similarity"""
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2, doc3])
    similarities = cosine_similarity(tfidf_matrix)
    # Average of the three pairwise similarities
    avg_similarity = (similarities[0,1] + similarities[0,2] + similarities[1,2]) / 3
    return avg_similarity

def calculate_semantic_similarity(doc1, doc2, doc3):
    """Calculate semantic similarity using sentence embeddings"""
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode([doc1, doc2, doc3])
    
    similarities = cosine_similarity(embeddings)
    # Average of the three pairwise similarities
    avg_similarity = (similarities[0,1] + similarities[0,2] + similarities[1,2]) / 3
    return avg_similarity

# Apply all three similarity measures to your topic_dict
word_similarities = []
tfidf_similarities = []
semantic_similarities = []

for topic_id, narratives in topic_dict.items():
    # Calculate word overlap similarity (one-hot)
    word_sim = calculate_word_overlap_similarity(
        narratives['mmr_10_nar'],
        narratives['mmr_20_nar'], 
        narratives['mmr_30_nar']
    )
    word_similarities.append(word_sim)
    
    # Calculate TF-IDF similarity
    tfidf_sim = calculate_tfidf_similarity(
        narratives['mmr_10_nar'],
        narratives['mmr_20_nar'], 
        narratives['mmr_30_nar']
    )
    tfidf_similarities.append(tfidf_sim)
    
    # Calculate semantic similarity
    semantic_sim = calculate_semantic_similarity(
        narratives['mmr_10_nar'],
        narratives['mmr_20_nar'], 
        narratives['mmr_30_nar']
    )
    semantic_similarities.append(semantic_sim)
    
    print(f"Topic {topic_id}:")
    print(f"  Word Overlap Similarity (One-Hot): {word_sim:.3f}")
    print(f"  TF-IDF Similarity: {tfidf_sim:.3f}")
    print(f"  Semantic Similarity: {semantic_sim:.3f}")
    print()

print("=== OVERALL RESULTS ===")
print(f"Average Word Overlap Similarity: {np.mean(word_similarities):.3f}")
print(f"Average TF-IDF Similarity: {np.mean(tfidf_similarities):.3f}")
print(f"Average Semantic Similarity: {np.mean(semantic_similarities):.3f}")
print(f"Difference (Semantic - Word): {np.mean(semantic_similarities) - np.mean(word_similarities):.3f}")
print(f"Difference (TF-IDF - Word): {np.mean(tfidf_similarities) - np.mean(word_similarities):.3f}")

# Create results DataFrame for analysis
results_df = pd.DataFrame({
    'topic': list(topic_dict.keys()),
    'word_overlap_similarity': word_similarities,
    'tfidf_similarity': tfidf_similarities,
    'semantic_similarity': semantic_similarities
})
results_df['semantic_vs_word'] = results_df['semantic_similarity'] - results_df['word_overlap_similarity']
results_df['tfidf_vs_word'] = results_df['tfidf_similarity'] - results_df['word_overlap_similarity']

print("\n=== DETAILED ANALYSIS ===")
print(results_df.describe())

print("\n=== SAVE RESULTs ===")
# save to json file
results_df.to_json(OUTPUT_FILE, orient='records', lines=True)
print(f"Results saved to {OUTPUT_FILE}")
