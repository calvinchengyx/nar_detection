### INPUT: json file with ct classification results
# input file: "4_post2claim_{date}_v{number}.json"
# key vars: "claim_id", "claim_text"

#### processing steps:
# 1. embed claim_text using sentence-transformers embedding model
# 2. use bertopic to cluster claims
# 3. create a new df, with `claim_id`, `claim_text`, and `topic_id` columns

### OUTPUT:
# there are sevearal output files:
# 1. "5_claim2topic_topic_{date}_v{number}.json" with topic_id, rep words, rep docs and rep_doc_ctfidf for later MMR
# 2. "5_claim2topic_doctopic_{date}_v{number}.json" with claim_id, claim_text, topic_id columns, so each claim_id is associated with each topic_id

import pandas as pd
import json
import os 
import time
import datetime
import gc
import re
import random
import logging
import torch
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP # dimension reductino
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures


# Try to import RAPIDS cuML UMAP, fall back to regular UMAP if not available
# try:
#     from cuml import UMAP
#     print("Using RAPIDS cuML UMAP for GPU acceleration")
#     RAPIDS_AVAILABLE = True
# except ImportError:
#     from umap import UMAP
#     print("RAPIDS cuML not available, using standard UMAP")
#     RAPIDS_AVAILABLE = False

RAPIDS_AVAILABLE = False

### Configuration
SERVER = "ct_platform_nar" #"ct_platform/ct_nar_platform_git"
INPUT = f"{SERVER}/pipeline_data/4_post2claim_20250702_v1.json"
OUTPUT_PATH = f"{SERVER}/pipeline_data"
EMBEDDINGS_OUTPUT_PATH = f"{SERVER}/pipeline_data/claim_embeddings"
CHUNK_SIZE = 10000
VERSION = 1
DATE = time.strftime("%Y%m%d")
GPU_DEVICE = 1  # Specify which GPU to use (0 for first GPU, 1 for second, etc.)

### 1. Load the INPUT data
print("Loading input data...")

with open(INPUT, 'r') as file:
    data = json.load(file)

claim_list = []
for item in data:
    for content_id, claim_dict in item.items():
        # skip empty claims (no ct claim extracted from the post)
        if not claim_dict: 
            continue

        # extract each claim
        for claim_id, claim_text in claim_dict.items():
            claim_list.append({
                'content_id': content_id,
                'claim_id': claim_id,
                'claim_text': claim_text
            })
df = pd.DataFrame(claim_list)
# convert IDs to strings for better matching later
df['content_id'] = df['content_id'].astype(str)
df['claim_id'] = df['claim_id'].astype(str)

#### 2. Embed claim text and save it to the embedding output ######
# Prepare data for embedding (using unique texts only)
ids = df['claim_id'].astype(str).tolist()  # Representative post_ids for unique texts
docs = df['claim_text'].tolist()  # Unique texts only

# Additional validation - ensure all docs are strings
clean_docs = []
clean_ids = []
for i, doc in enumerate(docs):
    if isinstance(doc, str) and len(doc.strip()) > 0:
        clean_docs.append(doc.strip())
        clean_ids.append(ids[i])

print(f"Final validation: {len(clean_docs)} valid documents out of {len(docs)}")
ids = clean_ids
docs = clean_docs

#### 3. Generate embeddings and save them in chunks ######
def generate_and_save_embeddings(docs, ids, output_path, chunk_size=CHUNK_SIZE):
    """
    Generate embeddings for documents and save them in chunks.
    
    Args:
        docs: List of document texts
        ids: List of document IDs
        output_path: Directory to save embedding files
        chunk_size: Number of documents per chunk
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load embedding model
    print("Loading Qwen3-Embedding-0.6B model...")
    print("Before model load:", torch.cuda.memory_allocated() / 1e9, "GB")
    model = sentence_transformers.SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    print("After model load:", torch.cuda.memory_allocated() / 1e9, "GB")
    
    # Split into chunks
    docs_chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    ids_chunks = [ids[x:x+chunk_size] for x in range(0, len(ids), chunk_size)]
    
    for i in range(len(docs_chunks)):
        out_file = f"{output_path}/embeddings_{i+1}.npy"
        if os.path.isfile(out_file):
            print(f"Chunk {i+1} already exists, skipping...")
            continue
            
        print(f"Processing chunk {i+1} of {len(docs_chunks)}")
        print(f"Starting at {datetime.datetime.now()}")
        
        # Generate embeddings for this chunk
        embeddings = model.encode(docs_chunks[i], show_progress_bar=True, batch_size=128)
        embeddings_dict = dict(zip(ids_chunks[i], embeddings))
        
        # Save chunk to file
        np.save(out_file, embeddings_dict)
        print(f"Saved {len(embeddings_dict)} embeddings to {out_file}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Embedding generation completed!")

def load_embeddings(file_path, ids_order=None):
    """Load embeddings from a numpy file."""
    try:
        embeddings_dict = np.load(file_path, allow_pickle=True).item()
        if ids_order:
            embeddings_ordered = {id: embeddings_dict[id] for id in ids_order if id in embeddings_dict}
            return embeddings_ordered
        return embeddings_dict
    except Exception as e:
        print(f"Failed to load embeddings from {file_path} with error {e}")
        return None

def load_all_embeddings(folder_path, ids_order=None):
    """Load all embeddings from a directory."""
    embeddings_dict = {}
    
    if not os.path.exists(folder_path):
        print(f"Embeddings folder {folder_path} does not exist!")
        return embeddings_dict
    
    # Get filenames and sort by number
    filenames = sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group()))
    
    for file_name in filenames:
        if file_name.endswith('.npy'):
            chunk_embeddings = load_embeddings(os.path.join(folder_path, file_name), ids_order)
            if chunk_embeddings:
                embeddings_dict.update(chunk_embeddings)
                print(f"Loaded {len(chunk_embeddings)} embeddings from {file_name}")
    
    print(f"Total embeddings loaded: {len(embeddings_dict)}")
    return embeddings_dict

def get_umap(n_neighbors, n_components): 
    if RAPIDS_AVAILABLE:
        # Use RAPIDS cuML UMAP for GPU acceleration
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0, 
            metric='cosine', 
            random_state=42,
            verbose=True
        )
    else:
        # Fallback to standard UMAP
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0, 
            metric='cosine', 
            low_memory=True,
            random_state=42
        )
    return umap_model

def get_hdbscan(min_cluster_size, min_samples):
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        metric='euclidean', 
        min_samples = min_samples,
        prediction_data=True)
    return hdbscan_model


def get_topic_model(min_topic_size, umap_model, hdbscan_model):
    # Create n-gram vectorizer for better representative word selection
    vectorizer_model = CountVectorizer(
        stop_words="english", 
        # min_df=2,  # Minimum document frequency - words must appear in at least 2 documents
        ngram_range=(1, 3),  # Include both unigrams and bigrams
        # max_features=10000,  # Limit vocabulary size for efficiency
        # lowercase=True,
        # strip_accents='unicode'
    )
    
    topic_model = BERTopic(
    # Pipeline models
    # embedding_model= SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
    umap_model = umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,  # Add the n-gram vectorizer
    calculate_probabilities=True,
    # representation_model=representation_model,
    min_topic_size = min_topic_size,
    # reduece the number of topics 
    nr_topics = "auto",
    # Hyperparameters
    verbose=True
    # top_n_words=10,
    )
    return topic_model


def evaluate_bertopic_clusters(topic_model, embeddings, topics, docs, min_sample_size=100, random_state=42):
    """
    Comprehensive evaluation of BERTopic clusters with sampling-based approach.
    
    Parameters:
    - topic_model: fitted BERTopic model
    - embeddings: document embeddings (numpy array)
    - topics: topic assignments (list)
    - docs: original documents (list)
    - random_state: random seed for reproducible sampling
    
    Returns:
    - Dictionary with all evaluation metrics based on sampled documents
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Get unique topics (excluding outliers for some metrics)
    unique_topics = sorted(list(set(topics)))
    clustered_topics = [t for t in unique_topics if t != -1]

    results = {
        'n_topics': len(clustered_topics),
        'n_outliers': sum(1 for t in topics if t == -1),
        'outlier_ratio': sum(1 for t in topics if t == -1) / len(topics),
        'topic_sizes': {},
        'sampled_sizes': {}
    }

    # Topic sizes
    for topic in unique_topics:
        results['topic_sizes'][topic] = sum(1 for t in topics if t == topic)

    print(f"=== CLUSTER EVALUATION SUMMARY ===")
    print(f"Number of topics: {results['n_topics']}")
    print(f"Number of outliers: {results['n_outliers']} ({results['outlier_ratio']:.2%})")
    print(f"Topic sizes: {dict(sorted(results['topic_sizes'].items()))}")

    # ============= 1. COHERENCE ANALYSIS (Sampling-based) =============
    print(f"\n=== 1. COHERENCE ANALYSIS (Sampled) ===")
    coherence_scores = {}
    sampled_topic_indices = {}  # Store sampled indices for silhouette calculation

    for topic in clustered_topics:
        # Get all documents in this topic
        topic_indices = [i for i, t in enumerate(topics) if t == topic]
        topic_embeddings = embeddings[topic_indices]
        
        # Sample up to 100 documents from this topic
        sample_size = min(min_sample_size, len(topic_indices))
        if len(topic_indices) <= 100:
            # Use all documents if 100 or fewer
            sampled_indices = topic_indices
            sampled_embeddings = topic_embeddings
        else:
            # Randomly sample 100 documents
            sampled_positions = np.random.choice(len(topic_indices), size=sample_size, replace=False)
            sampled_indices = [topic_indices[pos] for pos in sampled_positions]
            sampled_embeddings = topic_embeddings[sampled_positions]
        
        # Store sampled indices for later use
        sampled_topic_indices[topic] = sampled_indices
        results['sampled_sizes'][topic] = len(sampled_indices)

        if len(sampled_embeddings) > 1:
            # Calculate pairwise cosine similarities (not distances)
            cosine_sim_matrix = cosine_similarity(sampled_embeddings)
            
            # Get upper triangular part (excluding diagonal) for pairwise similarities
            upper_tri_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
            pairwise_similarities = cosine_sim_matrix[upper_tri_indices]
            
            # Coherence is the average pairwise cosine similarity (higher = more coherent)
            avg_pairwise_similarity = np.mean(pairwise_similarities)
            std_pairwise_similarity = np.std(pairwise_similarities)

            coherence_scores[topic] = {
                'pairwise_similarity': avg_pairwise_similarity,
                'similarity_std': std_pairwise_similarity,
                'original_size': len(topic_indices),
                'sampled_size': len(sampled_indices)
            }

            print(f"Topic {topic} (original n={len(topic_indices)}, sampled n={len(sampled_indices)}): "
                  f"Avg pairwise similarity={avg_pairwise_similarity:.3f} (±{std_pairwise_similarity:.3f})")

    results['coherence'] = coherence_scores

    # ============= 2. SILHOUETTE SCORE (Sampling-based) =============
    print(f"\n=== 2. SILHOUETTE ANALYSIS (Sampled) ===")

    # Collect all sampled embeddings and their labels
    all_sampled_embeddings = []
    all_sampled_labels = []
    
    for topic in clustered_topics:
        if topic in sampled_topic_indices:
            topic_sampled_indices = sampled_topic_indices[topic]
            topic_sampled_embeddings = embeddings[topic_sampled_indices]
            topic_labels = [topic] * len(topic_sampled_indices)
            
            all_sampled_embeddings.extend(topic_sampled_embeddings)
            all_sampled_labels.extend(topic_labels)
    
    if len(all_sampled_embeddings) > 1 and len(set(all_sampled_labels)) > 1:
        all_sampled_embeddings = np.array(all_sampled_embeddings)
        
        # Calculate overall silhouette score on sampled data
        overall_silhouette = silhouette_score(all_sampled_embeddings, all_sampled_labels, metric='cosine')
        silhouette_samples_scores = silhouette_samples(all_sampled_embeddings, all_sampled_labels, metric='cosine')

        # Per-topic silhouette scores
        topic_silhouettes = {}
        sample_idx = 0
        
        for topic in clustered_topics:
            if topic in sampled_topic_indices:
                topic_sample_size = results['sampled_sizes'][topic]
                topic_silhouette_scores = silhouette_samples_scores[sample_idx:sample_idx + topic_sample_size]
                
                topic_silhouettes[topic] = {
                    'mean_silhouette': np.mean(topic_silhouette_scores),
                    'std_silhouette': np.std(topic_silhouette_scores),
                    'sampled_size': topic_sample_size,
                    'original_size': results['topic_sizes'][topic]
                }
                sample_idx += topic_sample_size

                print(f"Topic {topic}: Silhouette = {topic_silhouettes[topic]['mean_silhouette']:.3f} "
                      f"(±{topic_silhouettes[topic]['std_silhouette']:.3f}) "
                      f"[sampled {topic_sample_size}/{results['topic_sizes'][topic]}]")

        results['silhouette'] = {
            'overall_score': overall_silhouette,
            'topic_scores': topic_silhouettes,
            'total_sampled': len(all_sampled_embeddings)
        }

        # print(f"Overall Silhouette Score: {overall_silhouette:.3f} (based on {len(all_sampled_embeddings)} sampled documents)")

    else:
        results['silhouette'] = {'overall_score': -1, 'topic_scores': {}, 'total_sampled': 0}
        # print("Cannot calculate silhouette score: insufficient sampled points")

    return results

def get_low_quality_topics(evaluation_results, coherence_threshold=0.4, silhouette_threshold=0.0):
    """
    Get topic numbers where coherence < threshold AND silhouette < threshold.
    
    Parameters:
    - evaluation_results: Dictionary from evaluate_bertopic_clusters()
    - coherence_threshold: Minimum coherence score (default: 0.2)
    - silhouette_threshold: Minimum silhouette score (default: 0.0)
    
    Returns:
    - List of topic numbers meeting both criteria
    """
    
    coherence_data = evaluation_results.get('coherence', {})
    silhouette_data = evaluation_results.get('silhouette', {}).get('topic_scores', {})
    
    low_quality_topics = []
    
    # Check topics that have both scores
    for topic in coherence_data.keys():
        if topic in silhouette_data:
            coherence_score = coherence_data[topic]['pairwise_similarity']
            silhouette_score = silhouette_data[topic]['mean_silhouette']
            
            # Both criteria must be met
            if coherence_score < coherence_threshold and silhouette_score < silhouette_threshold:
                low_quality_topics.append(topic)
    
    return sorted(low_quality_topics)



def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("SEMANTIC MATCHING PIPELINE FOR CONSPIRACY THEORY POSTS")
    print("=" * 60)
    
    # Set GPU device for RAPIDS cuML if available
    if RAPIDS_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)
        print(f"Using GPU device: {GPU_DEVICE}")
    
    # Step 1: Generate or load embeddings
    print("Step 1: Processing embeddings")
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_OUTPUT_PATH) and os.listdir(EMBEDDINGS_OUTPUT_PATH):
        print("Found existing embeddings, loading...")
        embeddings = load_all_embeddings(EMBEDDINGS_OUTPUT_PATH, ids)
        
        # Check if we have embeddings for all posts
        missing_ids = set(ids) - set(embeddings.keys())
        if missing_ids:
            print(f"Missing embeddings for {len(missing_ids)} posts, regenerating all...")
            generate_and_save_embeddings(docs, ids, EMBEDDINGS_OUTPUT_PATH)
            embeddings = load_all_embeddings(EMBEDDINGS_OUTPUT_PATH, ids)
    else:
        print("No existing embeddings found, generating...")
        generate_and_save_embeddings(docs, ids, EMBEDDINGS_OUTPUT_PATH)
        embeddings = load_all_embeddings(EMBEDDINGS_OUTPUT_PATH, ids)
    
    if not embeddings:
        print("ERROR: Failed to generate or load embeddings!")
        return
    
    # Convert embeddings to array format for processing
    embeddings_array = np.array([embeddings[id_] for id_ in ids])
    
    # Optimized parameters for ~9K documents
    n_doc = len(docs)  
    neighbors =  int((n_doc ** 0.5) / 6)  
    components = 25     
    min_cluster = int(n_doc * 0.5/100)   
    min_sample = min_cluster    
    min_topic_size = 10 if n_doc < 1000 else 25 if n_doc < 10**4 else 50 if n_doc < 10**5 else 100 if n_doc < 10**6 else 500 if n_doc < 10**7 else 1000
    print(f"Number of documents: {n_doc}, min_cluster_size: {min_cluster}, min_topic_size: {min_topic_size}")

    umap_model = get_umap(n_neighbors=neighbors, n_components=components)
    hdbscan_model = get_hdbscan(min_cluster_size=min_cluster, min_samples=min_sample)
    topic_model = get_topic_model(min_topic_size=min_cluster, umap_model=umap_model, hdbscan_model=hdbscan_model)

    topics, probs = topic_model.fit_transform(docs, embeddings_array)

    ### step 3: outlier handling
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.5)
    new_topics = topic_model.reduce_outliers(docs, new_topics, probabilities=probs, threshold=0.5, strategy="probabilities")
    topic_model.update_topics(docs, topics=new_topics)

    ### step 3.1: optional - update vectorizer for improved topic representation after outlier reduction
    # This step refines the c-TF-IDF representation with potentially better parameters
    # now that we have cleaner topics after outlier reduction
    improved_vectorizer = CountVectorizer(
        ngram_range=(1, 3),  # Extended to trigrams for more specific phrases
        stop_words="english",
        # min_df=max(2, int(len(docs) * 0.001)),  # Dynamic min_df: at least 0.1% of docs or 2, whichever is higher
        # max_df=0.95,  # Remove words that appear in >95% of documents
        # max_features=15000,  # Slightly larger vocabulary for better representation
        # lowercase=True,
        # strip_accents='unicode'
    )
    topic_model.update_topics(docs, vectorizer_model=improved_vectorizer)

    ### step 4: evaluate the topic (just print out)
    # print the percentage of outliers
    outliers = np.sum(np.array(topics) == -1)
    total = len(topics)
    outlier_percentage = (outliers / total) * 100
    print(f"Percentage of outliers: {outlier_percentage:.2f}%")

    new_outliers = np.sum(np.array(new_topics) == -1)
    new_outlier_percentage = (new_outliers / total) * 100
    print(f"Percentage of outliers after reduction: {new_outlier_percentage:.2f}%")

    # print the dominant topic percentage
    dominant_topic = np.sum(np.array(topics) == 0)
    dominant_topic_percentage = (dominant_topic / total) * 100
    print(f"Dominant topic: {dominant_topic_percentage:.2f}%")

    ### step 5: quantitative evaluation of the topic model
    cluster_evaluation = evaluate_bertopic_clusters(topic_model, embeddings_array, new_topics, docs)
    topics_to_drop = get_low_quality_topics(cluster_evaluation)
    print(f"Low quality topics to drop: {topics_to_drop}")

    ### step 5.1: generate visualizations
    print("Generating visualizations...")
    visualize_cluster_metrics(cluster_evaluation, topic_model, OUTPUT_PATH)
    visualize_topic_distribution(topic_model, embeddings_array, OUTPUT_PATH)

    ### step 6. save the topic model results 
    # Selected the most representative document (N = 1) for each topic, based on c-TF-IDF
    rep_doc = topic_model.representative_docs_
    rep_doc = {k: v[0] for k, v in rep_doc.items() if v}

    # Get topic probabilities for each document
    topic_probs = []
    for i, topic in enumerate(topic_model.topics_):
        if topic == -1:  # Outlier documents
            topic_probs.append(0.0)
        elif topic_model.probabilities_ is not None:
            topic_probs.append(topic_model.probabilities_[i][topic])
        else:
            topic_probs.append(None)  # No probability data available

    # Map topic to representative document content
    topic_to_rep_doc = {}
    for topic, doc in rep_doc.items():
        topic_to_rep_doc[topic] = doc

    # Create the DataFrame
    bertopic_result = pd.DataFrame({
        "claim_id": ids,
        "claim_text": docs,
        "topic": topic_model.topics_,
        "topic_prob": topic_probs,
        "rep_doc_ctfidf": [topic_to_rep_doc.get(topic, None) for topic in topic_model.topics_]
    })
    
    bertopic_result_raw = topic_model.get_topic_info()
    bertopic_result = bertopic_result.merge(bertopic_result_raw[['Topic', 'Count', 'Name', 'Representation']], left_on='topic', right_on='Topic', how='left')
    bertopic_result = bertopic_result.drop(columns=['claim_text'])
    
    # save topic info back to the original documents 
    bertopic_result_final = df.merge(bertopic_result, on='claim_id', how='left')
    if topics_to_drop: 
        bertopic_result_final = bertopic_result_final[~bertopic_result_final['topic'].isin(topics_to_drop)]
    
    # Save doctopic results as JSON
    doctopic_output_file = os.path.join(OUTPUT_PATH, f"5_claim2topic_doctopic_{DATE}_v{VERSION}.json")
    bertopic_result_final.to_json(doctopic_output_file, orient='records', indent=2)
    print(f"Saved doctopic results to: {doctopic_output_file}")

    # save topic info individually for descriptions 
    bertopic_result_raw['rep_doc_ctfidf'] = bertopic_result_raw['Topic'].map(rep_doc)
    if topics_to_drop: 
        bertopic_result_raw = bertopic_result_raw[~bertopic_result_raw['Topic'].isin(topics_to_drop)]
    
    # Save topic results as JSON
    topic_output_file = os.path.join(OUTPUT_PATH, f"5_claim2topic_topic_{DATE}_v{VERSION}.json")
    bertopic_result_raw.to_json(topic_output_file, orient='records', indent=2)
    print(f"Saved topic results to: {topic_output_file}")
    
    print("BERTopic completed successfully!")
    return bertopic_result_final


def visualize_cluster_metrics(evaluation_results, topic_model, output_path):
    """
    Create comprehensive visualizations of cluster quality for BERTopic models.
    """
    # Check if evaluation_results is None
    if evaluation_results is None:
        print("Error: evaluation_results is None. Make sure to run the evaluation function first.")
        return
    
    # Check if required keys exist
    if 'coherence' not in evaluation_results:
        print("Error: 'coherence' key not found in evaluation results.")
        return
    
    if 'silhouette' not in evaluation_results:
        print("Error: 'silhouette' key not found in evaluation results.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('BERTopic Cluster Quality Assessment', fontsize=16, fontweight='bold')
    
    # 1. Cohesion vs Topic Size
    ax1 = axes[0]
    cohesion_data = evaluation_results['coherence']
    
    if cohesion_data:  # Check if cohesion_data is not empty
        topics = list(cohesion_data.keys())
        sizes = [cohesion_data[t]['original_size'] for t in topics]
        cohesions = [cohesion_data[t]['pairwise_similarity'] for t in topics]
        
        scatter = ax1.scatter(sizes, cohesions, c=topics, cmap='tab10', alpha=0.7, s=80)
        ax1.set_xlabel('Topic Size (# documents)')
        ax1.set_ylabel('Coherence (avg pair-wise similarity)')
        ax1.set_title('Topic Cohesion vs Size')
        ax1.grid(True, alpha=0.3)
        
        # Add topic labels
        for i, topic in enumerate(topics):
            ax1.annotate(f'T{topic}', (sizes[i], cohesions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No cohesion data available', transform=ax1.transAxes, 
                ha='center', va='center')
    
    # 2. Silhouette Scores by Topic
    ax2 = axes[1]
    if 'topic_scores' in evaluation_results['silhouette'] and evaluation_results['silhouette']['topic_scores']:
        sil_data = evaluation_results['silhouette']['topic_scores']
        topics = list(sil_data.keys())
        sil_scores = [sil_data[t]['mean_silhouette'] for t in topics]
        sil_stds = [sil_data[t]['std_silhouette'] for t in topics]
        
        bars = ax2.bar(range(len(topics)), sil_scores, yerr=sil_stds, 
                      capsize=3, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Topic ID')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Scores by Topic')
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels([f'T{t}' for t in topics], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No silhouette data available', transform=ax2.transAxes, 
                ha='center', va='center')

    plt.tight_layout()
    
    # Save the figure
    cluster_metrics_file = os.path.join(output_path, f"5_cluster_metrics_{DATE}_v{VERSION}.png")
    plt.savefig(cluster_metrics_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster metrics visualization to: {cluster_metrics_file}")
    plt.close()

def visualize_topic_distribution(topic_model, embeddings, output_path):
    """
    Create a 2D visualization of topic distribution using UMAP.
    """
    # Create a matplotlib visualization
    plt.figure(figsize=(18, 12))  # Larger figure to accommodate longer legend

    # Get unique topics and create a color map
    unique_topics = sorted(list(set(topic_model.topics_)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_topics)))
    topic_colors = {topic: colors[i] for i, topic in enumerate(unique_topics)}

    # Get topic info for names
    topic_info = topic_model.get_topic_info()
    topic_names = {}
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_names[topic_id] = 'Outliers'
        else:
            # Create a meaningful name from top words (first 3 words)
            top_words = row['Name'].split('_')  # Skip the topic number, take first 3 words
            # Remove duplicates while preserving order and limit to 3
            seen = set()
            unique_words = []
            for word in top_words:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            topic_names[topic_id] = f"T{topic_id}: {', '.join(unique_words[:3])}"

    # Reduce embeddings to 2D for visualization
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2,
                              min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

    # Plot each topic with different colors
    for topic in unique_topics:
        indices = [i for i, t in enumerate(topic_model.topics_) if t == topic]
        if len(indices) > 0:
            topic_embeddings = reduced_embeddings[indices]
            
            # Use different markers for outliers vs topics
            marker = 'x' if topic == -1 else 'o'
            alpha = 0.6 if topic == -1 else 0.6
            size = 50 if topic == -1 else 20
            
            # Use topic name instead of just topic number
            label = topic_names.get(topic, f'Topic {topic}')
            
            plt.scatter(topic_embeddings[:, 0], topic_embeddings[:, 1], 
                       c=[topic_colors[topic]], 
                       label=label,
                       alpha=alpha, s=size, marker=marker)
    
    # Add legend with improved formatting
    if len(unique_topics) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                  frameon=True, fancybox=True, shadow=True)
    else:
        # Show only first 10 topics + outliers in legend for readability
        handles, labels = plt.gca().get_legend_handles_labels()
        # Ensure outliers are included if they exist
        outlier_idx = None
        for i, label in enumerate(labels):
            if 'Outliers' in label:
                outlier_idx = i
                break
        
        if outlier_idx is not None and outlier_idx >= 10:
            # Include outliers in the top 10
            selected_handles = handles[:9] + [handles[outlier_idx]]
            selected_labels = labels[:9] + [labels[outlier_idx]]
        else:
            selected_handles = handles[:10]
            selected_labels = labels[:10]
            
        plt.legend(selected_handles, selected_labels, 
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                  frameon=True, fancybox=True, shadow=True)
        
        # Add note about truncated legend
        plt.text(1.05, 0.02, f"Showing 10 of {len(unique_topics)} topics", 
                transform=plt.gca().transAxes, fontsize=8, style='italic')

    plt.title('Topic Distribution Visualization (UMAP 2D)', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # Save the figure
    topic_dist_file = os.path.join(output_path, f"5_topic_distribution_{DATE}_v{VERSION}.png")
    plt.savefig(topic_dist_file, dpi=300, bbox_inches='tight')
    print(f"Saved topic distribution visualization to: {topic_dist_file}")
    plt.close()

# Execute main pipeline
if __name__ == "__main__":
    result_df = main()