### INPUT: json file with ct classification results
# input file: "2_data_ct_filter_yes_{date}_v{number}.json"

### Key Vars in the input data:
# 1. `post_id` - the unique identifier for each post
# 2. `clean_post_text` - the cleaned text of the post. 

### OUTPUT: json file with claim-clusters information. 
## NOTE: content_id represents semantic clusters of similar posts
# 1. use annoy package and sentence-transformers embedding model to cluster posts 
# 2. KEY VAR: similarity threshold  X = 0.80 (by default) - meaning pair-wise similarity score between two directly linked posts should be greater than or equal to X to be considered as a part of the same cluster
# 3. expected output data with new information:
#  - `content_id` - the unique identifier for each cluster

### OUTPUT NAMING RULES:
# There are several output files: 
# 1.  "3_sem_match_{date}_v{number}.json", with all info from the input file and new `content_id` column (content_id represents cluster assignments)
# 2.  "3_sem_match_claim_{date}_v{number}.json", with only `post_id`, `content_id`, and `clean_post_text` columns (sent for the next step in the pipeline)
# 3.  "3_sem_match_edge_list_{date}_v{number}.pkl", with the edge list dictionary saved as a pickle file

# Load JSONL file (each line is a separate JSON object)
# INPUT FILE and output configuration will be defined below after imports



# # generate and save edge_list
import os
import re
import datetime
import gc
import pickle
from collections import defaultdict
from bisect import bisect_left

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
from annoy import AnnoyIndex

# Set GPU configuration before any CUDA calls
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    current_gpu = torch.cuda.current_device()
    print("Using GPU:", torch.cuda.get_device_name(current_gpu))
    print("Current GPU number:", current_gpu)
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Configuration
OUTPUT_PATH = "ct_platform/ct_nar_platform_git/pipeline_data"
INPUT_FILE = "ct_platform/ct_nar_platform_git/pipeline_data/2_data_ct_filter_yes_20250702_v1.json"
EMBEDDINGS_OUTPUT_PATH = "ct_platform/ct_nar_platform_git/pipeline_data/post_embeddings"
VERSION = 1
DEFAULT_DIMENSION = 1024  # Qwen3-Embedding-0.6B dimension
DEFAULT_TREES = 50
DEFAULT_NEIGHBORS = 50
DEFAULT_THRESHOLD = np.sqrt(0.8)  # Cosine similarity threshold of 0.8
DEFAULT_INCREASE = 2
CHUNK_SIZE = 10000

######## Load input data ######
print("Loading input data...")
df = pd.read_json(INPUT_FILE, lines=True)

######### Create content_id for unique "clean_post_text" ########
print("Creating content_id for unique clean_post_text...")
# Assign content_id to each unique text content
df['content_id'] = df.groupby('clean_post_text').ngroup() + 1
print(f"Found {df['content_id'].nunique()} unique text contents from {len(df)} total posts")
print(f"Loaded {len(df)} posts for processing")

# Create mapping dictionaries
print("Creating mapping dictionaries...")
content_id_to_text = df.drop_duplicates(subset=['content_id']).set_index('content_id')['clean_post_text'].to_dict()
content_id_to_posts = df.groupby('content_id')['post_id'].apply(list).to_dict()

# Prepare data for embedding (using content_id as identifier)
unique_df = df.drop_duplicates(subset=['content_id'], keep='first')
print(f"Reduced from {len(df)} posts to {len(unique_df)} unique contents for embedding")

# Prepare data for embedding (using unique texts only)
docs = unique_df['clean_post_text'].tolist()  # Unique texts only
ids = unique_df['content_id'].astype(str).tolist()  # Representative post_ids for unique texts

# ===== EMBEDDING GENERATION FUNCTIONS =====
# These functions handle embedding generation, saving, and loading

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

# ===== SIMILARITY SEARCH AND CLUSTERING FUNCTIONS =====
# These functions handle Annoy index creation and edge list generation, 

def create_annoy_index(embeddings_dict, dimension=DEFAULT_DIMENSION, trees=DEFAULT_TREES):
    """Create an Annoy index from embeddings dictionary."""
    index = AnnoyIndex(dimension, 'angular')
    
    # Get ordered list of IDs for consistent indexing
    ids_list = list(embeddings_dict.keys())
    
    print(f"Building Annoy index with {len(embeddings_dict)} embeddings...")
    for i, embedding in tqdm(enumerate(embeddings_dict.values()), total=len(embeddings_dict)):
        index.add_item(i, embedding)
    
    index.build(trees)
    print(f"Annoy index built with {trees} trees")
    return index, ids_list

def get_edge_list(embeddings_dict, index, ids_list, start_neighbors=DEFAULT_NEIGHBORS, 
                  increase_rate=DEFAULT_INCREASE, threshold=DEFAULT_THRESHOLD):
    """Generate edge list based on similarity threshold."""
    print(f"Generating edge list with threshold {threshold}")
    edge_list = {}
    
    for i, post_id in tqdm(enumerate(ids_list), total=len(ids_list)):
        neighbors = start_neighbors
        nn, distances = index.get_nns_by_item(i, neighbors, include_distances=True)
        
        # Increase neighbor search if needed
        while distances and distances[-1] < threshold:
            neighbors = int(neighbors * increase_rate)
            nn, distances = index.get_nns_by_item(i, neighbors, include_distances=True)
        
        # Find insertion point and create edges
        insertion_point = bisect_left(distances, threshold)
        edge_list[post_id] = {
            ids_list[nn[x]]: (2 - (distances[x] ** 2)) / 2 
            for x in range(insertion_point)
        }
    
    print(f"Edge list generated with {len(edge_list)} nodes")
    return edge_list

def logging(string, colour="green"):
    """Print timestamped log message."""
    now = datetime.datetime.now().strftime('%H:%M:%S')
    if colour == "red":
        print(f"\033[91m{string} Starting at {now}\033[0m")
    elif colour == "green":
        print(f"\033[92m{string} Starting at {now}\033[0m")
    else:
        print(f"{string} Starting at {now}")

# ===== CLUSTERING FUNCTIONS =====
# Union-Find algorithm for clustering

class Node:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.size = 1

class UnionFind(dict):
    def find(self, key):
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node: 
                # Walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a, key_b):
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # Disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size

def find_components(edge_iterator):
    """Find connected components using Union-Find algorithm."""
    forest = UnionFind()
    
    for edge in edge_iterator:
        forest.union(edge[0], edge[1])
    
    result = defaultdict(list)
    for key in forest.keys():
        root = forest.find(key)
        result[root.key].append(key)
    
    return list(result.values())

def create_edgelist(edge_dict, threshold):
    """Create edge list from dictionary based on threshold."""
    edge_list = []
    for key in edge_dict.keys():
        for neighbor, similarity in edge_dict[key].items():
            if similarity >= threshold:
                edge_list.append((key, neighbor))
    return edge_list

def create_clusters(edge_dict, threshold=0.80):
    """Create clusters from edge dictionary."""
    edge_list = create_edgelist(edge_dict, threshold)
    clusters = find_components(edge_list)
    
    # Create ID to cluster mapping
    id_2_cluster = {}
    for i, cluster in enumerate(clusters):
        for post_id in cluster:
            id_2_cluster[post_id] = i + 1  # Start cluster IDs from 1
    
    return id_2_cluster, clusters

# ===== MAIN EXECUTION PIPELINE =====

def generate_output_filename(prefix, suffix="json"):
    """Generate output filename with current date and version."""
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    return f"{prefix}_{current_date}_v{VERSION}.{suffix}"

def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("SEMANTIC MATCHING PIPELINE FOR CONSPIRACY THEORY POSTS")
    print("=" * 60)
    
    # Step 1: Generate or load embeddings
    logging("Step 1: Processing embeddings")
    
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
    
    # Step 2: Create Annoy index
    logging("Step 2: Creating Annoy index")
    annoy_index, ids_list = create_annoy_index(embeddings, DEFAULT_DIMENSION, DEFAULT_TREES)
    
    # Step 3: Generate edge list
    logging("Step 3: Generating edge list")
    edge_list = get_edge_list(embeddings, annoy_index, ids_list, 
                              DEFAULT_NEIGHBORS, DEFAULT_INCREASE, DEFAULT_THRESHOLD)
    
    # Save edge_list as pickle file
    output_dir = os.path.dirname(OUTPUT_PATH)
    edge_list_filename = generate_output_filename("3_sem_match_edge_list", "pkl")
    edge_list_path = os.path.join(output_dir, edge_list_filename)
    
    with open(edge_list_path, 'wb') as f:
        pickle.dump(edge_list, f)
    print(f"Edge list saved to: {edge_list_path}")
    
    # Step 4: Create clusters
    logging("Step 4: Creating clusters")
    threshold = 0.80  # Cosine similarity threshold
    id_2_cluster, clusters = create_clusters(edge_list, threshold)
    
    # Step 5: Add cluster information to dataframe
    logging("Step 5: Adding cluster information to dataframe")
    
    # Create expanded cluster mapping for all post_ids (including duplicates)
    expanded_id_2_cluster = {}
    for content_id_str, cluster_id in id_2_cluster.items():
        # Convert content_id back to int to use as dictionary key
        content_id = int(content_id_str)
        # Map all posts with this content_id to the same cluster
        for post_id in content_id_to_posts[content_id]:
            expanded_id_2_cluster[str(post_id)] = cluster_id
    
    # Also assign cluster IDs to unclustered unique texts (singletons)
    # These are texts that didn't form clusters but still need cluster IDs
    next_cluster_id = max(id_2_cluster.values()) + 1 if id_2_cluster else 1
    for content_id_str in ids:
        if content_id_str not in id_2_cluster:
            # This content wasn't clustered, give it its own cluster
            content_id = int(content_id_str)
            for post_id in content_id_to_posts[content_id]:
                expanded_id_2_cluster[str(post_id)] = next_cluster_id
            next_cluster_id += 1
    
    # Add content_id (cluster ID) to dataframe
    df['content_id'] = df['post_id'].astype(str).map(expanded_id_2_cluster)
    # All posts should have a cluster ID now, no need for fillna(0)
    df['content_id'] = df['content_id'].astype(int)
    
    # Statistics
    num_multi_clusters = len(clusters)  # Clusters with multiple content_ids
    num_singleton_clusters = next_cluster_id - 1 - num_multi_clusters  # Single content_id clusters
    total_clusters = num_multi_clusters + num_singleton_clusters
    
    # Count posts in multi-member clusters (content_ids that clustered together)
    multi_clustered_content_ids = set()
    for cluster in clusters:
        for content_id_str in cluster:
            multi_clustered_content_ids.add(int(content_id_str))
    
    multi_clustered_posts = sum(len(content_id_to_posts[cid]) for cid in multi_clustered_content_ids)
    total_posts = len(df)
    
    print(f"\nClustering Results:")
    print(f"- Total posts: {total_posts}")
    print(f"- Posts in multi-member clusters: {multi_clustered_posts} ({multi_clustered_posts/total_posts*100:.1f}%)")
    print(f"- Posts in singleton clusters: {total_posts - multi_clustered_posts}")
    print(f"- Multi-member clusters: {num_multi_clusters}")
    print(f"- Singleton clusters: {num_singleton_clusters}")
    print(f"- Total clusters: {total_clusters}")
    print(f"- Average multi-cluster size: {multi_clustered_posts/num_multi_clusters:.1f}" if num_multi_clusters > 0 else "- No multi-member clusters")
    
    # Step 6: Save output files
    logging("Step 6: Saving output files")
    
    # Output directory
    output_dir = os.path.dirname(INPUT_FILE)
    
    # Full output file (all columns + content_id)
    full_output_file = os.path.join(output_dir, generate_output_filename("3_sem_match"))
    df.to_json(full_output_file, orient='records', lines=True)
    print(f"Full output saved to: {full_output_file}")
    
    # Minimal output file (only post_id, content_id, clean_post_text)
    minimal_df = df[['post_id', 'content_id', 'clean_post_text']].copy()
    minimal_output_file = os.path.join(output_dir, generate_output_filename("3_sem_match_claim"))
    minimal_df.to_json(minimal_output_file, orient='records', lines=True)
    print(f"Minimal output saved to: {minimal_output_file}")
    
    print("\n" + "=" * 60)
    print("SEMANTIC MATCHING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return df

# Execute main pipeline
if __name__ == "__main__":
    result_df = main()


