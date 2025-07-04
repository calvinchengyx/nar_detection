### INPUT:
#  "5_claim2topic_topic_{date}_v{number}.json" with topic_id, rep words, rep docs and rep_doc_ctfidf for later MMR
#  "5_claim2topic_doctopic_{date}_v{number}.json" with claim_id, claim_text, topic_id columns, so each claim_id is associated with each topic_id

### processing steps:
# 1. for each topic_id, sample N claims using MMR algorithm
# 2. the first query is the top rep_doc_ctfidf OR randomly pick one from the topic cluster (TBD)
# 3. then feed N doc, and top keywords to LLMs to generate narative text
# 4. bootstrap N number to get stable narrative results 

### OUTPUT: json file with narative information

# "6_topic2nar_topic_{date}_v{number}.json", with topic_id, rep_docs and nar_text. 

import pandas as pd
import sentence_transformers
import numpy as np
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import random
from openai import OpenAI
import time
import json

### Configuration
SERVER = "ct_platform/ct_nar_platform_git"
# SERVER = "/data/scro4316/thesis/ct_platform_nar" 
INPUT_1 = f"{SERVER}/pipeline_data/5_claim2topic_topic_20250703_v1.json"
INPUT_2 = f"{SERVER}/pipeline_data/5_claim2topic_doctopic_20250703_v1.json"
INPUT_3 = f"{SERVER}/pipeline_data/4_post2claim_20250702_v1.json"
OUTPUT_PATH = f"{SERVER}/pipeline_data"
EMBEDDINGS_OUTPUT_PATH = f"{SERVER}/pipeline_data/claim_embeddings"
VERSION = 1
DATE = time.strftime("%Y%m%d")
GPU_DEVICE = 1  # Specify which GPU to use (0 for first GPU, 1 for second, etc.)
EMBEDDING_PATH = f"{SERVER}/pipeline_data/claim_embeddings"

# Load OpenAI key from environment variable
# openai_key = os.getenv("OPENROUTER_API_KEY_mohsen")
openai_key = "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"
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=openai_key)

##### load all documents and match their claim_id ######
print("Loading data...")

with open(INPUT_1, 'r') as file:
    data_topic = json.load(file)
df_topic = pd.DataFrame(data_topic)

with open(INPUT_2, 'r') as file:
    data_topic_docs = json.load(file)
df_topic_docs = pd.DataFrame(data_topic_docs)
df_topic_docs = df_topic_docs.drop(columns={"Topic"}) # drop duplicated Topic column

with open(INPUT_3, 'r') as file:
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
df_all = pd.DataFrame(claim_list)
# convert IDs to strings for better matching later
df_all['content_id'] = df_all['content_id'].astype(str)
df_all['claim_id'] = df_all['claim_id'].astype(str)

# for each topic, match its representative document 0 with the claim_id
# each topic is represented by a document, which is the top c-TF-IDF document, 
df_topic = df_topic[["Topic", "rep_doc_ctfidf"]]
df_topic = df_topic.merge(df_topic_docs[["claim_id", "claim_text"]], 
                          left_on="rep_doc_ctfidf", right_on="claim_text",
                          how="left").drop_duplicates(subset="Topic") 
df_topic = df_topic[["Topic", "claim_id","rep_doc_ctfidf"]]
df_topic = df_topic.rename(columns={"Topic": "topic"})
# Convert topic columns to strings for consistent matching
df_topic['topic'] = df_topic['topic'].astype(str)
df_topic_docs['topic'] = df_topic_docs['topic'].astype(str)

# Also convert claim_id columns to strings
df_topic['claim_id'] = df_topic['claim_id'].astype(str)
df_topic_docs['claim_id'] = df_topic_docs['claim_id'].astype(str)

# remove outlier topics
df_topic = df_topic[df_topic['topic'] != '-1'].reset_index(drop=True)
df_topic_docs = df_topic_docs[df_topic_docs['topic'] != '-1'].reset_index(drop=True)

# load embeddings all 
docs_all = df_all['claim_text'].tolist()
ids_all = df_all['claim_id'].tolist() # there is no duplicated post_id in the dataset (checked with df["post_id"].nunique())



##### Load embeddings from numpy files ######

def load_embeddings(file_path, ids_order = ids_all):
    """Load embeddings from a numpy file."""
    try:
        embeddings_dict = np.load(file_path, allow_pickle=True).item()
        embeddings_ordered = {id: embeddings_dict[id] for id in ids_order if id in embeddings_dict}
        return embeddings_ordered
    except Exception as e:
        print(f"Failed to load embeddings from {file_path} with error {e}")
        return None

def load_all_embeddings(folder_path):
    """Load all embeddings from a directory."""
    embeddings_dict = {}
    # Get the list of filenames and sort them by the number included in the filename
    filenames = sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group()))

    for file_name in filenames:
        if file_name.endswith('.npy'):
            embeddings_dict.update(load_embeddings(os.path.join(folder_path, file_name)))

    return embeddings_dict

# load all embeddings is a dictionary with media_id as key and embeddings as value
embeddings = load_all_embeddings(folder_path = EMBEDDING_PATH)
print(f"Loaded {len(embeddings)} embeddings from the directory.")


def mmr_select_representatives(embeddings_dict, df_topic, df_topic_docs, 
                            n_samples=10, n_candidates=1000, lambda_param=0.5, random_state=42):
    """
    Select representative documents for each topic using MMR method.
    
    Parameters:
    - embeddings_dict: Dictionary {claim_id: embedding_vector}
    - df_topic: DataFrame with topic, claim_id, rep_doc_ctfidf (top representative docs)
    - df_topic_docs: DataFrame with all documents, claim_id, topic, document content
    - n_samples: Number of documents to select per topic (default: 10)
    - n_candidates: Number of candidates to sample from (default: 100)
    - lambda_param: Balance between relevance and diversity (default: 0.5)
    - random_state: Random seed for reproducibility
    
    Returns:
    - Dictionary {topic_id: [list of 10 representative documents]}
    """
    
    # Set random seed
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Initialize result dictionary
    topic_representatives = {}
    
    # Process each topic
    for _, topic_row in df_topic.iterrows():
        topic_id = int(topic_row['topic'])  # Convert to int for consistency
        
        # use c-TF-IDF representative document as the first document
        rep_doc_content = topic_row['rep_doc_ctfidf']  # Top c-TF-IDF document
        rep_doc_claim_id = topic_row['claim_id']  # claim_id of representative document
        
        print(f"Processing Topic {topic_id}...")
        
        # Step 1: Get all documents in this topic
        topic_documents = df_topic_docs[df_topic_docs['topic'] == str(topic_id)].copy()
        
        if len(topic_documents) == 0:
            print(f"Warning: No documents found for topic {topic_id}")
            continue
        
        # Step 2: Randomly sample candidates (or use all if fewer than n_candidates)
        if len(topic_documents) <= n_candidates:
            candidate_docs = topic_documents.copy()
        else:
            candidate_docs = topic_documents.sample(n=n_candidates, random_state=random_state)
        
        print(f"Topic {topic_id}: {len(topic_documents)} total docs, {len(candidate_docs)} candidates")
        
        # Step 3: Randomly select the first document from the cluster
        # random_first_doc = candidate_docs.sample(n=1, random_state=random_state).iloc[0]
        # rep_doc_content = random_first_doc['claim_text']
        # rep_doc_claim_id = random_first_doc['claim_id']
    
        
        # Check if the randomly selected document has embeddings
        if rep_doc_claim_id not in embeddings_dict:
            print(f"Warning: randomly selected claim_id {rep_doc_claim_id} not found in embeddings")
            continue
            
        top_doc_embedding = embeddings_dict[rep_doc_claim_id]
        
        # Step 4: Initialize MMR selection
        selected_docs = [rep_doc_content]  # Start with randomly selected document
        selected_claim_ids = [rep_doc_claim_id]
        
        # Remove the selected document from candidates
        remaining_candidates = candidate_docs[candidate_docs['claim_id'] != rep_doc_claim_id].copy()
        doc_column = 'claim_text'  # Column with document content

        # Step 3-6: MMR selection process
        iteration = 0
        while len(selected_docs) < n_samples and len(remaining_candidates) > 0:
            iteration += 1
            best_score = -float('inf')
            best_candidate = None
            best_candidate_doc = None
            
            for _, candidate_row in remaining_candidates.iterrows():
                candidate_claim_id = candidate_row['claim_id']
                candidate_doc = candidate_row[doc_column]
                
                # Skip if embedding not available
                if candidate_claim_id not in embeddings_dict:
                    continue
                    
                candidate_embedding = embeddings_dict[candidate_claim_id]
                
                # Step 3: Calculate relevance (similarity to top document)
                relevance = cosine_similarity([candidate_embedding], [top_doc_embedding])[0][0]
                
                # Step 4-5: Calculate diversity (max similarity to selected documents)
                max_similarity = 0
                for selected_claim_id in selected_claim_ids:
                    if selected_claim_id in embeddings_dict:
                        selected_embedding = embeddings_dict[selected_claim_id]
                        similarity = cosine_similarity([candidate_embedding], [selected_embedding])[0][0]
                        max_similarity = max(max_similarity, similarity)
                
                # Step 5: Calculate MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate_claim_id
                    best_candidate_doc = candidate_doc
            
            # Add best candidate to selected documents
            if best_candidate is not None:
                selected_docs.append(best_candidate_doc)
                selected_claim_ids.append(best_candidate)
                
                # Remove selected candidate from remaining candidates
                remaining_candidates = remaining_candidates[remaining_candidates['claim_id'] != best_candidate]
                
                # Only print if MMR score is not 0
                if best_score != 0:
                    print(f"  Selected {len(selected_docs)}/{n_samples}: MMR score = {best_score:.3f}")
            else:
                print(f"  No more valid candidates found for topic {topic_id}")
                break
        
        # Store results for this topic
        topic_representatives[topic_id] = selected_docs
        print(f"  Topic {topic_id} completed: {len(selected_docs)} documents selected\n")
    
    return topic_representatives

# Define the prompt for GPT
prompt = """
Your task is to generate the conspiracy narrative based on the provided documents.

The conspiracy narrative is a short description (less than 20 words) of the conspiracy based on the provided documents.
Your description must follow the basic conspiracy narrative structure, which includes the following elements if any of them are available:
(1) actors: who is involved in the conspiracy,
(2) actions: what actions are taken by the actors,
(3) events: what events are happening as part of the conspiracy,
(4) secrecy: what is the plot of the conspiracy

You MUST NOT using generic terms or implications. 

Return in the following format:
topic: <description>

Here is a conspiracy that contains the following documents
[DOCUMENTS]
"""

# Function to generate topic labels using OpenAI
def generate_topic_label(topic_docs):
    # Replace placeholders in the prompt
    current_prompt = prompt.replace("[DOCUMENTS]", str(topic_docs))
    
    # Make the API request with the model parameter
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Your are a helpful assistant that excellent in summarizing concise conspiracy narratives based on provided documents. "},
            {"role": "user", "content": current_prompt}
        ],
        temperature=0  # Lower temperature for more consistent results
    )
    
    # Extract and return the generated label
    return response.choices[0].message.content.strip()

# Run MMR and GPT summary for different document counts
n_samples_list = [10, 20, 30]
all_results = {}

for n_samples in n_samples_list:
    print(f"\n=== PROCESSING WITH {n_samples} DOCUMENTS ===")
    
    # Run MMR selection
    topic_representatives_dict = mmr_select_representatives(
        embeddings_dict=embeddings,  # embeddings dictionary
        df_topic=df_topic,  # topic info DataFrame
        df_topic_docs=df_topic_docs,  # documents DataFrame
        n_samples=n_samples,  # number of documents to select per topic
        n_candidates=200,  # number of candidates to sample per topic
        lambda_param=0.5,  # balance between relevance and diversity
        random_state=42
    )
    
    # Print summary
    print(f"=== MMR SELECTION SUMMARY FOR {n_samples} DOCUMENTS ===")
    for topic_id, docs in topic_representatives_dict.items():
        print(f"Topic {topic_id}: {len(docs)} documents selected")
    
    # Generate narratives
    print(f"Generating narratives for {n_samples} documents...")
    topic_descriptions = {}
    
    for topic, rep_docs in list(topic_representatives_dict.items()):
        topic_docs = rep_docs
        topic_label = generate_topic_label(topic_docs)
        topic_descriptions[topic] = {
            "documents": topic_docs,
            "ct_nar": topic_label
        }
        print(f"Generated narrative for topic {topic} ({n_samples} docs): {topic_label}")
    
    # Store results
    all_results[n_samples] = {
        "documents": {k: v['documents'] for k, v in topic_descriptions.items()},
        "narratives": {k: v['ct_nar'] for k, v in topic_descriptions.items()}
    }

# Add new columns to df_topic
for n_samples in n_samples_list:
    # Add document columns
    doc_dict = all_results[n_samples]["documents"]
    df_topic[f'mmr_{n_samples}_doc'] = df_topic['topic'].astype(int).map(doc_dict)
    
    # Add narrative columns
    nar_dict = all_results[n_samples]["narratives"]
    df_topic[f'mmr_{n_samples}_nar'] = df_topic['topic'].astype(int).map(nar_dict)

print("\n=== FINAL RESULTS SUMMARY ===")
for n_samples in n_samples_list:
    print(f"Added columns: mmr_{n_samples}_doc, mmr_{n_samples}_nar")

# Save the results to JSON
output_file = f"{OUTPUT_PATH}/6_topic2nar_{DATE}_v{VERSION}.json"
df_topic_json = df_topic.to_dict(orient='records')

with open(output_file, 'w') as f:
    json.dump(df_topic_json, f, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"DataFrame shape: {df_topic.shape}")
print(f"Columns: {list(df_topic.columns)}")    