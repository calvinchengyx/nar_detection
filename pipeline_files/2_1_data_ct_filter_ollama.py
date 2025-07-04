### INPUT: json file with structured data for each post
# input file: "1_data_prep_{date}_v{number}.json"

### OUTPUT: json file with data that are classified as CT posts by LLMs
## expected output data format
# 1. with one new column `ct` - indicating whether the post is classified as CT or not "yes/no"
# 2. keep all other columns from the input file
# 3. keep only posts that are classified as CT posts (i.e. `ct` column is "yes")

### OUTPUT NAMING RULES:
# There are two OUTPUT files 
# 1. OUTPUT file A: "2_data_ct_filter_all_{date}_v{number}_ollama.json" - export all classification results
# 2. OUTPUT file B: "2_data_ct_filter_yes_{date}_v{number}_ollama.json" - export only CT posts (used for the next step in the pipeline)

#NOTE: this file uses Gemma3:12b via Ollama instead of GPT-4.1-nano
#NOTE: for OpenAI models, refer to `2_data_ct_filter.py` file

import pandas as pd
import os
import time
import ollama
from tqdm import tqdm

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Using the A100 GPU

# Load JSONL file (each line is a separate JSON object)
INPUT = "ct_platform/ct_nar_platform_git/pipeline_data/1_data_prep_20250702_v1.json"
df = pd.read_json(INPUT, lines=True)
# sample 10 posts for testing
# df = df.sample(75, random_state=42).reset_index(drop=True)
# Get current date for file naming
DATE = time.strftime("%Y%m%d")
VERSION = 1
LLM = 'gemma3:12b'  # Specify the Ollama model to use

# Function to process batch of texts (25 at a time for efficiency)
def ollama_ct_batch_classifier(texts_batch):
    global total_api_calls, total_input_tokens, total_output_tokens
    
    try:
        # Format documents with numbering
        formatted_docs = []
        for i, text in enumerate(texts_batch, 1):
            formatted_docs.append(f"Document {i}: {text}")
        
        documents_text = "\n\n".join(formatted_docs)
        
        # Create batch prompt using the full detailed prompt
        batch_prompt = f"""
        Your task is to determine whether each given text contains or reflects statements that are related to any conspiracy theories 
        that are relevant to aliens, covid-19, 911, illuminati/new world order, or moon landing. 
        A conspiracy theory is an explanation for an event or situation that invokes a conspiracy by powerful people or organizations, often without credible evidence. 
        Conspiracy theories often involve claims of secret plots, coverups, or the manipulation of information by influential groups. 

        Here are some examples of conspiracy theories: 
        1. "The moon landing was faked by the U.S. government to win the space race." 
        2. "The COVID-19 pandemic was planned and orchestrated by pharmaceutical companies to profit from vaccine sales." 
        3. "Climate change is a hoax perpetrated by scientists and politicians to gain funding and control the population." 

        And here are some examples of statements that are NOT conspiracy theories: 
        4. "The Watergate scandal involved a cover-up of illegal activities by the Nixon administration." 
        5. "The tobacco industry concealed the harmful effects of smoking for many years." 
        6. "Corporate lobbying influences political decisions in favor of special interests." 

        For each document provided below, you MUST respond with either "yes" or "no" to indicate whether it contains conspiracy theory content.
        For statements that you are not sure about, unknown or lacking context information, respond with "no".
        You MUST provide exactly one answer per document, separated by commas (e.g., "yes,no,yes,no,yes").
        You MUST NOT include any other text in your response.

        Here are the {len(texts_batch)} documents to classify:
        {documents_text}
        """
        
        # Make the Ollama API request
        response = ollama.chat(
            model=LLM,
            messages=[
                {"role": "system", "content": "You are a helpful fact-checking assistant that is expert in detecting conspiracy theories."},
                {"role": "user", "content": batch_prompt}
            ],
            options={"temperature": 0, "num_ctx": 16384,}
        )
        
        # Track API usage for processing statistics
        total_api_calls += 1
        # Estimate token usage (Ollama doesn't provide exact counts)
        estimated_input = len(batch_prompt.split()) * 1.3  # rough token estimate
        estimated_output = len(response["message"]["content"].split()) * 1.3
        total_input_tokens += int(estimated_input)
        total_output_tokens += int(estimated_output)
        
        # Extract and parse the response
        result = response["message"]["content"].strip()
        
        # Parse comma-separated results
        results = [r.strip().lower() for r in result.split(',')]
        
        # Ensure we have the right number of results
        if len(results) != len(texts_batch):
            # Pad with 'no' if we don't have enough results
            while len(results) < len(texts_batch):
                results.append('no')
            # Truncate if we have too many
            results = results[:len(texts_batch)]
        
        # Ensure all results are valid (yes/no)
        valid_results = []
        for r in results:
            valid_results.append("yes" if r == "yes" else "no")
        
        return valid_results
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return 'no' for all texts in the batch on error
        return ["no"] * len(texts_batch)



# Processing tracking variables
total_api_calls = 0
total_input_tokens = 0
total_output_tokens = 0

print(f"Loaded {len(df)} posts for CT classification")
print(f"Starting Ollama batch classification with {LLM} (25 posts per batch)...")

# Process in batches of 25 for efficiency
BATCH_SIZE = 25
all_results = []

# Split data into batches
texts = df['post_text'].tolist()
num_batches = len(texts) // BATCH_SIZE + (1 if len(texts) % BATCH_SIZE != 0 else 0)

print(f"Processing {len(texts)} posts in {num_batches} batches...")

# Process each batch with progress tracking
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_results = ollama_ct_batch_classifier(batch_texts)
    all_results.extend(batch_results)
    
    # Optional: Add a small delay between batches to manage GPU load
    time.sleep(0.5)

# Add results to dataframe
df['ct'] = all_results

print(f"Classification completed!")
print(f"Classification results:")
print(df['ct'].value_counts())

# Display processing summary
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total API calls made: {total_api_calls}")
print(f"Estimated input tokens: {total_input_tokens:,}")
print(f"Estimated output tokens: {total_output_tokens:,}")
print(f"Total estimated tokens: {total_input_tokens + total_output_tokens:,}")

if len(df) > 0:
    print(f"Average tokens per document: {(total_input_tokens + total_output_tokens) / len(df):.1f}")

print(f"Model used: {LLM}")
print(f"GPU device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 25)

# Create output file paths
output_dir = "ct_platform/ct_nar_platform_git/pipeline_data"
output_file_all = f"{output_dir}/2_data_ct_filter_all_{DATE}_v{VERSION}.json"
output_file_yes = f"{output_dir}/2_data_ct_filter_yes_{DATE}_v{VERSION}.json"

# Save all classification results
df.to_json(output_file_all, orient='records', lines=True, force_ascii=False)
print(f"All classification results saved to: {output_file_all}")

# Filter and save only CT posts (ct == "yes")
df_ct_only = df[df['ct'] == 'yes'].copy()
df_ct_only.to_json(output_file_yes, orient='records', lines=True, force_ascii=False)
print(f"CT posts only saved to: {output_file_yes}")
print(f"CT posts found: {len(df_ct_only)} out of {len(df)} total posts ({len(df_ct_only)/len(df)*100:.1f}%)")
