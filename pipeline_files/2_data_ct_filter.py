### INPUT: json file with structured data for each posFor each statement provided, you MUST only respond with either "yes" or "no" to indicate your answer.
# input file: "1_data_prep_{date}_v{number}.json"

### OUTPUT: json file with data that are classified as CT posts by LLMs
## expected output data format
# 1. with one new column `ct` - indicating whether the post is classified as CT or not "yes/no"
# 2. keep all other columns from the input file
# 3. keep only posts that are classified as CT posts (i.e. `ct` column is "yes")

### OUTPUT NAMING RULES:
# There are two OUTPUT files 
# 1. OUTPUT file A: "2_data_ct_filter_all_{date}_v{number}.json" - export all classification results
# 2. OUTPUT file B: "2_data_ct_filter_yes_{date}_v{number}.json" - export only CT posts (used for the next step in the pipeline)

#NOTE: we use GPT-4.1-nano model for this task
#NOTE: for opensourced LLMs models, plz refer to `2_1_data_ct_filter_ollama.py` file for reference

import pandas as pd
import os
import time
from openai import OpenAI
from tqdm import tqdm

# Load OpenAI key from environment variable
openai_key = os.getenv("OPENROUTER_API_KEY_mohsen")
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=openai_key)


# load JSONL file (each line is a separate JSON object)
INPUT = "ct_platform/ct_nar_platform_git/pipeline_data/1_data_prep_20250702_v1.json"
df = pd.read_json(INPUT, lines=True)
# sample 10 posts for testing
df = df.sample(10, random_state=42).reset_index(drop=True)

LLM = 'gpt-4o-mini' # "gpt-4.1-nano"  # Specify the model to use

# Function to process batch of texts (25 at a time for budget efficiency)
def gpt_ct_batch_classifier(texts_batch):
    global total_api_calls, total_input_tokens, total_output_tokens
    
    try:
        # Format documents with numbering
        formatted_docs = []
        for i, text in enumerate(texts_batch, 1):
            formatted_docs.append(f"Document {i}: {text}")
        
        documents_text = "\n\n".join(formatted_docs)
        
        # Create batch prompt using the original detailed prompt
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
        
        # Make the API request
        response = client.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "You are a helpful fact-checking assistant that is expert in detecting conspiracy theories."},
                {"role": "user", "content": batch_prompt}
            ],
            temperature=0
        )
        
        # Track API usage for budget calculation
        total_api_calls += 1
        if hasattr(response, 'usage'):
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
        else:
            # Fallback: rough estimate if usage data not available
            estimated_input = len(batch_prompt.split()) * 1.3  # rough token estimate
            estimated_output = len(texts_batch) * 2  # assume ~2 tokens per response
            total_input_tokens += int(estimated_input)
            total_output_tokens += int(estimated_output)
            print(f"Warning: Token usage not available, using estimates")
        
        # Extract and parse the response
        result = response.choices[0].message.content.strip()
        
        # Parse comma-separated results
        results = [r.strip().lower() for r in result.split(',')]
        
        # Ensure we have the right number of results
        if len(results) != len(texts_batch):
            print(f"Warning: Expected {len(texts_batch)} results, got {len(results)}. Using 'no' for missing results.")
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

# Get current date for file naming
date = time.strftime("%Y%m%d")
version = 1

# Budget tracking variables
total_api_calls = 0
total_input_tokens = 0
total_output_tokens = 0

# GPT-4.1-nano pricing (approximate rates - adjust as needed)
INPUT_TOKEN_COST = 0.0000002  # $0.0002 per 1K tokens
OUTPUT_TOKEN_COST = 0.0000004  # $0.0004 per 1K tokens

print(f"Loaded {len(df)} posts for CT classification")
print(f"Starting GPT batch classification (25 posts per batch)...")

# Process in batches of 25 for budget efficiency
BATCH_SIZE = 25
all_results = []

# Split data into batches
texts = df['clean_post_text'].tolist()
num_batches = len(texts) // BATCH_SIZE + (1 if len(texts) % BATCH_SIZE != 0 else 0)

print(f"Processing {len(texts)} posts in {num_batches} batches...")

# Process each batch with progress tracking
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_results = gpt_ct_batch_classifier(batch_texts)
    all_results.extend(batch_results)
    
    # Optional: Add a small delay between batches to avoid rate limits
    time.sleep(1)

# Add results to dataframe
df['ct'] = all_results

print(f"Classification completed!")
print(f"Classification results:")
print(df['ct'].value_counts())

# Calculate and display budget information
total_cost = (total_input_tokens * INPUT_TOKEN_COST) + (total_output_tokens * OUTPUT_TOKEN_COST)

print(f"\n=== BUDGET SUMMARY ===")
print(f"Total API calls made: {total_api_calls}")
print(f"Total input tokens: {total_input_tokens:,}")
print(f"Total output tokens: {total_output_tokens:,}")
print(f"Total tokens used: {total_input_tokens + total_output_tokens:,}")
print(f"Estimated cost: ${total_cost:.4f}")

if len(df) > 0:
    print(f"Average tokens per document: {(total_input_tokens + total_output_tokens) / len(df):.1f}")
    print(f"Average cost per document: ${total_cost / len(df):.6f}")
else:
    print("No documents processed")

# Compare with individual API call approach
if total_cost > 0:
    individual_call_cost = total_cost * 25  # Approximate cost if we made individual calls
    savings = individual_call_cost - total_cost
    print(f"\nCost savings from batch processing:")
    print(f"Individual calls would cost: ~${individual_call_cost:.4f}")
    print(f"Batch processing cost: ${total_cost:.4f}")
    print(f"Money saved: ~${savings:.4f} ({(savings/individual_call_cost)*100:.1f}% reduction)")
else:
    print("\nNo cost data available for comparison")
print("=" * 25)

# Create output file paths
output_dir = "ct_platform/ct_nar_platform_git/pipeline_data"
output_file_all = f"{output_dir}/2_data_ct_filter_all_{date}_v{version}.json"
output_file_yes = f"{output_dir}/2_data_ct_filter_yes_{date}_v{version}.json"

# Save all classification results
df.to_json(output_file_all, orient='records', lines=True, force_ascii=False)
print(f"All classification results saved to: {output_file_all}")

# Filter and save only CT posts (ct == "yes")
df_ct_only = df[df['ct'] == 'yes'].copy()
df_ct_only.to_json(output_file_yes, orient='records', lines=True, force_ascii=False)
print(f"CT posts only saved to: {output_file_yes}")
print(f"CT posts found: {len(df_ct_only)} out of {len(df)} total posts ({len(df_ct_only)/len(df)*100:.1f}%)")




