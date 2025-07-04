### INPUT: json file with ct classification results
# input file: "3_sem_match_claim_{date}_v{number}.json"

### processing steps:
# 1. for each "content_id", concatenate all `clean_post_text` into one string, saving to a new column "content_text" 
# 2. create a new df, with `content_id` and `content_text` column 
# 3. feed into LLMs for post-2-claim information extraction (information retrieval tasks)
# 3.1 expected output is each "content_id" is associated with a list of atomic claims strings
# 3.2 claims should be non-duplicated atomic clear propositions
# 3.3 each claim is also associatad with a unique identifier `claim_id` (e.g. "{content_id}_{claim_number}")

### OUTPUT: json file with claim information 

### OUTPUT NAMING RULES:
# there is one output file: "4_post2claim_{date}_v{number}.json" , format as [{"content_id_1": {"claim_id_1": "claim_text",
                                                                                            # "claim_id_2": "claim_text", ...
                                                                                            # },
                                                                                # } 
                                                                            #  {"content_id_2": {"claim_id_1": "claim_text",
                                                                                            # "claim_id_2": "claim_text", ...
                                                                                            # }
                                                                                #  } 
                                                                            #  ... ]


import pandas as pd
import os
import time
import json
from openai import OpenAI
from tqdm import tqdm

# configure the model and input file
LLM = 'gpt-4o-mini' # "gpt-4.1-nano"  # Specify the model to use
INPUT = "ct_platform/ct_nar_platform_git/pipeline_data/3_sem_match_claim_20250702_v1.json"
OUTPUT_FILE = "ct_platform/ct_nar_platform_git/pipeline_data"
DATE = time.strftime("%Y%m%d")
VERSION = 1

# Load OpenAI key from environment variable
openai_key = os.getenv("")
client = OpenAI(base_url="",
                api_key=openai_key)
df = pd.read_json(INPUT, lines=True)
# df = df.sample(10, random_state=42).reset_index(drop=True)  # Sample 10 posts for testing

# Group by content_id and concatenate all clean_post_text into one string
content_groups = df.groupby('content_id')['clean_post_text'].apply(lambda x: ' '.join(x)).reset_index()
content_groups.columns = ['content_id', 'content_text']

# Create the final dataframe for processing with content_id and content_text columns
p2c = content_groups.copy()
p2c.columns = ['content_id', 'content_text']  # Ensure proper column names

total_rows = len(p2c)
print(f"Loaded {total_rows} content_id entries.")


few_shot_examples = """"
Input: "Stars are in the firmament that cannot be breached by anything even though the evil ones have tried by shooting rockets at it and all the rockets have been destroyed. None had any humans in them. NASA films so called moon landings on a Hollyweird set and NASA is just a way for them to get money from We, the taxpayers, and money launder for all their fake projects"
Output:["Stars are in the firmament that cannot be breached by anything.",
        "Evil ones have tried to breach the firmament by shooting rockets at it.",
        "All rockets shot at the firmament have been destroyed.",
        "None of the rockets shot at the firmament had any humans in them.",
        "NASA films moon landings on a Hollywood set.",
        "NASA is a way for them to get money from taxpayers.",
        "NASA launder money for all their fake projects."
        ]

Input: "NEW pre-print Study points to DEPOPULATION Injection Young adults between the ages of 15 and 44 are experiencing more neurological disease than ever before in our nation's history. And data from the U.S. Centers for Disease Control and Prevention (CDC) points to Wuhan coronavirus (COVID-19) vaccines as the Technologies put together a study using this CDC data showing a sharp rise in neurological disease-induced death within this age group that started in 2020. Conditions like Guillain-Barre syndrome (GBS) and acute disseminated encephalomyelitis that are commonly associated with vaccination suddenly started afflicting the 15-44 age group, many from which were forced"
Output: ["COVID-19 vaccines are actually a depopulation injection",
        "Young adults aged 15 to 44 are experiencing more neurological disease than ever before in U.S. history",
        "CDC data indicates that COVID-19 vaccines are linked to a sharp rise in neurological disease–induced deaths in people aged 15 to 44 starting in 2020",
        "Vaccination-related conditions such as Guillain–Barré syndrome and acute disseminated encephalomyelitis have suddenly begun afflicting individuals aged 15 to 44",
        "Many individuals aged 15 to 44 were forced to receive COVID-19 vaccines"
        ]

Input: "LIVING IN THE MATRIX There is a trend sweeping the world. People are noticing something isn’t right. When you cut away all the fancy wrapping paper and look at our lives today you realize we are truly enslaved. We’re still being pushed the same goals in life. Have a beautiful loving family. Own a home and multiple vehicles. Vacation often and travel overseas. Be a positive contributor to the society. "
Output: ["People are living in the Matrix",
        "There is a global trend of people realizing something is not right.",
        "people are truly enslaved",
        "Society pushes same life goals as a form of control",
        "Society pressures individuals to have a beautiful, loving family",
        "Society pressures individuals to own a home and multiple vehicles",
        "Society pressures individuals to vacation often and travel overseas",
        "Society pressures individuals to be positive contributors to society"
        ]

Input: "Self-sterilization! Hitler would love it! All part of the “great reset” plan, as we enter the final phase. When she was a teenager, Cristina Hineman started testosterone after a 30-minute consult at Planned Parenthood. She’s now suing them. ‘I regretted everything.'"
Output: ["Self-sterilization is part of the “great reset” plan",
        "The “great reset” plan is entering its final phase",
        "Adolf Hitler would love the self-sterilization", 
        "Cristina Hineman started testosterone after a 30-minute consult at Planned Parenthood",
        "Cristina Hineman is now suing Planned Parenthood",
        "Cristina Hineman regretted everything"
        ]

Input: "And the burning building at top did not collapse below to the other floors. How in the Heaven's names can people be so ridiculously STUPID and these Reptilians be so much smarter. Satan is the father of all lies with his minions in high places!!"
Output: ["The burning building at the top did not collapse to the other floors.",
        "Reptilians are much smarter than humans.",
        "Satan is the father of all lies.",
        "Satan has minions in high places."
        ]

Input: "It's almost time for the UK electorate to go out and vote for a new leader. July 4th will be the day when the whole UK electorate consents to be governed by the New World Order. Keir Starmer of the Trilateral Commission will represent them in England. He was selected for the job many years ago, after he gained notoriety for failing to jail Jimmy Savile. I cannot say whether, or not, he was recruited for the TC by Jeffrey Epstein, but it would not be a surprise. Consent or Deregister."
Output: ["The New World Order will govern the UK electorate starting on July 4th",
        "July 4th is the day when the UK electorate consents to be governed by the New World Order",
        "Keir Starmer is a representative of the Trilateral Commission in England",
        "Keir Starmer was selected years ago to represent the UK in the New World Order",
        "Keir Starmer’s selection was due to his failure to jail Jimmy Savile",
        "Jeffrey Epstein recruited Keir Starmer into the Trilateral Commission",
        "You must consent or deregister."
        ]
"""

def process_conspiracy_claims_batched(dataframe, num_records=1000, num_batches=40, output_file=None):
    """
    Process conspiracy claim extraction with efficient batch processing and batch-level checkpointing
    
    Args:
        dataframe: The p2c DataFrame containing content_id and content_text columns
        num_records: Number of records to process (default 1000)
        num_batches: Number of batches to create (default 40, max 25 docs per batch)
        output_file: File path to save results after each batch
    
    Returns:
        tuple: (results_dict, budget_info_dict)
    """
    
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    
    # Calculate optimal batch size (max 25 documents per batch)
    batch_size = min(25, max(1, num_records // num_batches))
    actual_batches = (num_records + batch_size - 1) // batch_size
    
    # Initialize tracking variables
    total_input_tokens = 0
    total_output_tokens = 0
    results = {}
    processed_count = 0
    
    # GPT-4o-mini pricing
    input_cost_per_token = 0.15 / 1_000_000
    output_cost_per_token = 0.60 / 1_000_000
    
    # =============================================================================
    # PROCESSING SETUP
    # =============================================================================
    
    print("=" * 70)
    print("CONSPIRACY CLAIMS EXTRACTION - BATCH PROCESSING")
    print("=" * 70)
    print(f"Records to process: {min(num_records, len(dataframe))}")
    print(f"Available records: {len(dataframe)}")
    print(f"Requested batches: {num_batches}")
    print(f"Actual batches: {actual_batches}")
    print(f"Batch size: {batch_size} (max 25 documents per batch)")
    print("-" * 70)
    
    # Get subset of data to process
    data_to_process = dataframe.head(num_records)
    
    # =============================================================================
    # BATCH PROCESSING LOOP WITH PROGRESS BAR
    # =============================================================================
    
    batch_ranges = list(range(0, len(data_to_process), batch_size))
    
    for batch_num, start_idx in enumerate(tqdm(batch_ranges, desc="Processing batches", unit="batch"), 1):
        end_idx = min(start_idx + batch_size, len(data_to_process))
        batch = data_to_process.iloc[start_idx:end_idx]
        
        # Create batch prompt with your pre-defined structure
        batch_prompt = f"""
                        You are a helpful fact-checking assistant that excellent in extracting concise conspiracy claims from provided documents.

                        Your task is to process multiple documents and for EACH document, list all conspiracy present and reformulate them as clear atomic claims. 
                        A claim should be atomic (only contain one information), decontextualized (not require additional information to be understood), faithful to the source text and fluent (gramatically correct and intelligible with minimum rephrasing).
                        Your task is not to question the truth of the claims, but only to identify and extract them.

                        You MUST return results in the following JSON format:
                        {{
                        "doc_1": ["Claim 1 from doc 1", "Claim 2 from doc 1", ...],
                        "doc_2": ["Claim 1 from doc 2", "Claim 2 from doc 2", ...],
                        ...
                        }}

                        Here are some correct examples:
                        {few_shot_examples}

                        Here are the documents to process:
                        """
        
        # Build batch documents
        batch_docs = ""
        doc_id_mapping = {}
        for i, (_, row) in enumerate(batch.iterrows(), 1):
            doc_key = f"doc_{i}"
            doc_id_mapping[doc_key] = str(row['content_id'])  # Map doc_key to content_id as string
            batch_docs += f"\n--- Document {i} (ID: {doc_key}) ---\n{row['content_text']}\n"
        
        batch_prompt += batch_docs
        
        # =============================================================================
        # API CALL WITH ERROR HANDLING
        # =============================================================================
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a conspiracy claim extraction specialist. Return valid JSON only."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0
            )
            
            # Extract response and update token counts
            response_text = response.choices[0].message.content.strip()
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            
            # =============================================================================
            # PARSE BATCH RESULTS
            # =============================================================================
            
            try:
                # Clean JSON response
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                batch_results = json.loads(response_text)
                
                # Map results back to original claim_ids
                for doc_key, claims in batch_results.items():
                    if doc_key in doc_id_mapping:
                        original_claim_id = doc_id_mapping[doc_key]
                        results[original_claim_id] = claims
                        processed_count += 1
                
                # Save results after each successful batch (checkpoint)
                if output_file:
                    save_batch_results(results, output_file)
                
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON parsing error for batch {batch_num}: {e}")
                print(f"Skipping batch {batch_num} due to parsing error")
                continue
            
            # =============================================================================
            # PROGRESS REPORTING
            # =============================================================================
            
            if batch_num % 10 == 0 or batch_num == actual_batches:
                current_cost = (total_input_tokens * input_cost_per_token + 
                               total_output_tokens * output_cost_per_token)
                print(f"PROGRESS: Batch {batch_num}/{actual_batches} | Records: {processed_count} | Cost: ${current_cost:.4f}")
            
            # Save results after each batch
            if output_file:
                save_batch_results(results, output_file)
            
            # Rate limiting delay
            time.sleep(0.2)
            
        except Exception as e:
            print(f"ERROR: Error processing batch {batch_num}: {str(e)}")
            continue
    
    # =============================================================================
    # FINAL BUDGET REPORT
    # =============================================================================
    
    input_cost = total_input_tokens * input_cost_per_token
    output_cost = total_output_tokens * output_cost_per_token
    total_cost = input_cost + output_cost
    
    print("\n" + "=" * 70)
    print("FINAL PROCESSING REPORT")
    print("=" * 70)
    print(f"Records processed: {processed_count:,}")
    print(f"Total claims extracted: {sum(len(claims) for claims in results.values()):,}")
    print(f"Average claims per record: {sum(len(claims) for claims in results.values()) / max(len(results), 1):.2f}")
    print("-" * 70)
    print("BUDGET BREAKDOWN:")
    print(f"   Input tokens: {total_input_tokens:,} (${input_cost:.4f})")
    print(f"   Output tokens: {total_output_tokens:,} (${output_cost:.4f})")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Cost per record: ${total_cost/max(processed_count, 1):.4f}")
    print("=" * 70)
    
    # Prepare return data
    budget_info = {
        'processed_count': processed_count,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_cost': total_cost,
        'cost_per_record': total_cost / max(processed_count, 1),
        'num_batches': actual_batches,
        'batch_size': batch_size,
        'total_claims': sum(len(claims) for claims in results.values()),
        'avg_claims_per_record': sum(len(claims) for claims in results.values()) / max(len(results), 1)
    }
    
    return results, budget_info


def load_existing_results(file_prefix="p2ct_gpt_processed"):
    """
    Load existing results to resume processing from where we left off
    
    Args:
        file_prefix: Prefix of the results file to load
    
    Returns:
        tuple: (existing_results_dict, processed_claim_ids_set)
    """
    results_file = f"{OUTPUT_FILE}/{file_prefix}.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        processed_ids = set(existing_results.keys())
        print(f"Loaded existing results: {len(existing_results)} records already processed")
        return existing_results, processed_ids
    
    except FileNotFoundError:
        print("No existing results file found. Starting fresh.")
        return {}, set()
    
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return {}, set()

def process_conspiracy_claims_with_checkpoint(dataframe, num_records=1000, num_batches=40, 
                                            file_prefix="p2ct_gpt_processed", 
                                            resume=True, output_file=None):
    """
    Process conspiracy claims with checkpoint/resume capability
    
    Args:
        dataframe: The p2c DataFrame containing content_id and content_text columns
        num_records: Number of records to process (default 1000)
        num_batches: Number of batches to create (default 40, max 25 docs per batch)
        file_prefix: Prefix for checkpoint files
        resume: Whether to resume from existing checkpoint (default True)
        output_file: File path to save results after each batch
    
    Returns:
        tuple: (results_dict, budget_info_dict)
    """
    
    # Load existing results if resuming
    if resume:
        existing_results, processed_ids = load_existing_results(file_prefix)
    else:
        existing_results, processed_ids = {}, set()
    
    # Filter out already processed records
    if processed_ids:
        remaining_data = dataframe[~dataframe['content_id'].astype(str).isin(processed_ids)]
        print(f"Resuming: {len(processed_ids)} already done, {len(remaining_data)} remaining")
    else:
        remaining_data = dataframe
        print(f"Starting fresh: {len(remaining_data)} total records")
    
    # Adjust num_records based on what's left to process
    records_to_process = min(num_records - len(processed_ids), len(remaining_data))
    
    if records_to_process <= 0:
        print(f"All {num_records} records already processed!")
        return existing_results, {
            'processed_count': len(existing_results),
            'total_cost': 0,
            'message': 'All records already processed'
        }
    
    print(f"Will process {records_to_process} new records")
    
    # Process remaining records
    new_results, budget_info = process_conspiracy_claims_batched(
        dataframe=remaining_data,
        num_records=records_to_process,
        num_batches=max(1, records_to_process // 25),  # Adjust batches based on remaining work
        output_file=output_file
    )
    
    # Merge with existing results
    combined_results = {**existing_results, **new_results}
    
    # Update budget info to reflect total processing
    budget_info['total_processed_count'] = len(combined_results)
    budget_info['new_records_processed'] = len(new_results)
    budget_info['resumed_from_checkpoint'] = len(existing_results)
    
    return combined_results, budget_info

def format_output_for_pipeline(results):
    """
    Format results to match the expected pipeline output format:
    [{"content_id_1": {"claim_id_1": "claim_text", "claim_id_2": "claim_text", ...}}, ...]
    """
    formatted_output = []
    
    for content_id, claims_list in results.items():
        content_dict = {}
        claims_dict = {}
        
        for i, claim_text in enumerate(claims_list, 1):
            claim_id = f"{content_id}_{i}"  # content_id + number of claims
            claims_dict[claim_id] = claim_text
        
        content_dict[str(content_id)] = claims_dict
        formatted_output.append(content_dict)
    
    return formatted_output

def save_batch_results(results, output_file):
    """
    Save current results to file after each batch (checkpoint)
    
    Args:
        results: Dictionary of content_id -> list of claims
        output_file: File path to save results
    """
    try:
        # Format output for pipeline
        formatted_results = format_output_for_pipeline(results)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Failed to save batch results: {e}")

def main():
    """Main execution function"""
    print("="*60)
    print("POST-TO-CLAIM EXTRACTION PIPELINE")
    print("="*60)
    
    # Define output file path
    output_file = f"{OUTPUT_FILE}/4_post2claim_{DATE}_v{VERSION}.json"
    
    # Process claims with checkpoint capability
    results, budget_info = process_conspiracy_claims_with_checkpoint(
        dataframe=p2c,
        num_records=len(p2c),  # Process all records
        num_batches=max(1, len(p2c) // 25),  # Optimal batch size
        file_prefix="4_post2claim_20250702_v1",
        resume=True,
        output_file=output_file
    )
    
    # Final save (already saved after each batch, but ensure final state)
    formatted_results = format_output_for_pipeline(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal output saved to: {output_file}")
    print(f"Processed {budget_info.get('total_processed_count', 0)} content entries")
    
    # Print budget information at the end
    print("\n" + "="*60)
    print("FINAL BUDGET SUMMARY")
    print("="*60)
    print(f"Total cost: ${budget_info.get('total_cost', 0):.4f}")
    print(f"Input tokens: {budget_info.get('total_input_tokens', 0):,}")
    print(f"Output tokens: {budget_info.get('total_output_tokens', 0):,}")
    print(f"Cost per record: ${budget_info.get('cost_per_record', 0):.4f}")
    print(f"Total claims extracted: {budget_info.get('total_claims', 0):,}")
    print("="*60)

if __name__ == "__main__":
    main()

