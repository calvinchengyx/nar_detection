#### INPUT: raw data collected from the keyword-data collection step

### OUTPUT: json file with structured data for each post
## expected output data should be in the format with: 
## 1. post_id - the unique identifier for each post
## 2. clean_post_text - the cleaned text of the post (preliminary preprocessing, removed HTML tags, special characters, mentions, hashtags etc.)
## 3. text: the original text of the post
## 4. other metadata fields as needed

### OUTPUT NAMING RULES:
##  The output file should be named as: "1_data_prep_{date}_v{number}.json"

#NOTE: see the `1_1_data_prep_format.ipynb` and `1_2_data_ct_selection.ipynb` notebooks for more details on the data preparation steps. The key here is to 
# 1. understand the data structure and format collected from different sources
# 2. do a preliminary data cleaning and preprocessing

# This file is for demonstration purposes with a sample dataset where it contains 
# 1. single topic - "new_world_order" conspiracy topic 
# 2. from three platforms: truthsocial, bluesky and twitter
# 3. downsized Twitter data to comparable size with other platforms

# INPUT: "ct_platform/data/extract_tem4_sample.csv"

import pandas as pd
import time

date = time.strftime("%Y%m%d")

# load the structured data
df = pd.read_pickle("ct_platform/data/extract_tem4.pkl")
print(f"Initial dataset: {df['post_id'].nunique()} unique posts loaded")


###### PREPROCESSING STEPS ####### 

##### 1. Language Filter - remove non-English texts #####
# Filter 1 Language Filter 
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm

# Set seed for consistent results
DetectorFactory.seed = 0

# Enable tqdm for pandas
tqdm.pandas()

def detect_language(text):
    try:
        # Check if text exists and has enough content
        if pd.notnull(text) and len(text.strip()) > 3:
            # Remove extra whitespace and check again
            clean_text = ' '.join(text.split())
            if len(clean_text) > 3:
                return detect(clean_text)
        return 'unknown'
    except (LangDetectException, Exception):
        return 'unknown'

# Use progress_apply instead of apply
df['lang'] = df['clean_post_text'].progress_apply(detect_language) # it will take about 6 min to run
df = df[df['lang'] == 'en'].reset_index(drop=True)
# drop the 'lang' column as it's no longer needed
df.drop(columns=['lang'], inplace=True)
print(f"There are unique {df['post_id'].nunique()} posts after filtering by language (English)")

##### 2. TEXT Length Filter - remove relatively short texts that are difficult to interpret without enough context #####
# NOTE: the text length threshold is set to 14 words in this context (after pre-cleaning)
# the threshold needs qualitative assessment, so it can be adjusted later

def text_length_filter(df, MIN_TOKENS=14):
    # Step 1: Remove hashtags (e.g., #NewWorldOrder, #conspiracy)
    df['clean_text_v2'] = df['clean_post_text'].str.replace(r'#\w+', '', regex=True)

    # Step 2: Remove mentions (e.g., @username)
    df['clean_text_v2'] = df['clean_text_v2'].str.replace(r'@\w+', '', regex=True)

    # Step 3: Remove HTML entities
    df['clean_text_v2'] = df['clean_text_v2'].str.replace(
        r'&amp;|&lt;|&gt;|&quot;|&apos;', '', regex=True
    )
    # Step 4: Clean up extra whitespace
    df['clean_text_v2'] = df['clean_text_v2'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Step 5: Calculate token length and filter
    df['token_count'] = df['clean_text_v2'].str.split().str.len()
    df_filtered = df[df['token_count'] > MIN_TOKENS].copy()
    df_filtered.drop(columns=['clean_text_v2', 'token_count'], inplace=True)
    return df_filtered

df = text_length_filter(df, MIN_TOKENS=14)
print(f"There are {df['post_id'].nunique()} unique posts after filtering by text length (14+ words)")

##### 3. PLATFORM AND TOPIC FILTERING #####

# selected platforms and topic
platforms = ['bluesky', 'truthsocial','twitter']
topic_ct = 'Illuminati/New World Order'
df = df[df['platform'].isin(platforms) & (df['topic_5'] == topic_ct)]
print(f"After platform and topic filtering: {len(df)} posts remaining")
print("The number of posts in each platform:")
print(df['platform'].value_counts())

## I used an equal sampling strategy to balance the number of posts across platforms
# Use the smallest platform size as the sample size
min_size = df['platform'].value_counts().min() 
# Sample equally from each platform
sample = df.groupby('platform').apply(
    lambda x: x.sample(n=min_size, random_state=42)
).reset_index(drop=True)

print("The number of posts in each platform after balancing:")
print(sample['platform'].value_counts())

# save the output as json file
# manually set the version number each time you change the code 
version = 1
output_file = f"ct_platform/ct_nar_platform_git/pipeline_data/1_data_prep_{date}_v{version}.json"
sample.to_json(output_file, orient='records', lines=True, force_ascii=False)
print(f"Data successfully saved to: {output_file}")
print(f"Final dataset: {len(sample)} posts, balanced across {len(platforms)} platforms")