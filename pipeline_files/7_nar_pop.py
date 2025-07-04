### INPUT
# there are multiple input files:
# "6_topic2nar_topic_{date}_v{number}.json", with topic_id, rep words, rep_docs and nar_text. 
# "5_claim2topic_topic_{date}_v{number}.json" with topic_id, rep words, rep docs and rep_doc_ctfidf for later MMR
# "5_claim2topic_doctopic_{date}_v{number}.json" with claim_id, claim_text, topic_id columns, so each claim_id is associated with each topic_id
# "4_post2claim_{date}_v{number}.json" with content_id, content_text, claim_id columns, so each content_id is associated with each claim_id
# "3_sem_match_claim_{date}_v{number}.json" with post_id, content_id, clean_post_text columns, so each post_id is associated with each content_id
# "2_data_ct_filter_yes_{date}_v{number}.json" with post_id, clean_post_text columns, so each post_id is associated with each clean_post_text

### processing steps:
# 1. merge "topic_id", "nar_text", back to "claim_id"
# 2. merge "claim_id" back to "content_id"
# 3. merge "content_id" back to "post_id"
# 4. as a result, we want to link engagement metrics with "claim_id" AND "topic_id"

### OUTPUT
# "7_claim_eng_{date}_v{number}.json", with claim_id, claim_text, topic_id, nar_text, post_id, clean_post_text, engagement metrics

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

DATE = time.strftime("%Y%m%d")
VERSION = 1
SERVER = "ct_platform/ct_nar_platform_git"
OUTPUT_PATH = f"{SERVER}/pipeline_data/7_nar_pop_{DATE}_v{VERSION}.png"

###### merge engagement metrics with claim_id and topic_id
print("Loading data...")
INPUT_1 = f"{SERVER}/pipeline_data/6_topic2nar_20250703_v1.json"

with open(INPUT_1, 'r') as file:
    data_topic = json.load(file)
df = pd.DataFrame(data_topic)
df = df[["topic", "mmr_30_nar"]]
df['mmr_30_nar'] = df['mmr_30_nar'].str.replace('topic: ', '').str.strip()
df_nar = df.copy()

INPUT_2 = f"{SERVER}/pipeline_data/5_claim2topic_doctopic_20250703_v1.json"
with open(INPUT_2, 'r') as file:
    data_claim = json.load(file)
df_claim = pd.DataFrame(data_claim)
df_claim = df_claim[['content_id', 'claim_id', 'claim_text', 'topic']] 
for col in df_claim.columns:
    # covert to str
    df_claim[col] = df_claim[col].astype(str)

INPUT_3 = f"{SERVER}/pipeline_data/3_sem_match_20250702_v1.json"
data_sem = []
with open(INPUT_3, 'r') as file:
    for line in file:
        data_sem.append(json.loads(line.strip()))
df_sem = pd.DataFrame(data_sem)
df_sem = df_sem[['post_id', 'content_id', 'post_time', 'post_text','clean_post_text',
                'like', 'reply', 'repost', 
                'user_id', 'user_name', 'user_fans', 'user_follow', 'user_veri',
                'platform']]

# create a new df with 4 columns: 'content_id', 'like', 'reply', 'repost',
# where like, reply and repost are the avg of like, reply and repost within content_id group, mean by number of post_id
df_content_eng = df_sem.groupby('content_id').agg({
    'like': 'mean',
    'reply': 'mean', 
    'repost': 'mean'
}).reset_index()

# Rename columns to be clear they are averages
df_content_eng.columns = ['content_id', 'avg_like', 'avg_reply', 'avg_repost']
df_content_eng = df_content_eng.merge(df_sem[['content_id', 'platform']].drop_duplicates(), on='content_id', how='left')
df_content_eng['content_id'] = df_content_eng['content_id'].astype(str)

# base df is claims
df_eng = df_claim.copy()
# merge nar_description to the topic, so each claim is associated with its nar description
df_eng = df_eng.merge(df_nar, on='topic', how='left') 
df_eng = df_eng.merge(df_content_eng, on='content_id', how='left')
# remove outliers 
df_eng = df_eng[df_eng['topic']!="-1"]
print(df_eng.shape)


# calculate popularity as the sum of avg_like, avg_reply, and avg_repost
df_eng['popularity'] = df_eng['avg_like'] + df_eng['avg_reply'] + df_eng['avg_repost']

# popularity of the nar, avg by topic (each topic has one narrative)
df_nar_popularity = df_eng.groupby('topic').agg({
    'popularity': 'mean',
    'mmr_30_nar': 'first'
}).reset_index()

# Rename columns for clarity
df_nar_popularity.columns = ['topic', 'avg_popularity', 'narrative']

print(f"Narrative popularity shape: {df_nar_popularity.shape}")
print("Top 5 most popular narratives:")
print(df_nar_popularity.nlargest(5, 'avg_popularity')[['topic', 'avg_popularity', 'narrative']])



# Calculate popularity by platform and topic
df_platform_popularity = df_eng.groupby(['platform', 'topic']).agg({
    'popularity': 'mean',
    'mmr_30_nar': 'first'
}).reset_index()

# Rename columns for clarity
df_platform_popularity.columns = ['platform', 'topic', 'avg_popularity', 'narrative']

# Get unique platforms
platforms = df_platform_popularity['platform'].unique()
print(f"Available platforms: {platforms}")

# Find platform-specific max for better visibility (instead of global max)
platform_max = {}
for platform in platforms:
    platform_data = df_platform_popularity[df_platform_popularity['platform'] == platform]
    platform_max[platform] = platform_data['avg_popularity'].max()
    print(f"{platform} max popularity: {platform_max[platform]:.1f}")

# Create plots for each platform
fig, axes = plt.subplots(len(platforms), 1, figsize=(20, 10*len(platforms)))
if len(platforms) == 1:
    axes = [axes]  # Make it a list for consistency

# Define Nature-style colors (similar to Nature journal color palette)
nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']

for i, platform in enumerate(platforms):
    # Filter data for this platform
    platform_data = df_platform_popularity[df_platform_popularity['platform'] == platform]
    
    # Get top 10 most popular narratives for this platform and sort by popularity (descending)
    top_10 = platform_data.nlargest(10, 'avg_popularity').sort_values('avg_popularity', ascending=True)
    
    # Create horizontal bar plot
    ax = axes[i]
    bars = ax.barh(range(len(top_10)), top_10['avg_popularity'], 
                   color=nature_colors[i % len(nature_colors)], alpha=0.8, 
                   edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels([f"Nar {topic}" for topic in top_10['topic']], fontsize=12)
    ax.set_xlabel('Average Popularity Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Top 10 Popular Conspiracy Narratives on {platform.title()}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Use platform-specific scaling for better visibility
    current_max = platform_max[platform]
    if platform.lower() == 'bluesky':
        ax.set_xlim(0, 3000)  # Fixed scale for Bluesky
    else:
        ax.set_xlim(0, current_max * 1.15)
    
    # Add value labels on bars
    for j, (bar, value) in enumerate(zip(bars, top_10['avg_popularity'])):
        ax.text(value + current_max * 0.02, j, f'{value:.1f}', 
                va='center', fontsize=11, fontweight='bold')
    
    # Create legend with full narrative text (bottom right corner)
    legend_text = []
    # Sort all narratives from highest to lowest popularity (reverse of y-axis which goes bottom to top)
    all_narratives_sorted = top_10.sort_values('avg_popularity', ascending=False)  # Highest to lowest
    
    for idx, row in all_narratives_sorted.iterrows():
        # Use full narrative text, wrap if too long
        narrative_full = row['narrative']
        if len(narrative_full) > 80:
            # Split long narratives into multiple lines
            words = narrative_full.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= 80:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            narrative_display = '\n    '.join(lines)
        else:
            narrative_display = narrative_full
            
        legend_text.append(f"Nar {row['topic']}: {narrative_display}")
    
    # Add legend box (bottom right corner)
    legend_box = '\n\n'.join(legend_text)
    ax.text(0.50, 0.05, legend_box, transform=ax.transAxes, 
            fontsize=13, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1.5),
            linespacing=1.2)
    
    # # Print details for this platform
    # print(f"\n=== {platform.upper()} PLATFORM ===")
    # print(f"Number of narratives: {len(platform_data)}")
    # print(f"Top 3 most popular narratives:")
    # for idx, row in top_10.tail(3).iterrows():  # tail(3) to get highest 3 after sorting
    #     print(f"  Nar {row['topic']}: {row['avg_popularity']:.1f} - {row['narrative'][:100]}...")

plt.tight_layout()
plt.show()

# Summary statistics by platform
print("\n=== PLATFORM SUMMARY ===")
platform_summary = df_platform_popularity.groupby('platform').agg({
    'avg_popularity': ['count', 'mean', 'std', 'max']
}).round(2)
platform_summary.columns = ['Num_Narratives', 'Mean_Popularity', 'Std_Popularity', 'Max_Popularity']
print(platform_summary)

# Save the plot
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300)
print(f"Plot saved to {OUTPUT_PATH}")