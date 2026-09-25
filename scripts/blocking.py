import pandas as pd
import re

def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # strip punctuation
    replacements = {
        'corporation': 'corp', 'limited': 'ltd', 'private': 'pvt',
        'road': 'rd', 'street': 'st'
    }
    for full, short in replacements.items():
        text = text.replace(full, short)
    return text.strip()

train_s1['name_norm'] = train_s1['business_name'].apply(normalize)
train_s2['name_norm'] = train_s2['business_name'].apply(normalize)
train_s3['name_norm'] = train_s3['business_name'].apply(normalize)

# Simple blocking key: first 4 chars of normalized name + country
train_s1['block_key'] = train_s1['name_norm'].str[:4] + "_" + train_s1['country']
train_s2['block_key'] = train_s2['name_norm'].str[:4] + "_" + train_s2['country']
train_s3['block_key'] = train_s3['name_norm'].str[:4] + "_" + train_s3['country']

# Build candidate pairs per Source 1 entity (using a small sample first!)
sample_s1 = train_s1.head(1000)  # start small to test before running on all 2.2M

candidates = []
s2_grouped = train_s2.groupby('block_key')['entity_id'].apply(list).to_dict()
s3_grouped = train_s3.groupby('block_key')['entity_id'].apply(list).to_dict()

for _, row in sample_s1.iterrows():
    key = row['block_key']
    cands = s2_grouped.get(key, []) + s3_grouped.get(key, [])
    candidates.append({
        'source1_entity_id': row['entity_id'],
        'candidate_entity_ids': ",".join(cands)
    })

candidate_df = pd.DataFrame(candidates)
print(candidate_df.head(10))
print("Avg candidates per entity:", candidate_df['candidate_entity_ids'].apply(lambda x: len(x.split(",")) if x else 0).mean())