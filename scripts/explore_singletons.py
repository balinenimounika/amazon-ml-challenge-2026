import os
import random
import csv

def load_singletons(val_gt_path):
    singletons = []
    with open(val_gt_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0]
            if len(parts) < 2:
                singletons.append(s1_id)
            else:
                matched_val = parts[1].strip()
                if matched_val == '' or matched_val.lower() == 'nan':
                    singletons.append(s1_id)
    return singletons

def print_table(results):
    if not results:
        return
        
    # Find max width for each column
    max_id = max(len(r['id']) for r in results)
    max_name = max(len(r['name']) for r in results)
    max_addr = max(len(r['address']) for r in results)
    max_country = max(len(r['country']) for r in results)
    
    max_id = max(max_id, 9) # "Entity ID"
    max_name = max(max_name, 13) # "Business Name"
    max_addr = max(max_addr, 16) # "Business Address"
    max_country = max(max_country, 7) # "Country"
    
    header = f"{'Entity ID'.ljust(max_id)} | {'Business Name'.ljust(max_name)} | {'Business Address'.ljust(max_addr)} | {'Country'.ljust(max_country)}"
    sep = f"{'-' * max_id}-+-{'-' * max_name}-+-{'-' * max_addr}-+-{'-' * max_country}"
    
    print(header)
    print(sep)
    for r in results:
        print(f"{r['id'].ljust(max_id)} | {r['name'].ljust(max_name)} | {r['address'].ljust(max_addr)} | {r['country'].ljust(max_country)}")

def main():
    val_gt_path = os.path.join('dataset', 'train', 'dev_val_ground_truth.tsv')
    source1_path = os.path.join('dataset', 'train', 'train_source1.tsv')
    
    print(f"Identifying singletons in {val_gt_path}...")
    singletons = load_singletons(val_gt_path)
    
    if len(singletons) < 20:
        print(f"Warning: Only found {len(singletons)} singletons total. Using all of them.")
        sample_size = len(singletons)
    else:
        sample_size = 20
        
    random.seed(42)
    sample_ids = set(random.sample(singletons, sample_size))
    
    print(f"Looking up {sample_size} sampled entities in {source1_path}...\n")
    
    results_map = {}
    
    # Read source1. Format: entity_id, business_name, business_address, country
    # We will use simple string splitting to be robust
    with open(source1_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
                
            s1_id = parts[0]
            if s1_id in sample_ids:
                name = parts[1] if len(parts) > 1 else ""
                address = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""
                
                results_map[s1_id] = {
                    'id': s1_id,
                    'name': name,
                    'address': address,
                    'country': country
                }
                
                # Fast exit if we found all of them
                if len(results_map) == sample_size:
                    break
                    
    # Check if any were not found
    missing = sample_ids - set(results_map.keys())
    if missing:
        print(f"WARNING: The following {len(missing)} S1 entities from ground truth were not found in train_source1.tsv: {missing}")
        
    results = list(results_map.values())
    print_table(results)

if __name__ == "__main__":
    main()
