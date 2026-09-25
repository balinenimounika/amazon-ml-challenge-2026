import os
import random
from collections import defaultdict

def analyze_countries(file_path):
    country_counts = defaultdict(int)
    total_rows = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                country = parts[3].strip()
                country_counts[country] += 1
                total_rows += 1
                
    return country_counts, total_rows

def print_distribution(name, counts, total):
    print(f"\n--- {name} Country Distribution ---")
    if total == 0:
        print("No rows found!")
        return
        
    for country, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        print(f"{country.ljust(10)} : {count:9d} ({pct:5.2f}%)")

def sample_france(file_path, num_samples=10, seed=42):
    print(f"\n--- Sampling {num_samples} France entries from {os.path.basename(file_path)} ---")
    france_entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                country = parts[3].strip()
                if country.lower() == 'france':
                    name = parts[1].strip()
                    address = parts[2].strip()
                    france_entries.append((name, address))
                    
    if not france_entries:
        print("No France entries found.")
        return
        
    random.seed(seed)
    sample_size = min(num_samples, len(france_entries))
    sample = random.sample(france_entries, sample_size)
    
    max_name = max(len(s[0]) for s in sample)
    max_name = max(max_name, 13) # "Business Name"
    
    print(f"{'Business Name'.ljust(max_name)} | Business Address")
    print("-" * max_name + "-+-" + "-" * 40)
    for name, address in sample:
        print(f"{name.ljust(max_name)} | {address}")

def main():
    test_dir = os.path.join('dataset', 'test')
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')
    
    if not os.path.exists(s1_path):
        print(f"Error: {s1_path} not found.")
        return
        
    print("Scanning test files (this may take a moment)...")
    
    s1_counts, s1_total = analyze_countries(s1_path)
    s2_counts, s2_total = analyze_countries(s2_path)
    s3_counts, s3_total = analyze_countries(s3_path)
    
    print_distribution("test_source1.tsv", s1_counts, s1_total)
    print_distribution("test_source2.tsv", s2_counts, s2_total)
    print_distribution("test_source3.tsv", s3_counts, s3_total)
    
    sample_france(s1_path)

if __name__ == "__main__":
    main()
