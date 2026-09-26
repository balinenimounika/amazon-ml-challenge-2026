import os
import sys
import gc
import re
import time
import argparse
import unicodedata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

PUNCTUATION_FILTER = re.compile(r'[^a-z0-9\s]')
WHITESPACE_COLLAPSER = re.compile(r'\s+')

def sanitize_profile_string(raw_val):
    if raw_val is None or pd.isna(raw_val):
        return ""
    str_val = str(raw_val)
    ascii_val = unicodedata.normalize('NFKD', str_val).encode('ascii', 'ignore').decode('utf-8').lower()
    cleaned_val = PUNCTUATION_FILTER.sub(' ', ascii_val)
    return WHITESPACE_COLLAPSER.sub(' ', cleaned_val).strip()

def assemble_entity_profiles(title_series, location_series):
    compiled_profiles = []
    for t_item, l_item in zip(title_series, location_series):
        sanitized_t = sanitize_profile_string(t_item)
        sanitized_l = sanitize_profile_string(l_item)
        if sanitized_t and sanitized_l:
            compiled_profiles.append(f"{sanitized_t} {sanitized_l}")
        else:
            compiled_profiles.append(sanitized_t or sanitized_l or "empty")
    return compiled_profiles

def ingest_catalog_table(file_location, record_cap=None):
    if not os.path.isfile(file_location):
        raise FileNotFoundError(f"Missing file: {file_location}")
    time_marker = time.time()
    tabular_frame = pd.read_csv(
        file_location,
        sep='\t',
        nrows=record_cap,
        dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str},
        keep_default_na=False
    )
    print(f"Ingested {len(tabular_frame):,} rows from {os.path.basename(file_location)} in {time.time() - time_marker:.2f}s")
    return tabular_frame

class SparseCandidateIndex:
    def __init__(self, token_bounds=(1, 2), floor_freq=2, ceiling_ratio=0.05, dimension_cap=150000, max_retained=20, min_affinity=0.10):
        self.token_bounds = token_bounds
        self.floor_freq = floor_freq
        self.ceiling_ratio = ceiling_ratio
        self.dimension_cap = dimension_cap
        self.max_retained = max_retained
        self.min_affinity = min_affinity
        self.sparse_vector_model = None
        self.inverted_target_csr = None
        self.indexed_identifiers = None

    def construct_index(self, target_narratives, target_unique_keys):
        t_start = time.time()
        self.indexed_identifiers = np.array(target_unique_keys, dtype=object)
        self.sparse_vector_model = TfidfVectorizer(
            analyzer='word',
            ngram_range=self.token_bounds,
            min_df=self.floor_freq,
            max_df=self.ceiling_ratio,
            max_features=self.dimension_cap,
            sublinear_tf=True,
            norm='l2',
            dtype=np.float32
        )
        fitted_matrix = self.sparse_vector_model.fit_transform(target_narratives)
        self.inverted_target_csr = fitted_matrix.T.tocsc()
        del fitted_matrix
        gc.collect()
        print(f"Built index: {len(self.indexed_identifiers):,} entries, {self.inverted_target_csr.shape[0]:,} dimensions in {time.time() - t_start:.2f}s")

    def query_index_batches(self, query_narratives, query_unique_keys, chunk_volume=1000):
        total_queries = len(query_narratives)
        cutoff = self.min_affinity
        quota = self.max_retained
        entity_registry = self.indexed_identifiers
        matrix_transposed = self.inverted_target_csr

        for slice_origin in range(0, total_queries, chunk_volume):
            slice_terminus = min(slice_origin + chunk_volume, total_queries)
            sub_texts = query_narratives[slice_origin:slice_terminus]
            sub_keys = query_unique_keys[slice_origin:slice_terminus]

            transformed_query_batch = self.sparse_vector_model.transform(sub_texts)
            affinity_matrix = transformed_query_batch.dot(matrix_transposed).tocsr()

            for row_pointer, current_key in enumerate(sub_keys):
                idx_head = affinity_matrix.indptr[row_pointer]
                idx_tail = affinity_matrix.indptr[row_pointer + 1]

                if idx_tail == idx_head:
                    yield current_key, []
                    continue

                cell_values = affinity_matrix.data[idx_head:idx_tail]
                cell_columns = affinity_matrix.indices[idx_head:idx_tail]

                if cutoff > 0.0:
                    valid_mask = cell_values >= cutoff
                    cell_values = cell_values[valid_mask]
                    cell_columns = cell_columns[valid_mask]

                if len(cell_values) == 0:
                    yield current_key, []
                    continue

                if len(cell_values) > quota:
                    partitioned_sub = np.argpartition(-cell_values, quota)[:quota]
                    sorted_order = partitioned_sub[np.argsort(-cell_values[partitioned_sub])]
                    designated_indices = cell_columns[sorted_order]
                else:
                    designated_indices = cell_columns[np.argsort(-cell_values)]

                unique_accumulator = []
                encountered = set()
                for target_pos in designated_indices:
                    chosen_id = entity_registry[target_pos]
                    if chosen_id not in encountered:
                        encountered.add(chosen_id)
                        unique_accumulator.append(chosen_id)

                yield current_key, unique_accumulator

def execute_entity_blocking(
    input_directory="student_resource/dataset/test",
    output_directory="output",
    publish_root=True,
    processing_limit=None,
    retention_bound=20,
    similarity_gate=0.10
):
    print("Execution start: Entity resolution blocking")
    initiation_timestamp = time.time()
    os.makedirs(output_directory, exist_ok=True)

    split_identifier = "test" if "test" in input_directory else "train"
    path_primary = os.path.join(input_directory, f"{split_identifier}_source1.tsv")
    path_secondary = os.path.join(input_directory, f"{split_identifier}_source2.tsv")
    path_tertiary = os.path.join(input_directory, f"{split_identifier}_source3.tsv")

    primary_table = ingest_catalog_table(path_primary, record_cap=processing_limit)
    secondary_table = ingest_catalog_table(path_secondary, record_cap=processing_limit * 3 if processing_limit else None)
    tertiary_table = ingest_catalog_table(path_tertiary, record_cap=processing_limit * 3 if processing_limit else None)

    official_destination = os.path.join(output_directory, "candidate_pairs.tsv")
    root_destination = "candidate_pairs.tsv" if publish_root else None
    pairwise_destination = "candidate_pairs_pairwise.tsv" if publish_root else None

    writer_official = open(official_destination, 'w', encoding='utf-8', buffering=1024*1024)
    writer_official.write("source1_entity_id\tcandidate_entity_ids\n")

    writer_root = open(root_destination, 'w', encoding='utf-8', buffering=1024*1024) if root_destination else None
    if writer_root:
        writer_root.write("source1_entity_id\tcandidate_entity_ids\n")

    writer_pairwise = open(pairwise_destination, 'w', encoding='utf-8', buffering=1024*1024) if pairwise_destination else None
    if writer_pairwise:
        writer_pairwise.write("source1_id\tsource2_id\n")

    distinct_territories = list(primary_table['country'].unique())
    print(f"Territories to process: {distinct_territories}")

    global_cartesian_extent = len(primary_table) * (len(secondary_table) + len(tertiary_table))
    cumulative_retained_pairs = 0
    cumulative_anchors = 0
    per_entity_tallies = []

    for terr_num, territory in enumerate(distinct_territories, 1):
        print(f"Territory {terr_num}/{len(distinct_territories)}: {territory}")
        sub_primary = primary_table[primary_table['country'] == territory]
        sub_secondary = secondary_table[secondary_table['country'] == territory]
        sub_tertiary = tertiary_table[tertiary_table['country'] == territory]

        anchor_count = len(sub_primary)
        pool_count = len(sub_secondary) + len(sub_tertiary)
        print(f"Anchor records: {anchor_count:,} | Pool records: {pool_count:,}")

        if anchor_count == 0:
            continue

        if pool_count == 0:
            for blank_id in sub_primary['entity_id']:
                writer_official.write(f"{blank_id}\t\n")
                if writer_root:
                    writer_root.write(f"{blank_id}\t\n")
                per_entity_tallies.append(0)
            cumulative_anchors += anchor_count
            continue

        unified_target_ids = list(sub_secondary['entity_id']) + list(sub_tertiary['entity_id'])
        unified_target_titles = list(sub_secondary['business_name']) + list(sub_tertiary['business_name'])
        unified_target_locs = list(sub_secondary['business_address']) + list(sub_tertiary['business_address'])

        t_prep = time.time()
        compiled_pool_texts = assemble_entity_profiles(unified_target_titles, unified_target_locs)
        del unified_target_titles, unified_target_locs
        gc.collect()
        print(f"Profile assembly: {time.time() - t_prep:.2f}s")

        retrieval_engine = SparseCandidateIndex(
            token_bounds=(1, 2),
            floor_freq=2,
            ceiling_ratio=0.05,
            dimension_cap=150000,
            max_retained=retention_bound,
            min_affinity=similarity_gate
        )
        retrieval_engine.construct_index(compiled_pool_texts, unified_target_ids)
        del compiled_pool_texts, unified_target_ids
        gc.collect()

        anchor_ids = list(sub_primary['entity_id'])
        compiled_anchor_texts = assemble_entity_profiles(list(sub_primary['business_name']), list(sub_primary['business_address']))

        t_search = time.time()
        partition_retained = 0
        counter_anchor = 0

        for a_key, candidate_list in retrieval_engine.query_index_batches(compiled_anchor_texts, anchor_ids, chunk_volume=1000):
            formatted_targets = ",".join(candidate_list)
            writer_official.write(f"{a_key}\t{formatted_targets}\n")
            if writer_root:
                writer_root.write(f"{a_key}\t{formatted_targets}\n")

            if writer_pairwise:
                for target_entry in candidate_list:
                    writer_pairwise.write(f"{a_key}\t{target_entry}\n")

            curr_len = len(candidate_list)
            partition_retained += curr_len
            per_entity_tallies.append(curr_len)
            counter_anchor += 1

            if counter_anchor % 100000 == 0 or counter_anchor == anchor_count:
                cur_rate = counter_anchor / max(time.time() - t_search, 0.001)
                print(f"Anchor progress: {counter_anchor:,}/{anchor_count:,} ({counter_anchor/anchor_count*100:.1f}%) | {cur_rate:,.1f} QPS")

        qps_final = anchor_count / max(time.time() - t_search, 0.001)
        print(f"Territory {territory} done: {anchor_count:,} queries in {time.time() - t_search:.2f}s ({qps_final:,.1f} QPS)")
        print(f"Retained {partition_retained:,} pairs (Mean: {partition_retained / anchor_count:.2f} per anchor)")

        cumulative_retained_pairs += partition_retained
        cumulative_anchors += anchor_count

        del retrieval_engine, compiled_anchor_texts, anchor_ids, sub_primary, sub_secondary, sub_tertiary
        gc.collect()

    writer_official.flush()
    writer_official.close()
    if writer_root:
        writer_root.flush()
        writer_root.close()
    if writer_pairwise:
        writer_pairwise.flush()
        writer_pairwise.close()

    total_duration = time.time() - initiation_timestamp
    filtering_efficiency = 1.0 - (cumulative_retained_pairs / max(global_cartesian_extent, 1))

    print("=" * 70)
    print("BLOCKING SUMMARY METRICS")
    print("=" * 70)
    print(f"Anchors processed        : {cumulative_anchors:,}")
    print(f"Full Cartesian product   : {global_cartesian_extent:,}")
    print(f"Candidate pairs retained : {cumulative_retained_pairs:,}")
    print(f"Search space reduction   : {filtering_efficiency * 100:.6f}%")
    print(f"Compression ratio        : {global_cartesian_extent / max(cumulative_retained_pairs, 1):,.1f}x reduction")
    if per_entity_tallies:
        print(f"Average candidates/anchor: {np.mean(per_entity_tallies):.2f}")
        print(f"Median candidates/anchor : {np.median(per_entity_tallies):.1f}")
        print(f"Zero-candidate anchors   : {sum(1 for val in per_entity_tallies if val == 0):,} ({sum(1 for val in per_entity_tallies if val == 0)/len(per_entity_tallies)*100:.2f}%)")
    print(f"Runtime duration         : {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"Output files generated   :")
    print(f"  - {root_destination}")
    print(f"  - {official_destination}")
    if pairwise_destination:
        print(f"  - {pairwise_destination}")
    print("=" * 70)

    return {
        "anchors": cumulative_anchors,
        "cartesian": global_cartesian_extent,
        "retained": cumulative_retained_pairs,
        "efficiency": filtering_efficiency,
        "duration": total_duration
    }

def benchmark_ground_truth_recall(benchmark_volume=1000, retention_bound=20, similarity_gate=0.10):
    reference_path = "student_resource/dataset/train/train_ground_truth.tsv"
    if not os.path.isfile(reference_path):
        print(f"Ground truth absent at {reference_path}")
        return

    reference_table = pd.read_csv(reference_path, sep='\t', nrows=benchmark_volume * 5)
    matched_subset = reference_table[reference_table['matched_entity_ids'].notna() & (reference_table['matched_entity_ids'] != '')].head(benchmark_volume)
    reference_lookup = {
        row['source1_entity_id']: set(row['matched_entity_ids'].split(','))
        for _, row in matched_subset.iterrows()
    }
    designated_anchors = set(reference_lookup.keys())
    verified_targets = set()
    for entry_set in reference_lookup.values():
        verified_targets.update(entry_set)

    anchor_entries = []
    with open("student_resource/dataset/train/train_source1.tsv", encoding='utf-8') as stream:
        stream.readline()
        for rec in stream:
            tokens = rec.rstrip('\n').split('\t')
            if tokens[0] in designated_anchors:
                anchor_entries.append(tokens)
                if len(anchor_entries) == len(designated_anchors):
                    break

    df_anchors = pd.DataFrame(anchor_entries, columns=['entity_id', 'business_name', 'business_address', 'country'])

    pool_entries = []
    for source_file in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        with open(source_file, encoding='utf-8') as stream:
            stream.readline()
            for rec in stream:
                tokens = rec.rstrip('\n').split('\t')
                if len(tokens) >= 4:
                    if tokens[0] in verified_targets:
                        pool_entries.append(tokens)
                    elif len(pool_entries) < 50000:
                        pool_entries.append(tokens)

    df_pool = pd.DataFrame(pool_entries, columns=['entity_id', 'business_name', 'business_address', 'country'])
    print(f"Evaluating {len(df_anchors):,} anchor queries against {len(df_pool):,} candidate entities")

    overall_true_links = sum(len(items) for items in reference_lookup.values())
    recovered_true_links = 0
    tally_proposals = 0

    for reg in df_anchors['country'].unique():
        sub_a = df_anchors[df_anchors['country'] == reg]
        sub_p = df_pool[df_pool['country'] == reg]
        if len(sub_p) == 0:
            continue

        pool_narratives = assemble_entity_profiles(list(sub_p['business_name']), list(sub_p['business_address']))
        anchor_narratives = assemble_entity_profiles(list(sub_a['business_name']), list(sub_a['business_address']))

        test_engine = SparseCandidateIndex(
            token_bounds=(1, 2),
            floor_freq=2,
            ceiling_ratio=0.05,
            dimension_cap=150000,
            max_retained=retention_bound,
            min_affinity=similarity_gate
        )
        test_engine.construct_index(pool_narratives, list(sub_p['entity_id']))

        for a_id, proposals in test_engine.query_index_batches(anchor_narratives, list(sub_a['entity_id'])):
            set_proposals = set(proposals)
            set_expected = reference_lookup.get(a_id, set())
            recovered_true_links += len(set_proposals & set_expected)
            tally_proposals += len(set_proposals)

    observed_recall = recovered_true_links / overall_true_links
    print("=" * 60)
    print("EMPIRICAL RECALL EVALUATION")
    print("=" * 60)
    print(f"Ground truth target links: {overall_true_links:,}")
    print(f"Successfully retrieved   : {recovered_true_links:,}")
    print(f"Empirical match recall   : {observed_recall * 100:.2f}%")
    print(f"Average candidates/anchor: {tally_proposals / len(df_anchors):.1f}")
    print("=" * 60)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Entity Resolution Blocking")
    cli_parser.add_argument("--data-dir", default="student_resource/dataset/test", help="Path to input datasets")
    cli_parser.add_argument("--output-dir", default="output", help="Path to output folder")
    cli_parser.add_argument("--max-records", type=int, default=None, help="Record cap for test runs")
    cli_parser.add_argument("--top-k", type=int, default=20, help="Max candidates per anchor")
    cli_parser.add_argument("--min-sim", type=float, default=0.10, help="Cosine threshold")
    cli_parser.add_argument("--benchmark", action="store_true", help="Run recall benchmark")
    cli_args = cli_parser.parse_args()

    if cli_args.benchmark:
        benchmark_ground_truth_recall(benchmark_volume=1000, retention_bound=cli_args.top_k, similarity_gate=cli_args.min_sim)
    else:
        execute_entity_blocking(
            input_directory=cli_args.data_dir,
            output_directory=cli_args.output_dir,
            publish_root=True,
            processing_limit=cli_args.max_records,
            retention_bound=cli_args.top_k,
            similarity_gate=cli_args.min_sim
        )
