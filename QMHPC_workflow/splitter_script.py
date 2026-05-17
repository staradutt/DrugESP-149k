"""
Dataset Splitter for Parallel ORCA processing in mega_job.sh
Author: Taradutt Pattnaik
Date: 2026

Description:
Splits the master MOSES 3D dataset created by moses_pipeline into 
smaller JSON chunks for parallel quantum chemistry calculations

Input:
- MOSES_150k_Master.json (output of moses_pipeline)

Output:
- mega_parts/part_0.json
- mega_parts/part_1.json
- ... so on upto 
- mega_parts/part_125.json

Notes:
- Chunking is performed using near-equal partition sizes
- Relative paths are used 
"""
import json, os

# Relative paths for portability
JSON_PATH   = 'MOSES_150k_Master.json'
PARTS_DIR   = 'mega_parts'

NUM_WORKERS = 126

os.makedirs(PARTS_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    all_data = json.load(f)

chunk_size = (len(all_data) + NUM_WORKERS - 1) // NUM_WORKERS

for i in range(NUM_WORKERS):
    chunk = all_data[i * chunk_size : (i + 1) * chunk_size]
    if chunk:
        with open(f'{PARTS_DIR}/part_{i}.json', 'w') as f:
            json.dump(chunk, f)

print(f"Split {len(all_data):,} molecules into {NUM_WORKERS} chunks")