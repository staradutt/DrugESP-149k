#!/bin/bash

#
# ==========================
# Massive Parallel ORCA Pipeline
# ==========================
#
# Author: Taradutt Pattnaik
# Date: 2026
#
# Description:
# Launches parallel ORCA quantum chemistry calculations
# across chunked molecular datasets generated from the
# moses_pipeline.py
# Run splitter_script.py first
#
#
# Workflow:
# 1. Load chunked molecule JSON
# 2. Generate ORCA input files
# 3. Run ORCA calculations
# 4. Parse molecular properties
# 5. Save parsed JSONL outputs
#
# Important Notes:
# - One worker per dataset chunk (chunks created by splitter_script.py)
# - Scratch directories are created dynamically
# - Results are checkpointed automatically
# - Failed molecules are skipped and logged
#
# ==================================

#SBATCH --job-name=mep_chelpg
#SBATCH -N 1
#SBATCH -n 126
#SBATCH -p general
#SBATCH --time=14-00:00:00 
#SBATCH -o logs/mega_job.log

# Dynamic Base Directory Detection 
BASE_DIR=$(pwd)
PARTS_DIR="$BASE_DIR/mega_parts"
PARSED_DIR="$BASE_DIR/parsed_results"
LOG="$BASE_DIR/logs/mega_progress.log"
# ==============================
# ExterNal Exacutable and scripts used 
# ==============================
# Placeholder for ORCA executable - you will provide the path
ORCA_BIN="path/to/your/orca" 

PARSER="$BASE_DIR/parse_mol.py"
GEN_INPUT="$BASE_DIR/gen_input.py"

mkdir -p "$PARSED_DIR"
mkdir -p "$BASE_DIR/scratch"
mkdir -p "$BASE_DIR/logs"
touch "$LOG"

# Optional Conda/Environment activation. Check and installl modules needed by external scripts 
# source ~/miniconda3/bin/activate [your_env_name]


# Basic Run info 
echo "Python:   $(which python3)"
echo "Workers:  126"
echo "Started:  $(date)"

# Worker Function
run_worker() {
    local part_id=$1
    local local_json="$BASE_DIR/mega_parts/part_${part_id}.json"
    local worker_results="$BASE_DIR/parsed_results/worker_${part_id}.jsonl"
    local LOG="$BASE_DIR/logs/mega_progress.log"
    local ORCA_BIN="path/to/your/orca" 
    local PARSER="$BASE_DIR/parse_mol.py"
    local GEN_INPUT="$BASE_DIR/gen_input.py"
    local BASE_DIR=$(pwd)
    # Check chunk exists
    if [ ! -f "$local_json" ]; then
        echo "WORKER $part_id: chunk not found" >> "$LOG"
        return
    fi

    # Count molecules in chunk
    local num_mols=$(python3 -c "import json; print(len(json.load(open('$local_json'))))")
    echo "WORKER $part_id: starting $num_mols molecules" >> "$LOG"
    # Molecule loop
    for (( i=0; i<$num_mols; i++ )); do

        local mol_id=$(python3 -c "
import json
d = json.load(open('$local_json'))
print(d[$i]['mol_id'])
")

        # Skip if already parsed
        if grep -q "\"mol_id\": $mol_id," "$worker_results" 2>/dev/null; then
            echo "SKIP mol_$mol_id" >> "$LOG"
            continue
        fi

        # Create temporary scratch directory that will be deleted after calculation finish
        local scratch="$BASE_DIR/scratch/w${part_id}_m${mol_id}"
        rm -rf "$scratch"
        mkdir -p "$scratch"

        if [ ! -d "$scratch" ]; then
            echo "SCRATCH_FAIL mol_$mol_id worker_$part_id" >> "$LOG"
            continue
        fi

        # Generate input file into scratch using gen_input.py script
        python3 "$GEN_INPUT" "$local_json" "$i" "$scratch/input.inp"

        if [ ! -f "$scratch/input.inp" ]; then
            echo "INPUT_FAIL mol_$mol_id worker_$part_id" >> "$LOG"
            rm -rf "$scratch"
            continue
        fi

        # cd into scratch , ORCA must run from the same dir as input
        cd "$scratch"
        timeout 15m $ORCA_BIN input.inp > job.out 2>&1

        # Check result and parse using parse_mol.py
        if grep -q "ORCA TERMINATED NORMALLY" job.out; then
            python3 "$PARSER" "$mol_id" job.out "$worker_results"
            echo "SUCCESS mol_$mol_id worker_$part_id" >> "$LOG"
        
        else
            if grep -q "Killed" job.out; then
                echo "TIMEOUT mol_$mol_id worker_$part_id" >> "$LOG"
            else
                echo "FAIL mol_$mol_id worker_$part_id" >> "$LOG"
            fi
        fi

        # Clean scratch and return to base
        cd "$BASE_DIR"
        rm -rf "$scratch"

    done

    echo "WORKER $part_id: done" >> "$LOG"
}
# Export worker function
export BASE_DIR
export -f run_worker

# Launch 126 workers in parallel
for (( w=0; w<126; w++ )); do
    run_worker $w &
done
wait

# Final summary output to log file 

echo "" >> "$LOG"
echo "=== MEGA JOB COMPLETE $(date) ===" >> "$LOG"
success=$(grep -c "^SUCCESS" "$LOG" 2>/dev/null || echo 0)
timeout=$(grep -c "^TIMEOUT" "$LOG" 2>/dev/null || echo 0)
fail=$(grep -c    "^FAIL"    "$LOG" 2>/dev/null || echo 0)
total_parsed=$(cat "$BASE_DIR/parsed_results"/worker_*.jsonl 2>/dev/null | wc -l)

echo "Parsed molecules: $total_parsed" >> "$LOG"
echo "SUCCESS:          $success"      >> "$LOG"
echo "TIMEOUT:          $timeout"      >> "$LOG"
echo "FAIL:             $fail"         >> "$LOG"

cat "$LOG"
echo "Mega job complete"