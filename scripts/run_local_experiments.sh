#!/bin/bash
# =============================================================================
# ICLR Experiments: Local Runner for 16GB Mac
# =============================================================================
#
# Runs all experiments sequentially to avoid memory issues.
# Total experiments: 36 (Phase 1-3)
# Estimated time: 2-3 hours
#
# Usage:
#   ./scripts/run_local_experiments.sh [phase]
#
#   phase: 1, 2, 3, or "all" (default: all)
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

# Create output directories
mkdir -p bulk_runs logs agent_logs

# Configuration
SEEDS=(42 43 44)
DATASET="financebench"
MODEL="gpt-4o-mini"
TOP_K=10
CHROMA_PATH="chroma_docling"

# Function to run a single experiment
run_experiment() {
    local pipeline="$1"
    local seed="$2"
    local extra_args="$3"
    local name="$4"

    echo ""
    echo "========================================"
    echo "Running: $name (seed=$seed)"
    echo "========================================"

    python src/bulk_testing.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --pipeline "$pipeline" \
        --top-k "$TOP_K" \
        --chroma-path "$CHROMA_PATH" \
        --seed "$seed" \
        --num-runs 1 \
        --use-llm-judge \
        --judge-model gpt-4o-mini \
        $extra_args
}

# =============================================================================
# Phase 1: Baseline Experiments (12 runs)
# =============================================================================
run_phase1() {
    echo ""
    echo "============================================"
    echo "PHASE 1: Baseline Experiments"
    echo "============================================"

    for seed in "${SEEDS[@]}"; do
        run_experiment "semantic" "$seed" "" "Semantic Baseline"
        run_experiment "hybrid" "$seed" "" "Hybrid Baseline"
        run_experiment "hybrid_filter" "$seed" "" "Hybrid+Filter"
        run_experiment "hybrid_filter_rerank" "$seed" "" "Hybrid+Filter+Rerank"
    done

    echo ""
    echo "Phase 1 complete!"
}

# =============================================================================
# Phase 2: SC-RAG Experiments (6 runs)
# =============================================================================
run_phase2() {
    echo ""
    echo "============================================"
    echo "PHASE 2: SC-RAG Experiments"
    echo "============================================"

    for seed in "${SEEDS[@]}"; do
        run_experiment "hybrid_filter_rerank" "$seed" \
            "--use-agentic-retry --max-retries 1" \
            "SC-RAG (B=1)"
        run_experiment "hybrid_filter_rerank" "$seed" \
            "--use-agentic-retry --max-retries 2" \
            "SC-RAG (B=2)"
    done

    echo ""
    echo "Phase 2 complete!"
}

# =============================================================================
# Phase 3: Ablation Studies (12 runs)
# =============================================================================
run_phase3() {
    echo ""
    echo "============================================"
    echo "PHASE 3: Ablation Studies"
    echo "============================================"

    ABLATIONS=(
        "no_retrieval_escalation"
        "no_prompt_escalation"
        "no_hyde"
        "no_deterministic_verify"
    )

    for seed in "${SEEDS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            run_experiment "hybrid_filter_rerank" "$seed" \
                "--use-agentic-retry --max-retries 2 --ablation $ablation" \
                "SC-RAG − $ablation"
        done
    done

    echo ""
    echo "Phase 3 complete!"
}

# =============================================================================
# Aggregation
# =============================================================================
run_aggregation() {
    echo ""
    echo "============================================"
    echo "AGGREGATING RESULTS"
    echo "============================================"

    python scripts/aggregate_iclr_results.py --output-dir bulk_runs/

    echo ""
    echo "Aggregation complete! Check bulk_runs/ for results."
}

# =============================================================================
# Main
# =============================================================================
PHASE="${1:-all}"

echo "============================================"
echo "ICLR SC-RAG Experiments"
echo "============================================"
echo "Date: $(date)"
echo "Phase: $PHASE"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Seeds: ${SEEDS[*]}"
echo "============================================"

case "$PHASE" in
    1)
        run_phase1
        ;;
    2)
        run_phase2
        ;;
    3)
        run_phase3
        ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_aggregation
        ;;
    agg|aggregate)
        run_aggregation
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [1|2|3|all|agg]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================"
echo "End time: $(date)"
