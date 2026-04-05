# Towards Expert Financial QA via Self-Improving RAG

**Accepted at the AFA Workshop @ ICLR 2026**

Junjie Xiong (UC Berkeley), Shawheen Ghezavat (Cal Poly), Aum Hirpara (Hofstra University), Sean Wu (Pepperdine University)

[[Paper]](paper.pdf)

---

Expert-level financial QA requires both **grounded verification** to catch numeric hallucinations and **audit trails** for regulatory compliance -- attributes that standard single-pass RAG systems lack. Self-Improving RAG decomposes document QA into three specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator with feedback-driven self-correction. When the Judge scores an answer below a dynamic threshold, the system triggers retry with escalated strategies: broader retrieval, more careful prompting, and relaxed acceptance criteria.

## Key Results (FinanceBench)

- **86% oracle-guided accuracy** (+62.3% over single-pass RAG baseline)
- **36.4% Lazarus Rate** -- recovers nearly 4 in 10 initially incorrect answers through targeted retry
- A fixed retrieval pipeline with judge-driven retry achieves strong results **without dynamic routing**, providing full interpretability
- Every decision is logged with confidence scores, enabling **audit trails** for regulated financial applications

## Key Features

- **Three Specialized Agents**: Retrieval, Reasoning, and Judge agents with per-attempt escalation
- **Self-Correction Loop**: Judge-driven retry with dynamic threshold decay ($\tau_0{=}0.5$, $\lambda{=}0.1$, $\tau_{\min}{=}0.3$)
- **Grounded Verification**: Entailment checking against retrieved evidence with programmatic numeric verification
- **Audit-First Design**: Every agent decision logged with provenance, confidence scores, and reasoning traces
- **Walled Garden Constraint**: Retrieval stays within authorized corpora -- no web search fallback, ensuring compliance

## Architecture

```
                    ┌──────────────┐
                    │ Orchestrator │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌───────────┐ ┌──────────┐ ┌─────────┐
        │ Retrieval │ │Reasoning │ │  Judge  │
        │   Agent   │ │  Agent   │ │  Agent  │
        └───────────┘ └──────────┘ └─────────┘
```

The Orchestrator runs a retry loop (budget $B{=}2$, up to 3 attempts):

1. **Retrieval Agent** retrieves evidence with escalation: $k{=}10 \to 20 \to 30$, RSE on final attempt
2. **Reasoning Agent** generates answers with escalating prompts: Standard → Conservative → Detailed
3. **Judge Agent** scores on three dimensions: grounding ($\mu_g$), completeness ($\mu_c$), numeric faithfulness ($\mu_n$)
4. If $U_t = w_g\mu_g + w_c\mu_c + w_n\mu_n < \tau_t$, retry with escalation; otherwise accept

Best answer is always kept -- retry never degrades output quality.

## rLLM-FinQA Integration

We integrate [rLLM-FinQA](https://rllm-project.com/post.html?post=finqa.md) (Roongta, Tan et al., UC Berkeley / Snorkel AI) for RL-trained financial table reasoning. This adds a **TableAgent** with SQL + calculator tools that handles the 66% of FinanceBench questions requiring numeric computation.

| Component | Source | What it does |
|-----------|--------|-------------|
| Document retrieval + self-correction | Self-Improving RAG | Finds documents, verifies answers, retries on failure |
| Table reasoning + tool use | rLLM-FinQA | SQL queries, calculator, schema discovery over SEC tables |
| Combined pipeline | **New** | Retrieve documents, triage (table vs narrative), compute, verify |

### Setup rLLM-FinQA

```bash
git clone https://github.com/rllm-org/rllm.git   # in repo root
pip install -e rllm && pip install asteval
python -m projects.finqa.prepare_finqa_data        # downloads 207 companies, 6,900 tables
```

### Training with rLLM (Optional)

The existing [rLLM-FinQA-4B model](https://huggingface.co/rLLM/rLLM-FinQA-4B) can be used directly via vLLM. If you want to train on expanded data:

| Option | What | Cost | When |
|--------|------|------|------|
| **A. Use pre-trained model** | Serve rLLM-FinQA-4B via vLLM, use as-is | $0 | Start here |
| **B. Expand table data** | Run rLLM's synthetic data pipeline on FinanceBench companies | ~$50 | If company coverage is insufficient |
| **C. Fine-tune new model** | Use rLLM's GRPO framework on expanded data (8xH100, 21hrs) | ~$500 | For best accuracy / paper contribution |

```bash
# Option A: Inference only (requires GPU)
python -m vllm.entrypoints.openai.api_server \
    --model rLLM/rLLM-FinQA-4B --port 30000 --dtype bfloat16

# Option C: Training (requires 8xH100)
cd rllm && bash projects/finqa/train_finqa.sh
```

## Project Structure

```
├── src/
│   ├── agents/              # Multi-agent system (Algorithm 1)
│   │   ├── orchestrator.py  # Main retry loop + document triage
│   │   ├── retrieval_agent.py
│   │   ├── reasoning_agent.py
│   │   ├── table_agent.py   # Bridge to rLLM-FinQA tools
│   │   └── judge_agent.py   # Blind numeric verification
│   ├── retrieval_tools/     # Retrieval pipelines
│   │   ├── semantic.py      # Dense vector search
│   │   ├── hybrid.py        # BM25 + semantic ensemble
│   │   ├── rerank.py        # Cross-encoder reranking
│   │   ├── hyde.py          # Hypothetical Document Embeddings
│   │   └── router.py        # Rule-based routing
│   ├── providers/           # LLM adapters (OpenAI, Anthropic, Google)
│   │   └── cache.py         # SQLite LLM response cache
│   ├── config.py            # Configuration + cost tracking
│   └── bulk_testing.py      # Evaluation entry point
├── evaluation/              # Metrics & LLM-as-Judge
├── dataset_adapters/        # FinanceBench, FinQA loaders
├── scripts/
│   └── eval_finqa.py        # FinQA benchmark evaluation
└── rllm/                    # rLLM submodule (clone separately)
    └── projects/finqa/      # 4 tools, agent, environment, training
```

## Quick Start

### Installation
```bash
git clone https://github.com/JunjieAraoXiong/self-improving-rag.git
cd self-improving-rag
pip install -r requirements.txt
cp .env.example .env  # Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
```

### Run Single-Pass Baseline
```bash
python src/bulk_testing.py \
    --dataset financebench \
    --pipeline hybrid_filter_rerank \
    --model gpt-4o-mini \
    --use-llm-judge
```

### Run Self-Improving RAG (Agentic Mode)
```bash
python src/bulk_testing.py \
    --dataset financebench \
    --pipeline routed \
    --model gpt-4o-mini \
    --use-llm-judge \
    --use-agentic-retry \
    --max-retries 2
```

### Run with Table Agent (rLLM-FinQA Integration)
```bash
# With GPT-4o-mini driving rLLM tools (no GPU needed)
python src/bulk_testing.py \
    --dataset financebench \
    --pipeline hybrid_filter_rerank \
    --model gpt-4o-mini \
    --use-llm-judge \
    --use-agentic-retry \
    --use-table-agent

# With rLLM-FinQA-4B model (requires vLLM server on GPU)
python src/bulk_testing.py \
    --dataset financebench \
    --use-agentic-retry \
    --use-table-agent \
    --vllm-base-url http://localhost:30000/v1
```

### Run FinQA Benchmark
```bash
# Quick test (50 questions)
python scripts/eval_finqa.py --model gpt-4o-mini --n 50

# Full benchmark (558 questions)
python scripts/eval_finqa.py --model gpt-4o-mini --n 558

# With rLLM-FinQA-4B model
python scripts/eval_finqa.py --vllm-base-url http://localhost:30000/v1 --n 558
```

### Cross-Model Judging
```bash
# Generate with GPT-4o-mini, judge with GPT-4o
python src/bulk_testing.py \
    --dataset financebench \
    --model gpt-4o-mini \
    --judge-model gpt-4o \
    --use-llm-judge \
    --use-agentic-retry
```

## Reproducing Results

### Step 1: Prepare ChromaDB
```bash
# Build from scratch (requires SEC PDFs)
python src/ingest_docling.py --input-dir data/pdfs --output-dir chroma_docling
```

### Step 2: Run Experiments
```bash
# Table 1: Single-pass vs Self-Improving RAG (oracle-guided)
python src/bulk_testing.py --dataset financebench --pipeline hybrid_filter_rerank --model gpt-4o-mini --use-llm-judge
python src/bulk_testing.py --dataset financebench --pipeline hybrid_filter_rerank --model gpt-4o-mini --use-llm-judge --use-agentic-retry --max-retries 2

# Table 2: With Table Agent (rLLM-FinQA integration)
python src/bulk_testing.py --dataset financebench --pipeline hybrid_filter_rerank --model gpt-4o-mini --use-llm-judge --use-agentic-retry --use-table-agent

# Table 3: Component ablation (deployment mode)
python src/bulk_testing.py --dataset financebench --model gpt-4o-mini --use-agentic-retry --blind-judge --max-retries 2
python src/bulk_testing.py --dataset financebench --model gpt-4o-mini --use-agentic-retry --blind-judge --max-retries 1

# Table 4: FinQA benchmark
python scripts/eval_finqa.py --model gpt-4o-mini --n 558
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xiong2026selfimproving,
  title={Towards Expert Financial QA via Self-Improving RAG},
  author={Xiong, Junjie and Ghezavat, Shawheen and Hirpara, Aum and Wu, Sean},
  booktitle={AFA Workshop at the International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://github.com/JunjieAraoXiong/self-improving-rag}
}
```

## License

MIT License
