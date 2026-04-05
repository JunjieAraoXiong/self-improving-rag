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

## Project Structure

```
├── src/
│   ├── agents/              # Multi-agent system (Algorithm 1)
│   │   ├── orchestrator.py  # Main retry loop
│   │   ├── retrieval_agent.py
│   │   ├── reasoning_agent.py
│   │   └── judge_agent.py
│   ├── retrieval_tools/     # Retrieval pipelines
│   │   ├── semantic.py      # Dense vector search
│   │   ├── hybrid.py        # BM25 + semantic ensemble
│   │   ├── rerank.py        # Cross-encoder reranking
│   │   ├── hyde.py          # Hypothetical Document Embeddings
│   │   └── router.py        # Rule-based routing
│   ├── providers/           # LLM adapters (OpenAI, Anthropic)
│   ├── config.py            # Central configuration
│   └── bulk_testing.py      # Evaluation entry point
├── evaluation/              # Metrics & LLM-as-Judge
├── dataset_adapters/        # FinanceBench loader
└── scripts/                 # Experiment scripts
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
python src/bulk_testing.py --dataset financebench --pipeline routed --model gpt-4o-mini --use-llm-judge --use-agentic-retry --max-retries 2

# Table 3: Component ablation (deployment mode)
python src/bulk_testing.py --dataset financebench --pipeline routed --model gpt-4o-mini --use-agentic-retry --max-retries 2
python src/bulk_testing.py --dataset financebench --pipeline routed --model gpt-4o-mini --use-agentic-retry --max-retries 1
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
