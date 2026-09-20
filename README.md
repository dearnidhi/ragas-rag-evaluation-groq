# RAGAS: Evaluate a RAG Pipeline with Groq

Learn how to check if a RAG (search + AI answer) system is actually good, using **RAGAS**.
Everything runs for free: **Groq** as the judge model and **HuggingFace** for embeddings. No OpenAI key needed.

<!-- Video walkthrough: add your link here -->

## What you will learn

A RAG system can fail in two places: the search and the answer. RAGAS gives each one a score.

| Metric | Question it answers |
|---|---|
| **Context Recall** | Did search find the information we needed? |
| **Context Precision** | Was the information we found actually useful? |
| **Faithfulness** | Did the AI stick to the documents, or make things up? |
| **Response Relevancy** | Did the AI answer the question that was asked? |
| **Factual Correctness** | Is the answer actually true? |

The notebook also covers free (no-AI) metrics, custom metrics, comparing two pipelines, an automatic pass/fail check, and a bigger LangGraph pipeline.

## What is inside

```
.
├── RAGAS_Crash_Course.ipynb   # step-by-step tutorial (17 sections)
├── ragas_eval_app.py          # small Streamlit dashboard
├── requirements.txt
└── .gitignore
```

## How it works

```
Question -> LangGraph (search -> answer with Groq) -> RAGAS scores
```

- The RAG pipeline is built with **LangGraph** (search step, then answer step).
- **RAGAS** judges the result with a Groq model.
- **HuggingFace** (`BAAI/bge-small-en-v1.5`) gives free embeddings.

## Setup

You need Python 3.11 and a free Groq API key from [console.groq.com](https://console.groq.com).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install the packages
pip install -r requirements.txt ipykernel
```

3. Create a file named `.env` next to the notebook and add your key:

```
GROQ_API_KEY=your_key_here
```

Never share or commit your `.env` file.

## Run it

**Notebook:** open `RAGAS_Crash_Course.ipynb`, choose the `venv` as the kernel, then run the cells from top to bottom.

**Dashboard:**

```bash
streamlit run ragas_eval_app.py
```

Pick the metrics in the sidebar and click **Run Evaluation**.

## Troubleshooting

| Problem | Fix |
|---|---|
| Scores are blank (`None` / `NaN`) | Usually a Groq free-tier rate limit. Wait a minute and run again. |
| `GROQ_API_KEY not found` | Check that `.env` is in the same folder and has the key. |
| `model does not exist` (404) | Groq changes its models often. Change the model name in section 4 of the notebook and in `get_judge()` in the app. Current list: [console.groq.com/docs/models](https://console.groq.com/docs/models). |
| Rate limit says "tokens per day" | The free daily limit is used up. Wait for it to reset, or use another Groq key. |
| `ModuleNotFoundError: ...vertexai` | Known RAGAS bug with newer `langchain-community`. It is already handled in the notebook and app. Keep `ragas==0.3.9` from `requirements.txt`. |

## Notes

- `ragas` is pinned to `0.3.9` on purpose. Newer versions have an import bug at the time of writing.
- Scores from an AI judge can change a little between runs. That is normal.
