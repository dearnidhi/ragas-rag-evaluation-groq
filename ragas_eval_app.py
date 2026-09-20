import os
import sys
import types

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()



# ragas imports a module that newer langchain-community removed (known bug).
# We don't use Vertex AI, so register an empty stand-in before importing ragas.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = stub

st.set_page_config(page_title="RAGAS Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAGAS Evaluation Dashboard")
st.caption("Evaluate RAG responses using RAGAS + Groq")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your .env file.")
    st.stop()

THRESHOLD = 0.7  # rows scoring below this get flagged



# ---------------- Demo RAG Data ----------------
df = pd.DataFrame([
    {
        "question": "What does the Pro tier include?",
        "contexts": ["The Pro tier includes 500GB storage, 10 user seats, priority support, and API access."],
        "answer": "The Pro tier includes 500GB storage, 10 user seats, priority support, and API access.",
        "reference": "The Pro tier includes 500GB storage, 10 user seats, priority support, and API access.",
    },
    {
        "question": "How is data protected?",
        "contexts": ["Data is encrypted at rest using AES-256 and in transit using TLS 1.3."],
        "answer": "Data is encrypted using AES-256 and TLS 1.3.",
        "reference": "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
    },
])

# ---------------- Metrics ----------------
st.sidebar.header("Metrics")
faithfulness = st.sidebar.checkbox("Faithfulness", True)
relevancy = st.sidebar.checkbox("Response Relevancy", True)
recall = st.sidebar.checkbox("Context Recall", True)
factual = st.sidebar.checkbox("Factual Correctness", True)
st.subheader("RAG Pipeline Output")
st.dataframe(df.drop(columns="contexts"), width="stretch")

# ---------------- RAGAS Judge ----------------
@st.cache_resource
def get_judge():
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def build_dataset():
    from ragas import EvaluationDataset, SingleTurnSample

    return EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r.question,
            retrieved_contexts=r.contexts,
            response=r.answer,
            reference=r.reference,
        )
        for r in df.itertuples()
    ])


def get_metrics():
    from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall, ResponseRelevancy

    chosen = [
        (faithfulness, Faithfulness()),
        (relevancy, ResponseRelevancy(strictness=1)),  # Groq only allows n=1
        (recall, LLMContextRecall()),
        (factual, FactualCorrectness()),
    ]
    return [metric for is_on, metric in chosen if is_on]


# ---------------- Run Evaluation ----------------
if st.button("🚀 Run Evaluation", type="primary"):
    metrics = get_metrics()
    if not metrics:
        st.warning("Select at least one metric.")
        st.stop()

    from ragas import evaluate
    from ragas.run_config import RunConfig

    llm, embeddings = get_judge()
    with st.spinner("Evaluating with RAGAS..."):
        result = evaluate(
            dataset=build_dataset(),
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=4, max_wait=90),  # low concurrency = fewer rate limits
        )

    result_df = result.to_pandas()
    scores = [c for c in result_df.columns
              if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    st.subheader("📊 Results")
    st.dataframe(result_df, width="stretch")

    if result_df[scores].isna().any().any():
        st.warning("Some scores are blank (NaN): the judge call failed, usually a Groq rate limit. Run again.")

# ---------------- Average Scores ----------------
    st.subheader("📈 Average Scores")
    avg = result_df[scores].mean().round(3)
    for col, metric in zip(st.columns(len(scores)), scores):
        col.metric(metric.split("(")[0].replace("_", " ").title(), f"{avg[metric]:.3f}")
    st.bar_chart(avg)

    # ---------------- Low Scores ----------------
    flagged = result_df[(result_df[scores] < THRESHOLD).any(axis=1)]
    if len(flagged):
        st.warning(f"{len(flagged)} row(s) scored below {THRESHOLD}")
        st.dataframe(flagged, width="stretch")
    else:
        st.success(f"All rows scored ≥ {THRESHOLD}")
