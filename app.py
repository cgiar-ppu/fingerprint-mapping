import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
import base64
import re
import pickle
import concurrent.futures

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# For summarization (optional, requires OPENAI_API_KEY)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

device = 'cpu'

def init_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

init_nltk_resources()

st.set_page_config(
    page_title="Fingerprint Matching",
    layout="wide",
)

st.title("Fingerprint Matching Prototype")

st.markdown("""
This application allows you to upload two datasets (left and right), select text columns to combine, and then run "fingerprint" matching algorithms.  
The matching is done by clustering the combined texts or their sentences, and then measuring overlap in clusters between items from the left and right datasets.
""")

# Sidebar for dataset upload
st.sidebar.title("Dataset Upload")
st.sidebar.markdown("**Left-hand Side (LHS) Dataset:**")
lhs_file = st.sidebar.file_uploader("Upload LHS dataset (Excel)", type=["xlsx"], key="lhs")
if lhs_file is not None:
    lhs_df = pd.read_excel(lhs_file)
else:
    lhs_df = None

st.sidebar.markdown("**Right-hand Side (RHS) Dataset:**")
rhs_file = st.sidebar.file_uploader("Upload RHS dataset (Excel)", type=["xlsx"], key="rhs")
if rhs_file is not None:
    rhs_df = pd.read_excel(rhs_file)
else:
    rhs_df = None

@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

def combine_columns_into_text(df: pd.DataFrame, selected_columns: list):
    if not selected_columns:
        return pd.Series(["" for _ in range(len(df))], index=df.index)
    combined_text = df[selected_columns].astype(str).apply(lambda x: " ".join(x), axis=1)
    return combined_text

def generate_embeddings(texts, model):
    with st.spinner('Calculating embeddings...'):
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

def load_or_compute_embeddings(df, unique_id):
    # unique_id could be based on file name and column selection
    embeddings_file = f"{unique_id}.pkl"
    texts = df['combined_text'].tolist()

    if 'cached_embeddings' not in st.session_state:
        st.session_state['cached_embeddings'] = {}

    if embeddings_file in st.session_state['cached_embeddings']:
        embeddings = st.session_state['cached_embeddings'][embeddings_file]
        return embeddings

    # If file doesn't exist or no session cache, compute
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    st.session_state['cached_embeddings'][embeddings_file] = embeddings
    return embeddings

def cluster_documents(texts, embeddings):
    # Perform clustering on document level
    stop_words = set(stopwords.words('english'))
    texts_cleaned = []
    for text in texts:
        word_tokens = word_tokenize(text)
        filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
        texts_cleaned.append(filtered_text)

    sentence_model = get_embedding_model()
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings)
    return topics, topic_model

def cluster_sentences(all_texts, all_embeddings):
    # Sentence-level clustering
    # 1. Split each document into sentences
    # 2. Embeddings must correspond to sentences, so we must re-embed at sentence level
    sentences = []
    doc_indices = []
    for i, doc_text in enumerate(all_texts):
        sents = sent_tokenize(doc_text)
        sentences.extend(sents)
        doc_indices.extend([i]*len(sents))

    model = get_embedding_model()
    sentence_embeddings = model.encode(sentences, show_progress_bar=True, device=device)

    # Cluster sentences
    stop_words = set(stopwords.words('english'))
    texts_cleaned = []
    for text in sentences:
        word_tokens = word_tokenize(text)
        filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
        texts_cleaned.append(filtered_text)

    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=sentence_embeddings)

    # Return sentence-level clusters, plus mapping back to documents
    return sentences, sentence_embeddings, topics, doc_indices, topic_model

def compute_overlap_metrics(lhs_topics, rhs_topics):
    # This function computes some high-level overlap metrics between two lists of topics.
    # For document-level clustering:
    # - Compute how many clusters overlap. For instance, we can measure the Jaccard similarity
    #   of sets of topics or how many documents share the same topic.
    lhs_topic_counts = pd.Series(lhs_topics).value_counts()
    rhs_topic_counts = pd.Series(rhs_topics).value_counts()

    # Intersection of topic labels present in both sets
    common_topics = set(lhs_topic_counts.index).intersection(set(rhs_topic_counts.index))
    lhs_common = lhs_topic_counts[lhs_topic_counts.index.isin(common_topics)].sum()
    rhs_common = rhs_topic_counts[rhs_topic_counts.index.isin(common_topics)].sum()

    # Simple metric: fraction of documents from LHS that share at least one topic with RHS
    # (Here it's simplistic because each doc has only one main topic in doc-level clustering)
    # For sentence-level, we can refine.

    total_lhs = len(lhs_topics)
    total_rhs = len(rhs_topics)

    # Overlap ratio (very simplistic metric)
    overlap_ratio = (lhs_common + rhs_common) / (total_lhs + total_rhs)

    return {
        "Number_of_common_topics": len(common_topics),
        "Overlap_ratio": overlap_ratio
    }

def compute_sentence_cluster_overlap(doc_indices, topics, lhs_count):
    # Here doc_indices indicate which doc each sentence belongs to.
    # First portion (0 to lhs_count-1) are LHS documents, the rest are RHS.
    lhs_indices = [i for i in range(lhs_count)]
    rhs_indices = [i for i in range(lhs_count, max(doc_indices)+1)]

    # Map docs to sets of clusters
    doc_to_clusters = {}
    for i, t in enumerate(topics):
        d = doc_indices[i]
        if d not in doc_to_clusters:
            doc_to_clusters[d] = set()
        doc_to_clusters[d].add(t)

    # Compute pairwise overlaps between LHS and RHS docs
    # This could be large, so we just do a summary
    # For each LHS doc, find best matching RHS doc by largest cluster intersection
    lhs_docs = [d for d in doc_to_clusters if d in lhs_indices]
    rhs_docs = [d for d in doc_to_clusters if d in rhs_indices]

    best_matches = []
    for ld in lhs_docs:
        ld_clusters = doc_to_clusters[ld]
        best_match = None
        best_score = 0
        for rd in rhs_docs:
            rd_clusters = doc_to_clusters[rd]
            intersection = ld_clusters.intersection(rd_clusters)
            score = len(intersection)
            if score > best_score:
                best_score = score
                best_match = rd
        best_matches.append((ld, best_match, best_score))
    return best_matches

def display_overlaps(merge_df, method):
    st.subheader("Overlap Metrics")
    st.write("Here we show how items from LHS map to items from RHS based on the chosen clustering method.")

    if method == "Document-level":
        lhs_topics = merge_df[merge_df['side']=="LHS"]['Topic'].values
        rhs_topics = merge_df[merge_df['side']=="RHS"]['Topic'].values
        metrics = compute_overlap_metrics(lhs_topics, rhs_topics)
        st.write("Number of Common Topics:", metrics["Number_of_common_topics"])
        st.write("Overlap Ratio:", metrics["Overlap_ratio"])
    elif method == "Sentence-level":
        # We performed sentence-level clustering already
        # The details are computed separately (in the main logic)
        pass


# Tabs: We will create a flow
tab_instructions, tab_select_text, tab_run, tab_results, tab_faq = st.tabs(["Instructions", "Select Text Columns", "Run Fingerprint Matching", "Results & Visualization", "FAQ"])

with tab_instructions:
    st.header("How to Use")
    st.markdown("""
1. **Upload Datasets**: On the left sidebar, upload your LHS and RHS datasets (Excel files).
2. **Select Text Columns**: Navigate to the "Select Text Columns" tab. Choose the columns from each dataset that contain free-text data. These will be combined into a single text field per row.
3. **Run Fingerprint Matching**: In the "Run Fingerprint Matching" tab, select your method (Document-level or Sentence-level clustering) and start the process.  
   - Document-level clustering treats each row as a single document and clusters them.  
   - Sentence-level clustering splits each document into sentences, clusters all sentences, and then aggregates results back to the document level.
4. **View Results**: Check the "Results & Visualization" tab to see the assigned clusters and overlap metrics.
5. **FAQ**: Visit the "FAQ" tab for common questions and troubleshooting.
    """)

with tab_faq:
    st.header("FAQ")
    st.markdown("""
**Q:** What data format is required?  
**A:** The tool accepts Excel files (.xlsx). Ensure your text columns are readable by Pandas.

**Q:** How large can my datasets be?  
**A:** Larger datasets will take longer to process and cluster. There are no strict size limits here, but performance may degrade with very large inputs.

**Q:** What is Document-level vs Sentence-level clustering?  
**A:** Document-level clustering groups entire documents together into topics. Sentence-level clustering breaks documents into sentences, clusters them, and then derives overlap based on shared sentence-level topics.

**Q:** I get no common topics. What does that mean?  
**A:** It could mean that the two sets of documents are semantically disjoint based on the chosen model and clustering algorithm.

**Q:** Can I download the results?  
**A:** Yes, after running clustering, you can download CSV files of the assigned topics and any summary tables provided.
    """)


with tab_select_text:
    st.header("Select Text Columns")
    if lhs_df is None or rhs_df is None:
        st.warning("Please upload both LHS and RHS datasets first.")
    else:
        lhs_columns = lhs_df.columns.tolist()
        rhs_columns = rhs_df.columns.tolist()

        st.subheader("LHS Dataset Columns")
        lhs_selected_cols = st.multiselect("Select columns to combine as text (LHS)", lhs_columns, default=lhs_columns[:1] if lhs_columns else [])

        st.subheader("RHS Dataset Columns")
        rhs_selected_cols = st.multiselect("Select columns to combine as text (RHS)", rhs_columns, default=rhs_columns[:1] if rhs_columns else [])

        if st.button("Combine Columns"):
            if lhs_selected_cols and rhs_selected_cols:
                lhs_df['combined_text'] = combine_columns_into_text(lhs_df, lhs_selected_cols)
                rhs_df['combined_text'] = combine_columns_into_text(rhs_df, rhs_selected_cols)
                st.session_state['lhs_df'] = lhs_df
                st.session_state['rhs_df'] = rhs_df
                st.success("Text columns combined successfully!")
            else:
                st.warning("Please select at least one column from both datasets.")

        if 'lhs_df' in st.session_state and 'rhs_df' in st.session_state:
            st.write("LHS Sample:")
            st.write(st.session_state['lhs_df'].head())
            st.write("RHS Sample:")
            st.write(st.session_state['rhs_df'].head())


with tab_run:
    st.header("Run Fingerprint Matching")
    if 'lhs_df' not in st.session_state or 'rhs_df' not in st.session_state:
        st.warning("Please upload datasets and select text columns first.")
    else:
        method = st.selectbox("Select fingerprinting method", ["Document-level", "Sentence-level"])

        if st.button("Run Fingerprint Matching"):
            lhs_df = st.session_state['lhs_df']
            rhs_df = st.session_state['rhs_df']

            # Compute embeddings for LHS and RHS separately, then combine
            lhs_emb = load_or_compute_embeddings(lhs_df, f"lhs_{hash(tuple(lhs_df.columns))}")
            rhs_emb = load_or_compute_embeddings(rhs_df, f"rhs_{hash(tuple(rhs_df.columns))}")

            if method == "Document-level":
                # Combine data
                all_texts = lhs_df['combined_text'].tolist() + rhs_df['combined_text'].tolist()
                all_emb = np.vstack((lhs_emb, rhs_emb))
                topics, topic_model = cluster_documents(all_texts, all_emb)

                # Assign topics back
                lhs_df['Topic'] = topics[:len(lhs_df)]
                rhs_df['Topic'] = topics[len(lhs_df):]

                # Store results
                st.session_state['lhs_df'] = lhs_df
                st.session_state['rhs_df'] = rhs_df
                st.session_state['doc_topic_model'] = topic_model
                st.session_state['method'] = method

                st.success("Document-level fingerprinting completed!")

            elif method == "Sentence-level":
                # Sentence-level clustering
                all_texts = lhs_df['combined_text'].tolist() + rhs_df['combined_text'].tolist()
                # We ignore doc-level embeddings (lhs_emb, rhs_emb) for sentence-level,
                # as we re-embed at sentence level.
                sentences, sentence_embeddings, topics, doc_indices, topic_model = cluster_sentences(all_texts, None)

                # Store sentence-level data
                st.session_state['sentences'] = sentences
                st.session_state['sentence_topics'] = topics
                st.session_state['doc_indices'] = doc_indices
                st.session_state['sent_topic_model'] = topic_model
                st.session_state['lhs_count'] = len(lhs_df)
                st.session_state['rhs_count'] = len(rhs_df)
                st.session_state['method'] = method

                st.success("Sentence-level fingerprinting completed!")


with tab_results:
    st.header("Results & Visualization")
    if 'method' not in st.session_state:
        st.warning("Please run fingerprint matching first.")
    else:
        method = st.session_state['method']
        if method == "Document-level":
            if 'lhs_df' in st.session_state and 'rhs_df' in st.session_state and 'doc_topic_model' in st.session_state:
                lhs_df = st.session_state['lhs_df'].copy()
                rhs_df = st.session_state['rhs_df'].copy()
                lhs_df['side'] = "LHS"
                rhs_df['side'] = "RHS"
                combined_df = pd.concat([lhs_df, rhs_df], ignore_index=True)
                
                st.subheader("Assigned Topics (Document-level)")
                st.write(combined_df[['side', 'combined_text', 'Topic']].head(50))

                # Visualizations
                topic_model = st.session_state['doc_topic_model']
                st.subheader("Topic Visualization")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                display_overlaps(combined_df, method)

                # Download results
                csv = combined_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="fingerprint_results_document_level.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No document-level results found.")

        elif method == "Sentence-level":
            if 'sentences' in st.session_state and 'sentence_topics' in st.session_state and 'doc_indices' in st.session_state:
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']
                lhs_count = st.session_state['lhs_count']

                # Create a DataFrame with sentence-level info
                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics
                })

                # Mark LHS vs RHS
                df_sent['side'] = np.where(df_sent['doc_index'] < lhs_count, "LHS", "RHS")

                st.subheader("Assigned Topics (Sentence-level)")
                st.write(df_sent.head(50))

                topic_model = st.session_state['sent_topic_model']
                st.subheader("Topic Visualization (Sentence-level)")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Compute overlap by sentence-level clusters
                best_matches = compute_sentence_cluster_overlap(doc_indices, topics, lhs_count)
                match_df = pd.DataFrame(best_matches, columns=["LHS_doc_index", "Best_RHS_doc_index", "Shared_cluster_count"])
                st.subheader("Document Overlap Based on Sentence-level Topics")
                st.write(match_df.head(50))

                # Download results
                csv = df_sent.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="fingerprint_results_sentence_level.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No sentence-level results found.")
