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
import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# For summarization (optional)
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

def cluster_sentences(all_texts):
    # Sentence-level clustering
    sentences = []
    doc_indices = []
    for i, doc_text in enumerate(all_texts):
        sents = sent_tokenize(doc_text)
        sentences.extend(sents)
        doc_indices.extend([i]*len(sents))

    model = get_embedding_model()
    sentence_embeddings = model.encode(sentences, show_progress_bar=True, device=device)

    stop_words = set(stopwords.words('english'))
    texts_cleaned = []
    for text in sentences:
        word_tokens = word_tokenize(text)
        filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
        texts_cleaned.append(filtered_text)

    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=sentence_embeddings)

    return sentences, sentence_embeddings, topics, doc_indices, topic_model

def compute_cluster_frequencies(topics, lhs_count, level='document'):
    """Compute how many docs or sentences in LHS and RHS fall into each cluster."""
    df = pd.DataFrame({'doc_index': range(len(topics)), 'topic': topics})
    lhs_mask = df['doc_index'] < lhs_count
    rhs_mask = ~lhs_mask

    lhs_counts = df[lhs_mask]['topic'].value_counts()
    rhs_counts = df[rhs_mask]['topic'].value_counts()
    all_clusters = sorted(set(topics))
    freq_data = []
    for c in all_clusters:
        lhs_c = lhs_counts.get(c, 0)
        rhs_c = rhs_counts.get(c, 0)
        overlap = (lhs_c > 0 and rhs_c > 0)
        freq_data.append({
            'cluster': c,
            f'LHS_{level}_count': lhs_c,
            f'RHS_{level}_count': rhs_c,
            'In_both': overlap
        })
    freq_df = pd.DataFrame(freq_data)
    return freq_df

def compute_sentence_cluster_overlap(doc_indices, topics, lhs_count):
    # Map docs to sets of clusters
    doc_to_clusters = {}
    for i, t in enumerate(topics):
        d = doc_indices[i]
        if d not in doc_to_clusters:
            doc_to_clusters[d] = set()
        doc_to_clusters[d].add(t)

    lhs_indices = [d for d in doc_to_clusters if d < lhs_count]
    rhs_indices = [d for d in doc_to_clusters if d >= lhs_count]

    best_matches = []
    for ld in lhs_indices:
        ld_clusters = doc_to_clusters[ld]
        best_match = None
        best_score = 0
        for rd in rhs_indices:
            rd_clusters = doc_to_clusters[rd]
            intersection = ld_clusters.intersection(rd_clusters)
            score = len(intersection)
            if score > best_score:
                best_score = score
                best_match = rd
        best_matches.append((ld, best_match, best_score))
    return best_matches

# Color palette for highlighting clusters
COLOR_PALETTE = [
    "#FFB6C1", "#87CEFA", "#98FB98", "#FFD700", "#FFA07A", "#BA55D3", "#00FA9A", "#20B2AA", "#778899", "#FF69B4",
    "#7FFF00", "#DC143C", "#00FFFF", "#FFA500", "#8A2BE2", "#A9A9A9", "#6A5ACD", "#D2691E", "#5F9EA0", "#FF4500"
]

def highlight_text_by_cluster(sentences, sentence_clusters, selected_clusters):
    # Highlight only if selected_clusters is not empty
    # Assign each cluster a color
    unique_clusters = list(sorted(set(sentence_clusters)))
    cluster_color_map = {}
    for i, c in enumerate(unique_clusters):
        cluster_color_map[c] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    highlighted_sentences = []
    for s, c in zip(sentences, sentence_clusters):
        color = cluster_color_map.get(c, "#FFFFFF")
        # If we only want to highlight selected clusters, check that:
        if c in selected_clusters:
            highlighted_sentences.append(f'<span style="background-color:{color}; padding:2px; border-radius:3px">{s}</span>')
        else:
            highlighted_sentences.append(s)
    return " ".join(highlighted_sentences)


# Tabs
tab_instructions, tab_select_text, tab_run, tab_results, tab_cluster_browser, tab_faq = st.tabs(["Instructions", "Select Text Columns", "Run Fingerprint Matching", "Results & Visualization", "Cluster Browser", "FAQ"])

with tab_instructions:
    st.header("How to Use")
    st.markdown("""
1. **Upload Datasets**: On the left sidebar, upload your LHS and RHS datasets.
2. **Select Text Columns**: In the "Select Text Columns" tab, choose the columns from each dataset that contain free-text data.
3. **Run Fingerprint Matching**: In the "Run Fingerprint Matching" tab, select method (Document-level or Sentence-level) and run.
4. **View Results**: In the "Results & Visualization" tab, see assigned clusters and a cluster frequency table.
5. **Use Cluster Browser**: In the "Cluster Browser" tab, explore clusters, view associated docs/sentences from both sides, and optionally highlight text by cluster.
6. **FAQ**: Check for common questions.
    """)

with tab_faq:
    st.header("FAQ")
    st.markdown("""
**Q:** What data format is required?  
**A:** Excel (.xlsx).

**Q:** Document-level vs Sentence-level?  
**A:** Document-level clusters entire rows. Sentence-level clusters individual sentences.

**Q:** How to see which clusters overlap?  
**A:** Check the frequency tables in "Results & Visualization" and then go to "Cluster Browser" to see documents/sentences in chosen clusters.
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
            st.dataframe(st.session_state['lhs_df'])
            st.write("RHS Sample:")
            st.dataframe(st.session_state['rhs_df'])


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
                all_texts = lhs_df['combined_text'].tolist() + rhs_df['combined_text'].tolist()
                all_emb = np.vstack((lhs_emb, rhs_emb))
                topics, topic_model = cluster_documents(all_texts, all_emb)

                lhs_df['Topic'] = topics[:len(lhs_df)]
                rhs_df['Topic'] = topics[len(lhs_df):]

                st.session_state['lhs_df'] = lhs_df
                st.session_state['rhs_df'] = rhs_df
                st.session_state['doc_topic_model'] = topic_model
                st.session_state['method'] = method
                st.session_state['all_texts'] = all_texts

                st.success("Document-level fingerprinting completed!")

            elif method == "Sentence-level":
                all_texts = lhs_df['combined_text'].tolist() + rhs_df['combined_text'].tolist()
                sentences, sentence_embeddings, topics, doc_indices, topic_model = cluster_sentences(all_texts)

                st.session_state['sentences'] = sentences
                st.session_state['sentence_topics'] = topics
                st.session_state['doc_indices'] = doc_indices
                st.session_state['sent_topic_model'] = topic_model
                st.session_state['lhs_count'] = len(lhs_df)
                st.session_state['rhs_count'] = len(rhs_df)
                st.session_state['method'] = method
                st.session_state['all_texts'] = all_texts
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
                st.dataframe(combined_df[['side', 'combined_text', 'Topic']])

                topic_model = st.session_state['doc_topic_model']
                st.subheader("Topic Visualization")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Show cluster frequency
                freq_df = compute_cluster_frequencies(combined_df['Topic'].values, len(lhs_df), level='document')
                st.subheader("Cluster Frequency (Document-level)")
                st.dataframe(freq_df)

            else:
                st.warning("No document-level results found.")

        elif method == "Sentence-level":
            if 'sentences' in st.session_state and 'sentence_topics' in st.session_state and 'doc_indices' in st.session_state:
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']
                lhs_count = st.session_state['lhs_count']

                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics,
                    'side': np.where(pd.Series(doc_indices)<lhs_count, "LHS", "RHS")
                })

                st.subheader("Assigned Topics (Sentence-level)")
                st.dataframe(df_sent)  # full table

                topic_model = st.session_state['sent_topic_model']
                st.subheader("Topic Visualization (Sentence-level)")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Show cluster frequency
                freq_df = compute_cluster_frequencies(df_sent['topic'].values, lhs_count, level='sentence')
                st.subheader("Cluster Frequency (Sentence-level)")
                st.dataframe(freq_df)
            else:
                st.warning("No sentence-level results found.")


with tab_cluster_browser:
    st.header("Cluster Browser and Manual Review")

    if 'method' not in st.session_state:
        st.warning("Run fingerprint matching first.")
    else:
        method = st.session_state['method']
        all_texts = st.session_state.get('all_texts', [])

        if method == "Document-level":
            # Existing Document-level code remains as before
            if 'lhs_df' in st.session_state and 'rhs_df' in st.session_state:
                lhs_df = st.session_state['lhs_df']
                rhs_df = st.session_state['rhs_df']
                lhs_count = len(lhs_df)
                combined_df = pd.concat([lhs_df.assign(side='LHS'), rhs_df.assign(side='RHS')], ignore_index=True)
                all_topics = sorted(combined_df['Topic'].unique())

                selected_clusters = st.multiselect("Select clusters to explore", all_topics)
                if selected_clusters:
                    subset = combined_df[combined_df['Topic'].isin(selected_clusters)]
                    st.write("Documents in selected clusters:")
                    st.dataframe(subset[['side','combined_text','Topic']])

                st.info("For sentence-level highlighting, run the Sentence-level method.")

        elif method == "Sentence-level":
            if 'sentences' in st.session_state and 'sentence_topics' in st.session_state and 'doc_indices' in st.session_state:
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']
                lhs_count = st.session_state['lhs_count']
                rhs_count = st.session_state['rhs_count']

                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics,
                    'side': np.where(pd.Series(doc_indices)<lhs_count, "LHS", "RHS")
                })

                # Show bubble chart visualization
                # Compute frequencies again:
                freq_df = compute_cluster_frequencies(df_sent['topic'].values, lhs_count, level='sentence')
                # freq_df contains columns: cluster, LHS_sentence_count, RHS_sentence_count, In_both
                # Let's define total_count = LHS_count + RHS_count and a color_metric = LHS_count - RHS_count
                freq_df['Total_count'] = freq_df['LHS_sentence_count'] + freq_df['RHS_sentence_count']
                freq_df['Balance'] = freq_df['LHS_sentence_count'] - freq_df['RHS_sentence_count']

                st.subheader("Cluster Bubble Chart")
                st.markdown("""
                **Instructions**:  
                - Hover over bubbles to see cluster details.  
                - The bubble size = total number of sentences in that cluster.  
                - The bubble color = balance (LHS - RHS). Positive = more LHS sentences, Negative = more RHS.  
                - Identify a cluster of interest and then select it from the dropdown below to filter sentences.
                """)

                # Create a bubble chart
                # x-axis: cluster ID, y-axis: Total_count (just for visualization)
                # size: Total_count, color: Balance
                fig = px.scatter(
                    freq_df, 
                    x='cluster', 
                    y='Total_count', 
                    size='Total_count',
                    color='Balance',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    hover_data=['cluster', 'LHS_sentence_count', 'RHS_sentence_count', 'In_both'],
                    title="Cluster Overview"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Now, allow user to select a single cluster to filter
                selected_cluster = st.selectbox("Select a cluster to view sentences", freq_df['cluster'].sort_values())
                if selected_cluster is not None:
                    cluster_subset = df_sent[df_sent['topic'] == selected_cluster]
                    lhs_subset = cluster_subset[cluster_subset['side'] == "LHS"]
                    rhs_subset = cluster_subset[cluster_subset['side'] == "RHS"]

                    st.subheader(f"Sentences in Cluster {selected_cluster}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**LHS Sentences:**")
                        st.dataframe(lhs_subset[['doc_index', 'sentence', 'topic']])
                    with col2:
                        st.write("**RHS Sentences:**")
                        st.dataframe(rhs_subset[['doc_index', 'sentence', 'topic']])

                    # Highlighting text (existing code)
                    st.markdown("### Highlight a Single Document")
                    unique_docs_in_subset = sorted(cluster_subset['doc_index'].unique())
                    selected_doc = st.selectbox("Select a document index to highlight its text", unique_docs_in_subset)
                    if selected_doc is not None:
                        doc_sents = df_sent[df_sent['doc_index'] == selected_doc]
                        doc_sentences = doc_sents['sentence'].tolist()
                        doc_sent_topics = doc_sents['topic'].tolist()

                        # highlight only the selected cluster or all selected clusters?
                        # Here we highlight only the chosen cluster.
                        highlighted_html = highlight_text_by_cluster(doc_sentences, doc_sent_topics, [selected_cluster])
                        st.markdown(highlighted_html, unsafe_allow_html=True)

                        doc_side = "LHS" if selected_doc < lhs_count else "RHS"
                        st.write(f"Document index: {selected_doc} (Side: {doc_side})")

            else:
                st.warning("No sentence-level data available.")
