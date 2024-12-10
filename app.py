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

**Goal:** Identify how items (e.g., projects) from one dataset match or map to items from another dataset through shared sentence-level clusters.
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

    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    st.session_state['cached_embeddings'][embeddings_file] = embeddings
    return embeddings

def cluster_documents(texts, embeddings):
    stop_words = set(stopwords.words('english'))
    texts_cleaned = []
    for text in texts:
        word_tokens = word_tokenize(text)
        filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
        texts_cleaned.append(filtered_text)

    sentence_model = get_embedding_model()
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings)
    return topics, topic_model

def cluster_sentences(all_texts):
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
        filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
        texts_cleaned.append(filtered_text)

    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(embedding_model=model, hdbscan_model=hdbscan_model)
    topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=sentence_embeddings)

    return sentences, sentence_embeddings, topics, doc_indices, topic_model

def compute_cluster_frequencies(topics, doc_indices, lhs_count, level='document'):
    df = pd.DataFrame({'doc_index': doc_indices, 'topic': topics})
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

COLOR_PALETTE = [
    "#FFB6C1", "#87CEFA", "#98FB98", "#FFD700", "#FFA07A", "#BA55D3", "#00FA9A", "#20B2AA", "#778899", "#FF69B4",
    "#7FFF00", "#DC143C", "#00FFFF", "#FFA500", "#8A2BE2", "#A9A9A9", "#6A5ACD", "#D2691E", "#5F9EA0", "#FF4500"
]

def highlight_text_by_cluster(sentences, sentence_clusters, selected_clusters):
    unique_clusters = list(sorted(set(sentence_clusters)))
    cluster_color_map = {}
    for i, c in enumerate(unique_clusters):
        cluster_color_map[c] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    highlighted_sentences = []
    for s, c in zip(sentences, sentence_clusters):
        color = cluster_color_map.get(c, "#FFFFFF")
        if c in selected_clusters:
            highlighted_sentences.append(f'<span style="background-color:{color}; padding:2px; border-radius:3px">{s}</span>')
        else:
            highlighted_sentences.append(s)
    return " ".join(highlighted_sentences)

tab_instructions, tab_select_text, tab_run, tab_results, tab_cluster_browser, tab_faq = st.tabs(["Instructions", "Select Text Columns", "Run Fingerprint Matching", "Results & Visualization", "Cluster Browser", "FAQ"])

with tab_instructions:
    st.header("How to Use")
    st.markdown("""
1. **Upload Datasets**: On the left sidebar, upload your LHS and RHS datasets.
2. **Select Text Columns**: In the "Select Text Columns" tab, choose the columns from each dataset that contain free-text data.
3. **Run Fingerprint Matching**: In the "Run Fingerprint Matching" tab, select method (Document-level or Sentence-level) and run.
4. **View Results**: In the "Results & Visualization" tab, see assigned clusters and a cluster frequency table.
5. **Optionally Merge Topics**: After initial clustering, you can select certain merges to apply from the hierarchical structure.
6. **Use Cluster Browser**: In the "Cluster Browser" tab, explore clusters, view associated docs/sentences, and highlight text by cluster.
7. **Check Overlaps**: Use the overlap tables to identify clusters that appear in both LHS and RHS.
8. **High-level Similarity**: Compute direct similarity matches between items in LHS and their best match in RHS.
    """)

with tab_faq:
    st.header("FAQ")
    st.markdown("""
**Q:** What data format is required?  
**A:** Excel (.xlsx).

**Q:** Document-level vs Sentence-level?  
**A:** Document-level clusters entire rows. Sentence-level clusters individual sentences.

**Q:** How to see which clusters overlap?  
**A:** Check the frequency tables in "Results & Visualization" and then refer to the overlap table.
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

            lhs_emb = load_or_compute_embeddings(lhs_df, f"lhs_{hash(tuple(lhs_df.columns))}")
            rhs_emb = load_or_compute_embeddings(rhs_df, f"rhs_{hash(tuple(rhs_df.columns))}")

            # Store embeddings for future similarity computations
            st.session_state['lhs_emb'] = lhs_emb
            st.session_state['rhs_emb'] = rhs_emb

            all_texts = lhs_df['combined_text'].tolist() + rhs_df['combined_text'].tolist()

            if method == "Document-level":
                all_emb = np.vstack((lhs_emb, rhs_emb))
                topics, topic_model = cluster_documents(all_texts, all_emb)

                lhs_df['Topic'] = topics[:len(lhs_df)]
                rhs_df['Topic'] = topics[len(lhs_df):]

                st.session_state['lhs_df'] = lhs_df
                st.session_state['rhs_df'] = rhs_df
                st.session_state['doc_topic_model'] = topic_model
                st.session_state['method'] = method
                st.session_state['all_texts'] = all_texts
                st.session_state['original_doc_topics'] = topics
                st.session_state['hierarchical_topics_doc'] = topic_model.hierarchical_topics(all_texts)

                st.success("Document-level fingerprinting completed!")

            elif method == "Sentence-level":
                sentences, sentence_embeddings, topics, doc_indices, topic_model = cluster_sentences(all_texts)

                st.session_state['sentences'] = sentences
                st.session_state['sentence_topics'] = topics
                st.session_state['doc_indices'] = doc_indices
                st.session_state['sent_topic_model'] = topic_model
                st.session_state['lhs_count'] = len(lhs_df)
                st.session_state['rhs_count'] = len(rhs_df)
                st.session_state['method'] = method
                st.session_state['all_texts'] = all_texts
                st.session_state['original_sent_topics'] = topics
                st.session_state['hierarchical_topics_sent'] = topic_model.hierarchical_topics(sentences)

                st.success("Sentence-level fingerprinting completed!")


with tab_results:
    st.header("Results & Visualization")
    if 'method' not in st.session_state:
        st.warning("Please run fingerprint matching first.")
    else:
        method = st.session_state['method']

        if method == "Document-level":
            if ('doc_topic_model' in st.session_state and 
                'lhs_df' in st.session_state and 'rhs_df' in st.session_state):
                
                topic_model = st.session_state['doc_topic_model']
                lhs_df = st.session_state['lhs_df'].copy()
                rhs_df = st.session_state['rhs_df'].copy()
                lhs_df['side'] = "LHS"
                rhs_df['side'] = "RHS"
                combined_df = pd.concat([lhs_df, rhs_df], ignore_index=True)

                st.subheader("Assigned Topics (Document-level)")
                st.dataframe(combined_df[['side', 'combined_text', 'Topic']])

                st.subheader("Topic Visualization")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                freq_df = compute_cluster_frequencies(
                    combined_df['Topic'].values,
                    np.arange(len(combined_df['Topic'].values)),
                    len(lhs_df),
                    level='document'
                )

                st.subheader("Cluster Frequency (Document-level)")
                st.dataframe(freq_df)

                # Overlap table: Only show clusters In_both == True
                st.subheader("Overlapping Clusters (Document-level)")
                overlap_df = freq_df[freq_df['In_both'] == True]
                st.dataframe(overlap_df)

                if 'hierarchical_topics_doc' in st.session_state:
                    hierarchical_topics = st.session_state['hierarchical_topics_doc']
                    st.subheader("Hierarchical Topics (Document-level)")
                    st.dataframe(hierarchical_topics)

                    merges_options = []
                    merges_map = {}
                    for i, row in hierarchical_topics.iterrows():
                        p = row["Parent_ID"]
                        left_c = row["Child_Left_ID"]
                        right_c = row["Child_Right_ID"]
                        d = row["Distance"]
                        option_str = f"Merge {left_c} & {right_c} -> Parent {p} (Distance: {d})"
                        merges_options.append(option_str)
                        merges_map[option_str] = [left_c, right_c]

                    selected_merges = st.multiselect("Select merges to apply", merges_options)

                    if st.button("Apply Merges"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['all_texts'], merges_list)

                            lhs_count = len(st.session_state['lhs_df'])
                            st.session_state['lhs_df']['Topic'] = new_topics[:lhs_count]
                            st.session_state['rhs_df']['Topic'] = new_topics[lhs_count:]
                            st.session_state['doc_topic_model'] = new_model

                            st.success("Merges applied successfully! Refresh the tab to see updated results.")

            else:
                st.warning("No document-level results found.")

        elif method == "Sentence-level":
            if ('sent_topic_model' in st.session_state and 
                'sentence_topics' in st.session_state and 
                'doc_indices' in st.session_state and
                'lhs_count' in st.session_state):
                
                topic_model = st.session_state['sent_topic_model']
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
                st.dataframe(df_sent)

                st.subheader("Topic Visualization (Sentence-level)")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Exclude -1 topics for frequency chart
                df_sent_filtered = df_sent[df_sent['topic'] != -1]
                freq_df = compute_cluster_frequencies(
                    df_sent_filtered['topic'].values,
                    df_sent_filtered['doc_index'].values,
                    lhs_count,
                    level='sentence'
                )

                # Overlap table for sentence-level
                st.subheader("Overlapping Clusters (Sentence-level)")
                overlap_df = freq_df[freq_df['In_both'] == True]
                st.dataframe(overlap_df)

                st.subheader("Cluster Frequency Bar Chart (Sentence-level, excluding -1)")
                # Transform freq_df to long format for a stacked bar chart
                long_freq_df = pd.DataFrame({
                    'cluster': np.concatenate([freq_df['cluster'].values, freq_df['cluster'].values]),
                    'side': ['LHS']*len(freq_df) + ['RHS']*len(freq_df),
                    'count': np.concatenate([freq_df['LHS_sentence_count'].values, freq_df['RHS_sentence_count'].values])
                })

                # Filter out cluster == -1 if present
                long_freq_df = long_freq_df[long_freq_df['cluster'] != -1]

                # Create stacked bar chart
                fig_bar = px.bar(
                    long_freq_df,
                    x='cluster',
                    y='count',
                    color='side',
                    title="Sentence-level Cluster Frequency (Stacked by Side)",
                    barmode='stack'
                )
                st.plotly_chart(fig_bar)

                if 'hierarchical_topics_sent' in st.session_state:
                    hierarchical_topics = st.session_state['hierarchical_topics_sent']
                    st.subheader("Hierarchical Topics (Sentence-level)")
                    st.dataframe(hierarchical_topics)

                    merges_options = []
                    merges_map = {}
                    for i, row in hierarchical_topics.iterrows():
                        p = row["Parent_ID"]
                        left_c = row["Child_Left_ID"]
                        right_c = row["Child_Right_ID"]
                        d = row["Distance"]
                        option_str = f"Merge {left_c} & {right_c} -> Parent {p} (Distance: {d})"
                        merges_options.append(option_str)
                        merges_map[option_str] = [left_c, right_c]

                    selected_merges = st.multiselect("Select merges to apply", merges_options)

                    if st.button("Apply Merges"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['sentences'], merges_list)

                            st.session_state['sentence_topics'] = new_topics
                            st.session_state['sent_topic_model'] = new_model

                            st.success("Merges applied successfully! Refresh the tab to see updated results.")

            else:
                st.warning("No sentence-level results found.")

        # ADDITIONAL PROCESS: High-level Similarity between LHS and RHS items
        # This does not modify existing code, just adds a new section at the end.
        st.subheader("High-level Project Similarities")
        st.markdown("""
        Compute cosine similarities between each LHS item and all RHS items, 
        and find the top match for each LHS item.
        """)
        if 'lhs_emb' in st.session_state and 'rhs_emb' in st.session_state and 'lhs_df' in st.session_state and 'rhs_df' in st.session_state:
            if st.button("Compute High-level Project Similarities"):
                lhs_emb = st.session_state['lhs_emb']
                rhs_emb = st.session_state['rhs_emb']
                lhs_texts = st.session_state['lhs_df']['combined_text'].tolist()
                rhs_texts = st.session_state['rhs_df']['combined_text'].tolist()

                # Compute cosine similarity
                sim = cosine_similarity(lhs_emb, rhs_emb)
                # For each LHS, find the best match in RHS
                best_matches = sim.argmax(axis=1)  # index of best match for each LHS doc
                best_scores = sim.max(axis=1)      # best score for each LHS doc

                # Create a result dataframe
                results = []
                for i, (best_idx, score) in enumerate(zip(best_matches, best_scores)):
                    results.append({
                        "LHS_Index": i,
                        "LHS_Text": lhs_texts[i],
                        "Matched_RHS_Index": best_idx,
                        "Matched_RHS_Text": rhs_texts[best_idx],
                        "Similarity_Score": score
                    })
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

with tab_cluster_browser:
    st.header("Cluster Browser and Manual Review")

    if 'method' not in st.session_state:
        st.warning("Run fingerprint matching first.")
    else:
        method = st.session_state['method']

        if method == "Document-level":
            if 'lhs_df' in st.session_state and 'rhs_df' in st.session_state:
                lhs_df = st.session_state['lhs_df']
                rhs_df = st.session_state['rhs_df']
                lhs_count = len(lhs_df)
                combined_df = pd.concat([lhs_df.assign(side='LHS'), rhs_df.assign(side='RHS')], ignore_index=True)
                all_topics = sorted(combined_df['Topic'].unique())

                selected_clusters = st.multiselect("Select clusters to explore", all_topics)
                if selected_clusters:
                    subset = combined_df[combined_df['Topic'].isin(selected_clusters)]
                    lhs_subset = subset[subset['side']=='LHS']
                    rhs_subset = subset[subset['side']=='RHS']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**LHS Documents in selected clusters:**")
                        st.dataframe(lhs_subset[['side','combined_text','Topic']])
                    with col2:
                        st.write("**RHS Documents in selected clusters:**")
                        st.dataframe(rhs_subset[['side','combined_text','Topic']])

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

                freq_df = compute_cluster_frequencies(df_sent['topic'].values, df_sent['doc_index'].values, lhs_count, level='sentence')
                freq_df['Total_count'] = freq_df['LHS_sentence_count'] + freq_df['RHS_sentence_count']
                freq_df['Balance'] = freq_df['LHS_sentence_count'] - freq_df['RHS_sentence_count']

                st.subheader("Cluster Selection")
                st.markdown("Select a cluster to view sentences from LHS and RHS side-by-side.")
                available_clusters = freq_df['cluster'].sort_values()
                selected_cluster = st.selectbox("Select a cluster to view sentences", available_clusters)
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

                    st.markdown("### Highlight a Single Document")
                    unique_docs_in_subset = sorted(cluster_subset['doc_index'].unique())
                    selected_doc = st.selectbox("Select a document index to highlight its text", unique_docs_in_subset)
                    if selected_doc is not None:
                        doc_sents = df_sent[df_sent['doc_index'] == selected_doc]
                        doc_sentences = doc_sents['sentence'].tolist()
                        doc_sent_topics = doc_sents['topic'].tolist()

                        highlighted_html = highlight_text_by_cluster(doc_sentences, doc_sent_topics, [selected_cluster])
                        st.markdown(highlighted_html, unsafe_allow_html=True)

                        doc_side = "LHS" if selected_doc < lhs_count else "RHS"
                        st.write(f"Document index: {selected_doc} (Side: {doc_side})")

            else:
                st.warning("No sentence-level data available.")
