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
This application allows you to upload multiple datasets (2 to 5), select text columns to combine, and then run "fingerprint" matching algorithms.  
The matching is done by clustering the combined texts or their sentences, and then measuring overlap in clusters between items from these datasets.

**Goal:** Identify how items (e.g., projects) from one dataset match or map to items from others through shared sentence-level clusters.
""")

# Step 1: Select number of datasets
num_datasets = st.sidebar.selectbox("Number of Datasets", [2,3,4,5], index=0)

# Sidebar for dataset uploads
datasets = []
for i in range(num_datasets):
    st.sidebar.markdown(f"**Dataset {i+1}:**")
    file = st.sidebar.file_uploader(f"Upload dataset {i+1} (Excel)", type=["xlsx"], key=f"file_{i}")
    if file is not None:
        df = pd.read_excel(file)
    else:
        df = None
    datasets.append(df)

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

    # We'll keep cached embeddings in session_state to avoid repeated computation
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

def compute_cluster_frequencies(topics, doc_indices, dataset_sizes, level='document'):
    """
    Compute how many docs or sentences in each dataset fall into each cluster.
    dataset_sizes: list of sizes of each dataset
    """
    df = pd.DataFrame({'doc_index': doc_indices, 'topic': topics})
    freq_data = []
    all_clusters = sorted(set(topics))

    # Precompute the doc_index ranges for each dataset
    dataset_counts = []
    start = 0
    for size in dataset_sizes:
        end = start + size
        mask = (df['doc_index'] >= start) & (df['doc_index'] < end)
        counts = df[mask]['topic'].value_counts()
        dataset_counts.append(counts)
        start = end

    for c in all_clusters:
        row = {'cluster': c}
        in_any = False
        for i, counts in enumerate(dataset_counts):
            cnt = counts.get(c, 0)
            row[f'Dataset{i+1}_{level}_count'] = cnt
            if cnt > 0:
                in_any = True

        # Determine if cluster is in more than one dataset
        nonzero_counts = [row[k] for k in row.keys() if '_count' in k and row[k] > 0]
        row['In_multiple'] = (len(nonzero_counts) > 1)
        freq_data.append(row)

    freq_df = pd.DataFrame(freq_data)
    return freq_df

def highlight_text_by_cluster(sentences, sentence_clusters, selected_clusters):
    COLOR_PALETTE = [
        "#FFB6C1", "#87CEFA", "#98FB98", "#FFD700", "#FFA07A", "#BA55D3", "#00FA9A", "#20B2AA", "#778899", "#FF69B4",
        "#7FFF00", "#DC143C", "#00FFFF", "#FFA500", "#8A2BE2", "#A9A9A9", "#6A5ACD", "#D2691E", "#5F9EA0", "#FF4500"
    ]
    unique_clusters = list(sorted(set(sentence_clusters)))
    cluster_color_map = {}
    for i, c in enumerate(unique_clusters):
        cluster_color_map[c] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    highlighted_sentences = []
    for s, c in zip(sentences, sentence_clusters):
        color = cluster_color_map.get(c, "#FFFFFF")
        if c in selected_clusters:
            highlighted_sentences.append(
                f'<span style="background-color:{color}; padding:2px; border-radius:3px">{s}</span>'
            )
        else:
            highlighted_sentences.append(s)
    return " ".join(highlighted_sentences)


tab_instructions, tab_select_text, tab_run, tab_results, tab_cluster_browser, tab_cosine_sim, tab_faq = st.tabs(
    ["Instructions", "Select Text Columns", "Run Fingerprint Matching", "Results & Visualization", "Cluster Browser", "Cosine Similarity", "FAQ"]
)

# -------------------------------------------------------------------------------------------
# Instructions
# -------------------------------------------------------------------------------------------
with tab_instructions:
    st.header("How to Use")
    st.markdown(f"""
1. **Select Number of Datasets**: Already done (you chose {num_datasets}).  
2. **Upload Datasets**: On the left sidebar, upload all {num_datasets} datasets.  
3. **Select Text Columns**: In the "Select Text Columns" tab, choose the columns from each dataset to combine into a single text field.  
4. **Run Fingerprint Matching**: In the "Run Fingerprint Matching" tab, select method (Document-level or Sentence-level) and run.  
5. **View Results**: In the "Results & Visualization" tab, see assigned clusters and a cluster frequency table across all datasets.  
6. **Optionally Merge Topics**: After initial clustering, select merges from the hierarchical structure.  
7. **Use Cluster Browser**: In the "Cluster Browser" tab, explore clusters, view associated docs/sentences.  
8. **Cosine Similarity**: In the "Cosine Similarity" tab, compare items across two datasets using the vector embeddings.  
    """)

# -------------------------------------------------------------------------------------------
# FAQ
# -------------------------------------------------------------------------------------------
with tab_faq:
    st.header("FAQ")
    st.markdown("""
**Q:** What data format is required?  
**A:** Excel (.xlsx).

**Q:** Document-level vs Sentence-level?  
**A:** Document-level clusters entire rows. Sentence-level clusters individual sentences.

**Q:** How to see overlaps?  
**A:** Check the frequency tables in "Results & Visualization" and the overlap table.
    """)

# -------------------------------------------------------------------------------------------
# Select Text Columns
# -------------------------------------------------------------------------------------------
with tab_select_text:
    st.header("Select Text Columns")
    if any([df is None for df in datasets]):
        st.warning("Please upload all selected datasets first.")
    else:
        combined_text_cols = []
        for i, df in enumerate(datasets):
            if df is not None:
                cols = df.columns.tolist()
                st.subheader(f"Dataset {i+1} Columns")
                selected_cols = st.multiselect(
                    f"Select columns to combine as text (Dataset {i+1})",
                    cols,
                    default=cols[:1] if cols else []
                )
                combined_text_cols.append(selected_cols)
            else:
                combined_text_cols.append([])

        if st.button("Combine Columns for All Datasets"):
            all_combined = True
            for i, df in enumerate(datasets):
                if df is None or not combined_text_cols[i]:
                    all_combined = False
                    break
            if all_combined:
                new_dfs = []
                for i, df in enumerate(datasets):
                    df['combined_text'] = combine_columns_into_text(df, combined_text_cols[i])
                    new_dfs.append(df)
                st.session_state['datasets'] = new_dfs
                st.success("Text columns combined successfully for all datasets!")
            else:
                st.warning("Please ensure all datasets are uploaded and columns selected.")

        if 'datasets' in st.session_state:
            for i, df in enumerate(st.session_state['datasets']):
                st.write(f"Dataset {i+1} Sample:")
                st.dataframe(df)

# -------------------------------------------------------------------------------------------
# Run Fingerprint Matching
# -------------------------------------------------------------------------------------------
with tab_run:
    st.header("Run Fingerprint Matching")
    if 'datasets' not in st.session_state:
        st.warning("Please upload and combine columns first.")
    else:
        method = st.selectbox("Select fingerprinting method", ["Document-level", "Sentence-level"])

        if st.button("Run Fingerprint Matching"):
            datasets_combined = st.session_state['datasets']

            # Compute embeddings for each dataset
            embeddings_list = []
            for i, df in enumerate(datasets_combined):
                emb = load_or_compute_embeddings(df, f"dataset{i}_{hash(tuple(df.columns))}")
                embeddings_list.append(emb)

            # Store embeddings for future usage
            st.session_state['embeddings_list'] = embeddings_list

            # Combine all texts
            all_texts = []
            dataset_sizes = []
            for df in datasets_combined:
                texts = df['combined_text'].tolist()
                all_texts.extend(texts)
                dataset_sizes.append(len(df))

            st.session_state['all_texts'] = all_texts
            st.session_state['dataset_sizes'] = dataset_sizes

            if method == "Document-level":
                all_emb = np.vstack(embeddings_list)
                topics, topic_model = cluster_documents(all_texts, all_emb)

                # Assign topics back to each dataset
                start = 0
                topic_assignments = []
                for size in dataset_sizes:
                    end = start + size
                    topic_assignments.append(topics[start:end])
                    start = end

                new_dfs = []
                for i, df in enumerate(datasets_combined):
                    df['Topic'] = topic_assignments[i]
                    new_dfs.append(df)

                st.session_state['datasets'] = new_dfs
                st.session_state['doc_topic_model'] = topic_model
                st.session_state['method'] = method
                st.session_state['original_doc_topics'] = topics
                st.session_state['hierarchical_topics_doc'] = topic_model.hierarchical_topics(all_texts)

                st.success("Document-level fingerprinting completed!")

            elif method == "Sentence-level":
                sentences, sentence_embeddings, topics, doc_indices, topic_model = cluster_sentences(all_texts)

                st.session_state['sentences'] = sentences
                st.session_state['sentence_topics'] = topics
                st.session_state['doc_indices'] = doc_indices
                st.session_state['sent_topic_model'] = topic_model
                st.session_state['method'] = method
                st.session_state['original_sent_topics'] = topics
                st.session_state['hierarchical_topics_sent'] = topic_model.hierarchical_topics(sentences)

                st.success("Sentence-level fingerprinting completed!")

# -------------------------------------------------------------------------------------------
# Results & Visualization
# -------------------------------------------------------------------------------------------
with tab_results:
    st.header("Results & Visualization")
    if 'method' not in st.session_state:
        st.warning("Please run fingerprint matching first.")
    else:
        method = st.session_state['method']
        dataset_sizes = st.session_state.get('dataset_sizes', [])

        if method == "Document-level":
            if 'doc_topic_model' in st.session_state and 'datasets' in st.session_state:
                topic_model = st.session_state['doc_topic_model']
                datasets_combined = st.session_state['datasets']

                # Show assigned topics
                st.subheader("Assigned Topics (Document-level)")
                for i, df in enumerate(datasets_combined):
                    st.write(f"Dataset {i+1}:")
                    st.dataframe(df[['combined_text', 'Topic']])

                st.subheader("Topic Visualization")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Compute frequency
                total_docs = sum(dataset_sizes)
                topics = np.concatenate([df['Topic'].values for df in datasets_combined])
                doc_indices = np.arange(total_docs)
                freq_df = compute_cluster_frequencies(
                    topics,
                    doc_indices,
                    dataset_sizes,
                    level='document'
                )

                st.subheader("Cluster Frequency (Document-level)")
                st.dataframe(freq_df)

                # Overlapping clusters
                st.subheader("Overlapping Clusters (Document-level)")
                overlap_df = freq_df[freq_df['In_multiple'] == True]
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

                    if st.button("Apply Merges (Doc-level)"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['all_texts'], merges_list)
                            # Reassign topics
                            start = 0
                            for i,df in enumerate(datasets_combined):
                                size = dataset_sizes[i]
                                df['Topic']=new_topics[start:start+size]
                                start += size
                            st.session_state['datasets']=datasets_combined
                            st.session_state['doc_topic_model']=new_model
                            st.success("Merges applied successfully! Please refresh or revisit the tab to see updated results.")

            else:
                st.warning("No document-level results found.")

        elif method == "Sentence-level":
            if ('sent_topic_model' in st.session_state and 
                'sentence_topics' in st.session_state and 
                'doc_indices' in st.session_state and
                'datasets' in st.session_state):
                
                topic_model = st.session_state['sent_topic_model']
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']

                # Show assigned topics
                st.subheader("Assigned Topics (Sentence-level)")
                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics
                })
                cumulative_sizes = np.cumsum(dataset_sizes)
                def find_dataset_idx(x):
                    return np.searchsorted(cumulative_sizes, x, side='right') + 1
                df_sent['dataset_id'] = df_sent['doc_index'].apply(find_dataset_idx)
                st.dataframe(df_sent)

                st.subheader("Topic Visualization (Sentence-level)")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization (Sentence-level)")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Exclude -1 cluster if you want
                df_sent_filtered = df_sent[df_sent['topic'] != -1]
                freq_df = compute_cluster_frequencies(
                    df_sent_filtered['topic'].values,
                    df_sent_filtered['doc_index'].values,
                    dataset_sizes,
                    level='sentence'
                )

                st.subheader("Overlapping Clusters (Sentence-level)")
                overlap_df = freq_df[freq_df['In_multiple'] == True]
                st.dataframe(overlap_df)

                # Stacked bar of frequencies
                count_cols = [c for c in freq_df.columns if c.endswith('_count')]
                long_data = []
                for i,row in freq_df.iterrows():
                    c = row['cluster']
                    for cc in count_cols:
                        long_data.append((c, cc, row[cc]))
                long_freq_df = pd.DataFrame(long_data, columns=['cluster','dataset_col','count'])
                long_freq_df['dataset'] = long_freq_df['dataset_col'].apply(lambda x: x.split('_')[0])

                fig_bar = px.bar(
                    long_freq_df,
                    x='cluster',
                    y='count',
                    color='dataset',
                    title="Sentence-level Cluster Frequency (Stacked by Dataset)",
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

                    if st.button("Apply Merges (Sentence-level)"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['sentences'], merges_list)
                            st.session_state['sentence_topics'] = new_topics
                            st.session_state['sent_topic_model'] = new_model
                            st.success("Merges applied successfully! Refresh or revisit the tab to see updated results.")

            else:
                st.warning("No sentence-level results found.")

        # Already existing "High-level Project Similarities" or other content can remain here
        # ...

# -------------------------------------------------------------------------------------------
# Cluster Browser
# -------------------------------------------------------------------------------------------
with tab_cluster_browser:
    st.header("Cluster Browser and Manual Review")

    if 'method' not in st.session_state:
        st.warning("Run fingerprint matching first.")
    else:
        method = st.session_state['method']
        if 'datasets' in st.session_state and 'dataset_sizes' in st.session_state:
            datasets_combined = st.session_state['datasets']
            dataset_sizes = st.session_state['dataset_sizes']

            if method == "Document-level" and 'doc_topic_model' in st.session_state:
                combined_df = []
                for i, df in enumerate(datasets_combined):
                    df_copy = df.copy()
                    df_copy['dataset_id'] = i+1
                    combined_df.append(df_copy)
                combined_df = pd.concat(combined_df, ignore_index=True)

                all_topics = sorted(combined_df['Topic'].unique())
                selected_clusters = st.multiselect("Select clusters to explore", all_topics)
                if selected_clusters:
                    subset = combined_df[combined_df['Topic'].isin(selected_clusters)]
                    for i in range(len(datasets_combined)):
                        sub = subset[subset['dataset_id']==i+1]
                        st.write(f"**Dataset {i+1} Documents in selected clusters:**")
                        st.dataframe(sub[['dataset_id','combined_text','Topic']])

            elif method == "Sentence-level" and 'sentences' in st.session_state and 'sentence_topics' in st.session_state and 'doc_indices' in st.session_state:
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']
                dataset_sizes = st.session_state['dataset_sizes']
                cumulative_sizes = np.cumsum(dataset_sizes)

                def find_dataset_idx(x):
                    return np.searchsorted(cumulative_sizes, x, side='right') + 1
                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics
                })
                df_sent['dataset_id'] = df_sent['doc_index'].apply(find_dataset_idx)

                freq_df = compute_cluster_frequencies(df_sent['topic'].values, df_sent['doc_index'].values, dataset_sizes, level='sentence')
                freq_df['Total_count'] = freq_df[[c for c in freq_df.columns if c.endswith('_count')]].sum(axis=1)

                st.subheader("Cluster Selection")
                available_clusters = freq_df['cluster'].sort_values()
                selected_cluster = st.selectbox("Select a cluster to view sentences", available_clusters)
                if selected_cluster is not None:
                    cluster_subset = df_sent[df_sent['topic'] == selected_cluster]
                    for i in range(len(datasets_combined)):
                        ds_sub = cluster_subset[cluster_subset['dataset_id']==i+1]
                        st.subheader(f"Dataset {i+1} Sentences in Cluster {selected_cluster}")
                        st.dataframe(ds_sub[['doc_index','sentence','topic']])

                    st.markdown("### Highlight a Single Document")
                    unique_docs = cluster_subset['doc_index'].unique()
                    selected_doc = st.selectbox("Select a document index to highlight its text", unique_docs)
                    if selected_doc is not None:
                        doc_sents = df_sent[df_sent['doc_index'] == selected_doc]
                        doc_sentences = doc_sents['sentence'].tolist()
                        doc_sent_topics = doc_sents['topic'].tolist()
                        highlighted_html = highlight_text_by_cluster(doc_sentences, doc_sent_topics, [selected_cluster])
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                        ds_id = doc_sents['dataset_id'].iloc[0]
                        st.write(f"Document index: {selected_doc} (Dataset {ds_id})")

            else:
                st.warning("No suitable data for cluster browser available.")
        else:
            st.warning("No datasets available.")

# -------------------------------------------------------------------------------------------
# NEW: Cosine Similarity Tab
# -------------------------------------------------------------------------------------------
with tab_cosine_sim:
    st.header("Cosine Similarity")

    st.markdown("""
    **Compute pairwise cosine similarity** between any two selected datasets.
    - Uses the same sentence-transformer embeddings (`all-MiniLM-L6-v2`).
    - Select which dataset is the "source" and which is the "target."
    - We'll produce a matrix/table where **rows = items from target dataset** and **columns = items from source dataset**, 
      showing the similarity scores.
    """)

    if 'datasets' not in st.session_state or 'embeddings_list' not in st.session_state:
        st.warning("Please upload datasets, combine columns, and run the embedding step at least once.")
    else:
        datasets_combined = st.session_state['datasets']
        embeddings_list = st.session_state['embeddings_list']

        # Let user pick which datasets to compare (if multiple)
        dataset_options = [f"Dataset {i+1}" for i in range(len(datasets_combined))]
        if len(dataset_options) < 2:
            st.warning("You need at least 2 datasets to perform Cosine Similarity.")
        else:
            source_ds_idx = st.selectbox("Select Source Dataset", range(len(datasets_combined)), 
                                         format_func=lambda x: dataset_options[x])
            target_ds_idx = st.selectbox("Select Target Dataset", range(len(datasets_combined)), 
                                         format_func=lambda x: dataset_options[x])

            if source_ds_idx == target_ds_idx:
                st.info("Select two different datasets for a meaningful comparison.")
            else:
                df_source = datasets_combined[source_ds_idx]
                df_target = datasets_combined[target_ds_idx]

                # Allow user to select columns for labeling source and target
                st.subheader("Select Label Columns for Source & Target Datasets")
                source_label_col = st.selectbox("Source Dataset Label Column", df_source.columns.tolist())
                target_label_col = st.selectbox("Target Dataset Label Column", df_target.columns.tolist())

                if st.button("Compute Cosine Similarity"):
                    # Get the embeddings
                    source_emb = embeddings_list[source_ds_idx]
                    target_emb = embeddings_list[target_ds_idx]

                    # shape: (len(target), len(source))
                    sim_matrix = cosine_similarity(target_emb, source_emb)

                    # Use user-selected columns for labeling
                    source_labels = df_source[source_label_col].astype(str).tolist()
                    target_labels = df_target[target_label_col].astype(str).tolist()

                    # Build a DataFrame where
                    # - index = target dataset rows (named from target_labels)
                    # - columns = source dataset rows (named from source_labels)
                    sim_df = pd.DataFrame(sim_matrix, index=target_labels, columns=source_labels)

                    st.markdown(f"**Similarity Matrix** (rows = {dataset_options[target_ds_idx]}, columns = {dataset_options[source_ds_idx]})")
                    st.dataframe(sim_df)

                    st.markdown("""
                    **Interpretation**:
                    - Each cell shows how similar an item (row) from the target dataset is to an item (column) in the source dataset.
                    - Values closer to 1.0 indicate higher similarity; closer to 0.0 indicate lower similarity.
                    """)

                    # Optionally, let user download the matrix
                    csv_data = sim_df.to_csv()
                    st.download_button(
                        "Download Similarity Matrix (CSV)",
                        data=csv_data,
                        file_name="similarity_matrix.csv",
                        mime="text/csv"
                    )
