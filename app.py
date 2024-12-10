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
    cumulative_sizes = np.cumsum(dataset_sizes)
    # For dataset i, the docs range from sum(dataset_sizes[:i]) to sum(dataset_sizes[:i+1])-1
    freq_data = []
    all_clusters = sorted(set(topics))

    # Precompute counts per dataset
    dataset_counts = []
    start = 0
    for size in dataset_sizes:
        end = start + size
        mask = (df['doc_index']>=start)&(df['doc_index']<end)
        counts = df[mask]['topic'].value_counts()
        dataset_counts.append(counts)
        start = end

    for c in all_clusters:
        row = {'cluster': c}
        # Count per dataset
        in_any = False
        for i, counts in enumerate(dataset_counts):
            cnt = counts.get(c,0)
            row[f'Dataset{i+1}_{level}_count'] = cnt
            if cnt > 0:
                in_any = True
        # Determine if cluster is in more than one dataset
        nonzero_counts = [row[k] for k in row.keys() if '_count' in k and row[k]>0]
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
            highlighted_sentences.append(f'<span style="background-color:{color}; padding:2px; border-radius:3px">{s}</span>')
        else:
            highlighted_sentences.append(s)
    return " ".join(highlighted_sentences)


tab_instructions, tab_select_text, tab_run, tab_results, tab_cluster_browser, tab_faq = st.tabs(
    ["Instructions", "Select Text Columns", "Run Fingerprint Matching", "Results & Visualization", "Cluster Browser", "FAQ"]
)

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
8. **Check Overlaps and Similarities**: Use overlap tables and compute multi-dataset similarities at a high level.
    """)

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
                selected_cols = st.multiselect(f"Select columns to combine as text (Dataset {i+1})", cols, default=cols[:1] if cols else [])
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

            # Store embeddings for future similarity computations
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
                # doc_indices = a simple range from 0 to sum(dataset_sizes)-1
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

                # Overlap (In_multiple) means cluster appears in >1 dataset
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

                    if st.button("Apply Merges"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['all_texts'], merges_list)
                            # Reassign topics
                            start=0
                            for i,df in enumerate(datasets_combined):
                                size = dataset_sizes[i]
                                df['Topic']=new_topics[start:start+size]
                                start+=size
                            st.session_state['datasets']=datasets_combined
                            st.session_state['doc_topic_model']=new_model
                            st.success("Merges applied successfully! Refresh the tab to see updated results.")

            else:
                st.warning("No document-level results found.")

        elif method == "Sentence-level":
            if ('sent_topic_model' in st.session_state and 
                'sentence_topics' in st.session_state and 'doc_indices' in st.session_state and
                'datasets' in st.session_state):
                
                topic_model = st.session_state['sent_topic_model']
                sentences = st.session_state['sentences']
                topics = st.session_state['sentence_topics']
                doc_indices = st.session_state['doc_indices']

                # Show assigned topics at sentence-level is tricky, but we just show a table
                st.subheader("Assigned Topics (Sentence-level)")
                df_sent = pd.DataFrame({
                    'doc_index': doc_indices,
                    'sentence': sentences,
                    'topic': topics
                })
                # Add dataset side info
                cumulative_sizes = np.cumsum(dataset_sizes)
                def find_dataset_idx(x):
                    return np.searchsorted(cumulative_sizes, x, side='right')
                df_sent['dataset_id'] = df_sent['doc_index'].apply(find_dataset_idx)
                st.dataframe(df_sent)

                st.subheader("Topic Visualization (Sentence-level)")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1)

                st.subheader("Hierarchical Topic Visualization")
                fig2 = topic_model.visualize_hierarchy()
                st.plotly_chart(fig2)

                # Exclude -1
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

                # Stacked bar chart of frequencies
                # Transform freq_df to long format
                count_cols = [c for c in freq_df.columns if c.endswith('_count')]
                long_data = []
                for i,row in freq_df.iterrows():
                    c=row['cluster']
                    if c == -1:
                        continue
                    for cc in count_cols:
                        long_data.append((c, cc, row[cc]))
                long_freq_df = pd.DataFrame(long_data, columns=['cluster','dataset_col','count'])
                # dataset_col format: DatasetX_sentence_count
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

                    if st.button("Apply Merges"):
                        if selected_merges:
                            merges_list = [merges_map[m] for m in selected_merges]
                            new_model, new_topics = topic_model.merge_topics(st.session_state['sentences'], merges_list)
                            st.session_state['sentence_topics'] = new_topics
                            st.session_state['sent_topic_model'] = new_model
                            st.success("Merges applied successfully! Refresh the tab to see updated results.")

            else:
                st.warning("No sentence-level results found.")

        # ADDITIONAL PROCESS: High-level Similarity among multiple datasets
        st.subheader("High-level Project Similarities")
        st.markdown("""
        Compute cosine similarities between each document in the first dataset and all documents in the other datasets.
        For each document in the first dataset:
        - Find the best match in each of the other datasets
        - Show those matches and their similarity scores
        - Compute an average similarity score across all other datasets
        """)

        # We need embeddings_list and datasets
        if 'embeddings_list' in st.session_state and 'datasets' in st.session_state:
            embeddings_list = st.session_state['embeddings_list']
            datasets_combined = st.session_state['datasets']
            if st.button("Compute High-level Project Similarities"):
                # We'll treat the first dataset as the "baseline"
                base_emb = embeddings_list[0]
                base_texts = datasets_combined[0]['combined_text'].tolist()

                # For each other dataset, compute similarity
                other_datasets_emb = embeddings_list[1:]
                other_datasets_texts = [d['combined_text'].tolist() for d in datasets_combined[1:]]

                results = []
                for i, lhs_vec in enumerate(base_emb):
                    # lhs_vec shape: (dim,)
                    # compute similarity to each other dataset
                    # We'll store best match from each dataset and their score
                    row = {
                        "Dataset1_Index": i,
                        "Dataset1_Text": base_texts[i]
                    }

                    scores_all = []
                    for j, emb in enumerate(other_datasets_emb):
                        sim = cosine_similarity(lhs_vec.reshape(1,-1), emb).flatten()
                        best_idx = sim.argmax()
                        best_score = sim.max()
                        # Store info
                        row[f"BestMatch_Dataset{j+2}_Index"] = best_idx
                        row[f"BestMatch_Dataset{j+2}_Text"] = other_datasets_texts[j][best_idx]
                        row[f"BestMatch_Dataset{j+2}_Score"] = best_score
                        scores_all.append(best_score)
                    # Compute average score
                    if scores_all:
                        row["Average_Score"] = np.mean(scores_all)
                    else:
                        row["Average_Score"] = np.nan

                    results.append(row)

                results_df = pd.DataFrame(results)
                st.dataframe(results_df)


with tab_cluster_browser:
    st.header("Cluster Browser and Manual Review")

    if 'method' not in st.session_state:
        st.warning("Run fingerprint matching first.")
    else:
        method = st.session_state['method']
        if 'datasets' in st.session_state and 'dataset_sizes' in st.session_state:
            datasets_combined = st.session_state['datasets']
            dataset_sizes = st.session_state['dataset_sizes']
            # For browsing clusters, we currently only implemented logic for 2 datasets scenario.
            # For multiple datasets, let's show a generic cluster browser for document-level:
            if method == "Document-level" and 'doc_topic_model' in st.session_state:
                # Combine all datasets
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
                cumulative_sizes = np.cumsum(dataset_sizes)

                def find_dataset_idx(x):
                    return np.searchsorted(cumulative_sizes, x, side='right')
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

                    # Optionally highlight text for a selected doc
                    st.markdown("### Highlight a Single Document")
                    unique_docs = cluster_subset['doc_index'].unique()
                    selected_doc = st.selectbox("Select a document index to highlight its text", unique_docs)
                    if selected_doc is not None:
                        doc_sents = cluster_subset[cluster_subset['doc_index']==selected_doc].sort_values('doc_index')
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
