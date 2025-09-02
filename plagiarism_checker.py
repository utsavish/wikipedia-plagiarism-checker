import streamlit as st
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

class UnifiedPlagiarismChecker:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.static_index = None
        self.dynamic_index = None
        self.static_corpus = None
        self.dynamic_corpus = None
        self.static_metadata = None
        self.dynamic_metadata = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def load_corpus_files(self, corpus_type: str = "static") -> bool:
        try:
            if corpus_type == "static":
                if os.path.exists("static_wiki_index.faiss"):
                    self.static_index = faiss.read_index("static_wiki_index.faiss")
                    self.static_corpus = np.load("static_wiki_texts.npy", allow_pickle=True)
                    if os.path.exists("static_corpus_metadata.json"):
                        with open("static_corpus_metadata.json", "r") as f:
                            self.static_metadata = json.load(f)
                    return True
            elif corpus_type == "dynamic":
                if os.path.exists("dynamic_wiki_index.faiss"):
                    self.dynamic_index = faiss.read_index("dynamic_wiki_index.faiss")
                    self.dynamic_corpus = np.load("dynamic_wiki_texts.npy", allow_pickle=True)
                    if os.path.exists("dynamic_corpus_metadata.json"):
                        with open("dynamic_corpus_metadata.json", "r") as f:
                            self.dynamic_metadata = json.load(f)
                    return True
            return False
        except Exception as e:
            st.error(f"Error loading {corpus_type} corpus: {str(e)}")
            return False

    def chunk_text(self, text: str, method: str = "sentence", chunk_size: int = 100) -> List[str]:
        if method == "sentence":
            return sent_tokenize(text)
        elif method == "paragraph":
            return [p.strip() for p in text.split('\n\n') if p.strip()]
        elif method == "character":
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size//2)]
        else:
            return [text]

    def semantic_similarity(self, query_chunks: List[str], corpus: np.ndarray, 
                          index: faiss.Index, threshold: float = 0.7) -> List[Dict]:
        if not query_chunks or index is None or corpus is None:
            return []
        matches = []
        query_embeddings = self.model.encode(query_chunks)
        for i, query_embedding in enumerate(query_embeddings):
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, min(10, len(corpus)))
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(corpus):
                    matches.append({
                        'query_chunk': query_chunks[i],
                        'matched_text': corpus[idx][:500] + "..." if len(corpus[idx]) > 500 else corpus[idx],
                        'similarity_score': float(score),
                        'chunk_index': i,
                        'corpus_index': int(idx),
                        'method': 'semantic'
                    })
        return matches

    def tfidf_similarity(self, query_chunks: List[str], corpus: np.ndarray, 
                        threshold: float = 0.3) -> List[Dict]:
        if not query_chunks or corpus is None:
            return []
        matches = []
        try:
            all_texts = list(query_chunks) + list(corpus)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            query_vectors = tfidf_matrix[:len(query_chunks)]
            corpus_vectors = tfidf_matrix[len(query_chunks):]
            similarities = cosine_similarity(query_vectors, corpus_vectors)
            for i, query_chunk in enumerate(query_chunks):
                for j, similarity in enumerate(similarities[i]):
                    if similarity >= threshold:
                        matches.append({
                            'query_chunk': query_chunk,
                            'matched_text': corpus[j][:500] + "..." if len(corpus[j]) > 500 else corpus[j],
                            'similarity_score': float(similarity),
                            'chunk_index': i,
                            'corpus_index': j,
                            'method': 'tfidf'
                        })
        except Exception as e:
            st.warning(f"TF-IDF similarity calculation failed: {str(e)}")
        return matches

    def fuzzy_similarity(self, query_chunks: List[str], corpus: np.ndarray, 
                        threshold: int = 70) -> List[Dict]:
        if not query_chunks or corpus is None:
            return []
        matches = []
        for i, query_chunk in enumerate(query_chunks):
            for j, corpus_text in enumerate(corpus):
                similarity = fuzz.partial_ratio(query_chunk.lower(), corpus_text.lower())
                if similarity >= threshold:
                    matches.append({
                        'query_chunk': query_chunk,
                        'matched_text': corpus_text[:500] + "..." if len(corpus_text) > 500 else corpus_text,
                        'similarity_score': similarity / 100.0,
                        'chunk_index': i,
                        'corpus_index': j,
                        'method': 'fuzzy'
                    })
        return matches

    def hybrid_plagiarism_detection(self, text: str, config: Dict) -> Dict:
        results = {
            'static_matches': [],
            'dynamic_matches': [],
            'combined_matches': [],
            'statistics': {},
            'config': config
        }
        query_chunks = self.chunk_text(text, config['chunking_method'], config['chunk_size'])
        if self.static_index is not None and self.static_corpus is not None:
            static_matches = []
            if config['use_semantic']:
                static_matches.extend(self.semantic_similarity(query_chunks, self.static_corpus, self.static_index, config['semantic_threshold']))
            if config['use_tfidf']:
                static_matches.extend(self.tfidf_similarity(query_chunks, self.static_corpus, config['tfidf_threshold']))
            if config['use_fuzzy']:
                static_matches.extend(self.fuzzy_similarity(query_chunks, self.static_corpus, int(config['fuzzy_threshold'] * 100)))
            results['static_matches'] = static_matches
        if self.dynamic_index is not None and self.dynamic_corpus is not None:
            dynamic_matches = []
            if config['use_semantic']:
                dynamic_matches.extend(self.semantic_similarity(query_chunks, self.dynamic_corpus, self.dynamic_index, config['semantic_threshold']))
            if config['use_tfidf']:
                dynamic_matches.extend(self.tfidf_similarity(query_chunks, self.dynamic_corpus, config['tfidf_threshold']))
            if config['use_fuzzy']:
                dynamic_matches.extend(self.fuzzy_similarity(query_chunks, self.dynamic_corpus, int(config['fuzzy_threshold'] * 100)))
            results['dynamic_matches'] = dynamic_matches
        all_matches = results['static_matches'] + results['dynamic_matches']
        unique_matches = []
        seen_combinations = set()
        for match in all_matches:
            key = (match['chunk_index'], match['method'], round(match['similarity_score'], 3))
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_matches.append(match)
        unique_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        results['combined_matches'] = unique_matches
        results['statistics'] = self.calculate_statistics(text, query_chunks, unique_matches)
        return results

    def calculate_statistics(self, original_text: str, chunks: List[str], matches: List[Dict]) -> Dict:
        total_chunks = len(chunks)
        flagged_chunks = len(set(match['chunk_index'] for match in matches))
        plagiarism_percentage = (flagged_chunks / total_chunks * 100) if total_chunks > 0 else 0
        method_counts = {}
        for match in matches:
            method_counts[match['method']] = method_counts.get(match['method'], 0) + 1
        high_similarity = sum(1 for match in matches if match['similarity_score'] >= 0.8)
        medium_similarity = sum(1 for match in matches if 0.5 <= match['similarity_score'] < 0.8)
        low_similarity = sum(1 for match in matches if match['similarity_score'] < 0.5)
        if plagiarism_percentage >= 50:
            risk_level = "HIGH"
        elif plagiarism_percentage >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        return {
            'total_chunks': total_chunks,
            'flagged_chunks': flagged_chunks,
            'plagiarism_percentage': round(plagiarism_percentage, 2),
            'total_matches': len(matches),
            'method_distribution': method_counts,
            'similarity_distribution': {
                'high': high_similarity,
                'medium': medium_similarity,
                'low': low_similarity
            },
            'risk_level': risk_level,
            'original_text_length': len(original_text),
            'average_similarity_score': round(np.mean([match['similarity_score'] for match in matches]), 3) if matches else 0
        }

    def create_visualizations(self, results: Dict) -> Tuple:
        stats = results['statistics']
        matches = results['combined_matches']
        fig_overview = go.Figure(data=[go.Pie(
            labels=['Flagged Content', 'Original Content'],
            values=[stats['flagged_chunks'], stats['total_chunks'] - stats['flagged_chunks']],
            hole=.3,
            marker_colors=['#ff6b6b', '#51cf66']
        )])
        fig_overview.update_layout(
            title=f"Content Analysis Overview - {stats['plagiarism_percentage']:.1f}% Potential Plagiarism",
            height=400
        )
        if stats['method_distribution']:
            methods = list(stats['method_distribution'].keys())
            counts = list(stats['method_distribution'].values())
            fig_methods = px.bar(
                x=methods, y=counts,
                title="Detection Method Distribution",
                labels={'x': 'Detection Method', 'y': 'Number of Matches'},
                color=counts,
                color_continuous_scale='Viridis'
            )
            fig_methods.update_layout(height=400)
        else:
            fig_methods = go.Figure()
            fig_methods.add_annotation(text="No matches found", xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)
        if matches:
            similarity_scores = [match['similarity_score'] for match in matches]
            fig_similarity = px.histogram(
                x=similarity_scores,
                nbins=20,
                title="Similarity Score Distribution",
                labels={'x': 'Similarity Score', 'y': 'Frequency'},
                color_discrete_sequence=['#36a2eb']
            )
            fig_similarity.update_layout(height=400)
        else:
            fig_similarity = go.Figure()
            fig_similarity.add_annotation(text="No similarity scores to display", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['plagiarism_percentage'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Plagiarism Risk Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        return fig_overview, fig_methods, fig_similarity, fig_gauge

    def generate_pdf_report(self, text: str, results: Dict) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Unified Plagiarism Detection Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Type:</b> Hybrid (Static + Dynamic Corpus)", styles['Normal']))
        story.append(Spacer(1, 20))
        stats = results['statistics']
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Text Length', f"{stats['original_text_length']:,} characters"],
            ['Text Chunks Analyzed', str(stats['total_chunks'])],
            ['Flagged Chunks', str(stats['flagged_chunks'])],
            ['Plagiarism Percentage', f"{stats['plagiarism_percentage']:.1f}%"],
            ['Risk Level', stats['risk_level']],
            ['Total Matches Found', str(stats['total_matches'])],
            ['Average Similarity Score', str(stats['average_similarity_score'])]
        ]
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        if stats['method_distribution']:
            story.append(Paragraph("Detection Method Distribution", styles['Heading2']))
            method_data = [['Method', 'Matches']]
            for method, count in stats['method_distribution'].items():
                method_data.append([method.capitalize(), str(count)])
            method_table = Table(method_data)
            method_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(method_table)
            story.append(Spacer(1, 20))
        matches = results['combined_matches'][:10]
        if matches:
            story.append(Paragraph("Top Similarity Matches", styles['Heading2']))
            for i, match in enumerate(matches, 1):
                story.append(Paragraph(f"<b>Match {i} ({match['method'].upper()}):</b>", styles['Normal']))
                story.append(Paragraph(f"<b>Similarity Score:</b> {match['similarity_score']:.3f}", styles['Normal']))
                story.append(Paragraph(f"<b>Query Text:</b> {match['query_chunk'][:200]}...", styles['Normal']))
                story.append(Paragraph(f"<b>Matched Text:</b> {match['matched_text'][:200]}...", styles['Normal']))
                story.append(Spacer(1, 15))
        config = results['config']
        story.append(Paragraph("Analysis Configuration", styles['Heading2']))
        config_data = [
            ['Parameter', 'Value'],
            ['Chunking Method', config.get('chunking_method', 'sentence')],
            ['Chunk Size', str(config.get('chunk_size', 100))],
            ['Semantic Analysis', 'Enabled' if config.get('use_semantic') else 'Disabled'],
            ['TF-IDF Analysis', 'Enabled' if config.get('use_tfidf') else 'Disabled'],
            ['Fuzzy Matching', 'Enabled' if config.get('use_fuzzy') else 'Disabled'],
            ['Semantic Threshold', str(config.get('semantic_threshold', 0.7))],
            ['TF-IDF Threshold', str(config.get('tfidf_threshold', 0.3))],
            ['Fuzzy Threshold', str(config.get('fuzzy_threshold', 0.7))]
        ]
        config_table = Table(config_data)
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(config_table)
        doc.build(story)
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Plagiarism Checker", layout="wide", initial_sidebar_state="expanded")
    st.title("🔍 Wikipedia Plagiarism Detection")
    st.markdown("**Advanced plagiarism detection with hybrid corpus analysis**")
    if 'checker' not in st.session_state:
        st.session_state.checker = UnifiedPlagiarismChecker()
    checker = st.session_state.checker
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Corpus Selection")
        use_static = st.checkbox("Use Static Corpus", value=True)
        use_dynamic = st.checkbox("Use Dynamic Corpus", value=False)
        static_loaded = False
        dynamic_loaded = False
        if use_static:
            static_loaded = checker.load_corpus_files("static")
            if static_loaded:
                st.success("✅ Static corpus loaded")
                if checker.static_metadata:
                    st.info(f"📊 Static corpus: {len(checker.static_corpus)} articles")
            else:
                st.error("❌ Static corpus not found. Please generate it first.")
        if use_dynamic:
            dynamic_loaded = checker.load_corpus_files("dynamic")
            if dynamic_loaded:
                st.success("✅ Dynamic corpus loaded")
                if checker.dynamic_metadata:
                    st.info(f"📊 Dynamic corpus: {len(checker.dynamic_corpus)} articles")
            else:
                st.warning("⚠️ Dynamic corpus not found. Generate during analysis or use static only.")
        st.subheader("Analysis Methods")
        use_semantic = st.checkbox("Semantic Similarity", value=True)
        use_tfidf = st.checkbox("TF-IDF Analysis", value=True)
        use_fuzzy = st.checkbox("Fuzzy Matching", value=True)
        st.subheader("Sensitivity Settings")
        semantic_threshold = st.slider("Semantic Threshold", 0.1, 1.0, 0.7, 0.05)
        tfidf_threshold = st.slider("TF-IDF Threshold", 0.1, 1.0, 0.3, 0.05)
        fuzzy_threshold = st.slider("Fuzzy Threshold", 0.1, 1.0, 0.7, 0.05)
        st.subheader("Text Processing")
        chunking_method = st.selectbox("Chunking Method", ["sentence", "paragraph", "character"], index=0)
        chunk_size = st.number_input("Chunk Size (for character method)", 50, 500, 100, 25)
    if not static_loaded and not dynamic_loaded:
        st.error("⚠️ No corpus loaded. Please ensure at least one corpus is available.")
        st.info("💡 Run the wiki_generator.py script to generate corpus files.")
        return
    st.header("📝 Text Analysis")
    text_input_method = st.radio("Choose input method:", ["Text Area", "File Upload"])
    user_text = ""
    if text_input_method == "Text Area":
        user_text = st.text_area("Enter text to analyze for plagiarism:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'doc', 'docx'])
        if uploaded_file:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", value=user_text, height=200, disabled=True)
    # Updated dynamic corpus generation block
    if user_text:
        st.info("💡 Dynamic corpus will be regenerated based on your current text for more targeted analysis.")

        if st.button("🚀 Generate / Regenerate Dynamic Corpus", type="secondary"):
            with st.spinner("Generating dynamic corpus... This may take a few minutes."):
                try:
                    from wiki_generator import UnifiedWikiGenerator
                    generator = UnifiedWikiGenerator()

                    # Delete old dynamic corpus files
                    for f in ["dynamic_wiki_index.faiss", "dynamic_wiki_texts.npy", "dynamic_corpus_metadata.json"]:
                        if os.path.exists(f):
                            os.remove(f)

                    # Generate new dynamic corpus
                    articles, metadata = generator.generate_dynamic_corpus(user_text)

                    if articles:
                        # Create FAISS index
                        index, embeddings = generator.create_faiss_index(articles)

                        # Save index, text, and metadata
                        faiss.write_index(index, "dynamic_wiki_index.faiss")
                        np.save("dynamic_wiki_texts.npy", np.array(articles, dtype=object))
                        with open("dynamic_corpus_metadata.json", "w") as f:
                            json.dump(metadata, f, indent=2)

                        st.success(f"✅ Dynamic corpus regenerated with {len(articles)} articles!")

                        # Set session flag so we only rerun once
                        st.session_state.dynamic_regenerated = True
                        st.rerun()
                    else:
                        st.warning("⚠️ Could not generate dynamic corpus. Using static corpus only.")
                except Exception as e:
                    st.error(f"Failed to generate dynamic corpus: {str(e)}")

    # Reset the flag so it doesn't trigger on other button clicks
    if "dynamic_regenerated" in st.session_state:
        st.session_state.dynamic_regenerated = False

    if user_text and st.button("🔍 Analyze for Plagiarism", type="primary"):
        config = {
            'chunking_method': chunking_method,
            'chunk_size': chunk_size,
            'use_semantic': use_semantic,
            'use_tfidf': use_tfidf,
            'use_fuzzy': use_fuzzy,
            'semantic_threshold': semantic_threshold,
            'tfidf_threshold': tfidf_threshold,
            'fuzzy_threshold': fuzzy_threshold
        }
        with st.spinner("Analyzing text for plagiarism..."):
            results = checker.hybrid_plagiarism_detection(user_text, config)
        st.header("📊 Analysis Results")
        stats = results['statistics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Plagiarism Risk", f"{stats['plagiarism_percentage']:.1f}%")
        with col2:
            st.metric("Total Matches", stats['total_matches'])
        with col3:
            st.metric("Flagged Chunks", f"{stats['flagged_chunks']}/{stats['total_chunks']}")
        with col4:
            st.metric("Risk Level", stats['risk_level'])
        if stats['risk_level'] == 'HIGH':
            st.error(f"🚨 HIGH RISK: {stats['plagiarism_percentage']:.1f}% potential plagiarism detected!")
        elif stats['risk_level'] == 'MEDIUM':
            st.warning(f"⚠️ MEDIUM RISK: {stats['plagiarism_percentage']:.1f}% potential plagiarism detected.")
        else:
            st.success(f"✅ LOW RISK: {stats['plagiarism_percentage']:.1f}% potential plagiarism detected.")
        if results['combined_matches']:
            st.header("📈 Analysis Visualizations")
            fig_overview, fig_methods, fig_similarity, fig_gauge = checker.create_visualizations(results)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_overview, use_container_width=True)
                st.plotly_chart(fig_similarity, use_container_width=True)
            with col2:
                st.plotly_chart(fig_methods, use_container_width=True)
                st.plotly_chart(fig_gauge, use_container_width=True)
        st.header("🔍 Detailed Match Analysis")
        tab1, tab2, tab3 = st.tabs(["🔄 Combined Results", "📚 Static Corpus", "🎯 Dynamic Corpus"])
        with tab1:
            matches = results['combined_matches']
            if matches:
                st.write(f"**Found {len(matches)} potential matches across all detection methods:**")
                for i, match in enumerate(matches[:20], 1):
                    with st.expander(f"Match {i} - {match['method'].upper()} "
                                   f"(Score: {match['similarity_score']:.3f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Your Text:**")
                            st.write(match['query_chunk'])
                        with col2:
                            st.write("**Matched Content:**")
                            st.write(match['matched_text'])
                        st.write(f"**Similarity Score:** {match['similarity_score']:.3f}")
                        st.write(f"**Detection Method:** {match['method'].capitalize()}")
            else:
                st.info("No potential plagiarism detected.")
        with tab2:
            static_matches = results.get('static_matches', [])
            if static_matches:
                st.write(f"**Static corpus matches: {len(static_matches)}**")
                for i, match in enumerate(static_matches[:10], 1):
                    with st.expander(f"Static Match {i} - {match['similarity_score']:.3f}"):
                        st.write(f"**Method:** {match['method']}")
                        st.write(f"**Query:** {match['query_chunk']}")
                        st.write(f"**Match:** {match['matched_text']}")
            else:
                st.info("No matches found in static corpus.")
        with tab3:
            dynamic_matches = results.get('dynamic_matches', [])
            if dynamic_matches:
                st.write(f"**Dynamic corpus matches: {len(dynamic_matches)}**")
                for i, match in enumerate(dynamic_matches[:10], 1):
                    with st.expander(f"Dynamic Match {i} - {match['similarity_score']:.3f}"):
                        st.write(f"**Method:** {match['method']}")
                        st.write(f"**Query:** {match['query_chunk']}")
                        st.write(f"**Match:** {match['matched_text']}")
            else:
                st.info("No matches found in dynamic corpus or dynamic corpus not loaded.")
        st.header("💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_data,
                file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col2:
            pdf_buffer = checker.generate_pdf_report(user_text, results)
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_buffer,
                file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    st.markdown("<hr><p style='text-align: center;'>Made with ❤️ by Utsav Hadiya</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
