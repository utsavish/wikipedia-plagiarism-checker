import bs4
from collections import Counter
from datetime import datetime
import faiss
import json
import logging
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import time
from typing import List, Dict, Tuple, Optional
import wikipedia


# Monkey-patch BeautifulSoup to always use 'html.parser' if not specified
_original_bs4_init = bs4.BeautifulSoup.__init__
def _patched_bs4_init(self, markup="", features=None, builder=None, **kwargs):
    if features is None:
        features = "html.parser"
    _original_bs4_init(self, markup, features, builder, **kwargs)
bs4.BeautifulSoup.__init__ = _patched_bs4_init

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedWikiGenerator:
    def __init__(self):
        #Imported miniLM model for symentic anylysis
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = None
        self.setup_nlp()

        self.static_categories = {
            "Technology": [
                "Artificial Intelligence", "Machine learning", "Deep learning", "Neural Network",
                "Computer Science", "Programming", "Software Engineering", "Data Science",
                "Cybersecurity", "Internet", "World Wide Web", "Cloud Computing", "Blockchain",
                "Cryptocurrency", "Virtual Reality", "Augmented Reality", "Internet of Things",
                "Big Data", "Quantum Computing", "Robotics"
            ],
            "Science": [
                "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy", "Genetics",
                "Evolution", "Climate change", "Renewable energy", "Nuclear physics",
                "Quantum mechanics", "Relativity", "DNA", "Cell biology", "Ecology",
                "Biochemistry", "Molecular biology", "Neuroscience", "Psychology", "Medicine"
            ],
            "Literature": [
                "William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain",
                "Ernest Hemingway", "Virginia Woolf", "George Orwell", "J.K. Rowling",
                "Agatha Christie", "Leo Tolstoy", "Fyodor Dostoevsky", "Homer", "Dante Alighieri",
                "Miguel de Cervantes", "Gabriel García Márquez", "Toni Morrison", "Maya Angelou"
            ],
            "History": [
                "World War II", "Renaissance", "Industrial Revolution", "French Revolution",
                "American Civil War", "Roman Empire", "Ancient Egypt", "Medieval Europe",
                "Cold War", "World War I", "Great Depression", "Age of Exploration",
                "Ancient Greece", "British Empire", "Ottoman Empire", "Ming Dynasty"
            ],
            "Philosophy": [
                "Plato", "Aristotle", "Socrates", "Immanuel Kant", "Friedrich Nietzsche",
                "Jean-Paul Sartre", "Existentialism", "Stoicism", "Ethics", "Metaphysics",
                "Epistemology", "Logic", "Political philosophy", "Philosophy of mind"
            ],
            "Business": [
                "Economics", "Marketing", "Management", "Entrepreneurship", "Finance",
                "Accounting", "Supply chain", "Human resources", "Strategic planning",
                "Corporate governance", "Business ethics", "Innovation", "Leadership"
            ]
        }

    #For name, places, persons etc (NER)
    def setup_nlp(self):
        """Setup spaCy NLP model for keyword extraction"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy 'en_core_web_sm' model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from user text using multiple methods"""
        keywords = set()
        
        # Method 1: TF-IDF
        try:
            tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords from TF-IDF
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            for keyword, score in keyword_scores[:15]:
                if len(keyword.strip()) > 2 and score > 0:
                    keywords.add(keyword.strip())
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")

        # Method 2: Named Entity Recognition
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                        if len(ent.text.strip()) > 2:
                            keywords.add(ent.text.strip())
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")

        # Method 3: Capitalized words (proper nouns)
        words = text.split()
        for word in words:
            cleaned = ''.join(c for c in word if c.isalnum())
            if len(cleaned) > 2 and cleaned[0].isupper() and cleaned.lower() not in ['the', 'and', 'but', 'for']:
                keywords.add(cleaned)

        # Method 4: Frequency analysis
        word_freq = Counter(word.lower().strip('.,!?;:"()[]') for word in words
                            if len(word) > 3 and word.isalpha())
        for word, freq in word_freq.most_common(10):
            if freq > 1:
                keywords.add(word.capitalize())
        
        return list(keywords)[:max_keywords]

    def fetch_wikipedia_content(self, title: str, max_retries: int = 3) -> Optional[str]:
        """Fetch Wikipedia article content with retries and robust fallbacks."""
        for attempt in range(max_retries):
            try:
                time.sleep(0.5)  # be polite
                # 1) Try exact title without auto_suggest to avoid wrong guesses
                page = wikipedia.page(title, auto_suggest=False, redirect=True, preload=False)
                return page.content

            except wikipedia.exceptions.DisambiguationError as e:
                # 2) If ambiguous, try the first few options deterministically
                for opt in e.options[:5]:
                    try:
                        page = wikipedia.page(opt, auto_suggest=False, redirect=True, preload=False)
                        return page.content
                    except Exception:
                        continue
                # If none worked, optionally perform a search and try first result
                try:
                    results = wikipedia.search(title, results=3)
                    for r in results:
                        try:
                            page = wikipedia.page(r, auto_suggest=False, redirect=True, preload=False)
                            return page.content
                        except Exception:
                            continue
                except Exception:
                    pass

            except wikipedia.exceptions.PageError:
                # 3) Fall back to search + auto_suggest on a top result
                try:
                    results = wikipedia.search(title, results=3)
                    for r in results:
                        try:
                            page = wikipedia.page(r, auto_suggest=False, redirect=True, preload=False)
                            return page.content
                        except wikipedia.exceptions.PageError:
                            continue
                    # last resort: allow auto_suggest to help
                    page = wikipedia.page(title, auto_suggest=True, redirect=True, preload=False)
                    return page.content
                except Exception:
                    logger.warning(f"Page not found after fallbacks: {title}")
                    return None

            except Exception as e:
                logger.warning(f"Error fetching {title} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None

        return None


    def search_wikipedia_articles(self, keywords: List[str], max_articles: int = 30) -> List[str]:
        """Search for Wikipedia articles based on keywords"""
        articles = []
        titles_found = set()
        for keyword in keywords:
            try:
                # Search for articles related to keyword
                search_results = wikipedia.search(keyword, results=5)
                for title in search_results:
                    if title not in titles_found and len(articles) < max_articles:
                        content = self.fetch_wikipedia_content(title)
                        if content and len(content) > 500: # Minimum content length
                            articles.append(content)
                            titles_found.add(title)
                            logger.info(f"Added article: {title}")
                    if len(articles) >= max_articles:
                        break
            except Exception as e:
                logger.warning(f"Search failed for keyword '{keyword}': {e}")
                continue
        return articles

    def generate_static_corpus(self) -> Tuple[List[str], Dict]:
        """Generate comprehensive static corpus"""
        logger.info("Generating static Wikipedia corpus...")
        corpus = []
        metadata = {"articles": [], "categories": {}, "generation_time": datetime.now().isoformat()}
        
        for category, titles in self.static_categories.items():
            category_articles = []
            logger.info(f"Processing category: {category}")
            for title in titles:
                content = self.fetch_wikipedia_content(title)
                if content:
                    corpus.append(content)
                    category_articles.append(title)
                    metadata["articles"].append({
                        "title": title,
                        "category": category,
                        "length": len(content),
                        "index": len(corpus) - 1
                    })
            metadata["categories"][category] = {
                "articles": category_articles,
                "count": len(category_articles)
            }
        
        logger.info(f"Static corpus generated: {len(corpus)} articles")
        return corpus, metadata

    def generate_dynamic_corpus(self, user_text: str) -> Tuple[List[str], Dict]:
        """Generate dynamic corpus based on user text"""
        logger.info("Generating dynamic Wikipedia corpus based on user content...")
        
        # Extract keywords from user text
        keywords = self.extract_keywords(user_text)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Search for relevant articles
        articles = self.search_wikipedia_articles(keywords, max_articles=50)
        
        metadata = {
            "keywords_used": keywords,
            "articles_found": len(articles),
            "generation_time": datetime.now().isoformat(),
            "user_text_length": len(user_text)
        }
        
        logger.info(f"Dynamic corpus generated: {len(articles)} articles")
        return articles, metadata

    def create_faiss_index(self, corpus: List[str]) -> Tuple[faiss.IndexIVFFlat, np.ndarray]:
        """Create FAISS index from corpus"""
        logger.info("Creating FAISS index...")
        
        # Generate embeddings
        embeddings = self.model.encode(corpus, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create index
        dimension = embeddings.shape[1]
        nlist = min(max(int(np.sqrt(len(corpus))), 1), len(corpus))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        faiss.normalize_L2(embeddings)
        index.train(embeddings)
        index.add(embeddings)
        
        logger.info(f"FAISS index created with {len(corpus)} vectors")
        return index, embeddings

    def generate_hybrid_corpus(self, user_text: str = None, use_dynamic: bool = True) -> Dict:
        """Generate hybrid corpus combining static and dynamic approaches"""
        results = {"static": None, "dynamic": None, "hybrid_metadata": {}}
        
        # Always generate static corpus
        static_corpus, static_metadata = self.generate_static_corpus()
        static_index, static_embeddings = self.create_faiss_index(static_corpus)
        
        # Save static corpus
        faiss.write_index(static_index, "static_wiki_index.faiss")
        np.save("static_wiki_texts.npy", np.array(static_corpus, dtype=object))
        with open("static_corpus_metadata.json", "w") as f:
            json.dump(static_metadata, f, indent=2)
            
        results["static"] = {
            "corpus_size": len(static_corpus),
            "metadata": static_metadata,
            "files": ["static_wiki_index.faiss", "static_wiki_texts.npy", "static_corpus_metadata.json"]
        }
        
        # Generate dynamic corpus if user text provided
        if user_text and use_dynamic:
            try:
                dynamic_corpus, dynamic_metadata = self.generate_dynamic_corpus(user_text)
                if dynamic_corpus:
                    dynamic_index, dynamic_embeddings = self.create_faiss_index(dynamic_corpus)
                    
                    # Save dynamic corpus
                    faiss.write_index(dynamic_index, "dynamic_wiki_index.faiss")
                    np.save("dynamic_wiki_texts.npy", np.array(dynamic_corpus, dtype=object))
                    with open("dynamic_corpus_metadata.json", "w") as f:
                        json.dump(dynamic_metadata, f, indent=2)
                        
                    results["dynamic"] = {
                        "corpus_size": len(dynamic_corpus),
                        "metadata": dynamic_metadata,
                        "files": ["dynamic_wiki_index.faiss", "dynamic_wiki_texts.npy", "dynamic_corpus_metadata.json"]
                    }
            except Exception as e:
                logger.error(f"Dynamic corpus generation failed: {e}")
                results["dynamic"] = {"error": str(e)}

        # Create hybrid metadata
        dynamic_size = 0
        if results.get("dynamic") and "corpus_size" in results["dynamic"]:
            dynamic_size = results["dynamic"]["corpus_size"]

        results["hybrid_metadata"] = {
            "generation_time": datetime.now().isoformat(),
            "static_articles": len(static_corpus),
            "dynamic_articles": dynamic_size,
            "total_articles": len(static_corpus) + dynamic_size
        }
        
        return results

if __name__ == "__main__":
    generator = UnifiedWikiGenerator()
    
    # Example usage
    sample_text = """
    Artificial intelligence has revolutionized modern technology. Machine learning algorithms
    are now being used in various applications from natural language processing to computer vision.
    Deep learning neural networks have shown remarkable performance in tasks like image recognition
    and speech synthesis.
    """
    
    results = generator.generate_hybrid_corpus(sample_text, use_dynamic=True)
    
    print("Corpus Generation Complete!")
    print(f"Static corpus: {results['static']['corpus_size']} articles")
    if results.get('dynamic') and 'error' not in results['dynamic']:
        print(f"Dynamic corpus: {results['dynamic']['corpus_size']} articles")
    
    print(f"Total articles: {results['hybrid_metadata']['total_articles']}")
