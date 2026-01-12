# Wikipedia Plagiarism Checker

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
[![License](https://img.shields.io/badge/License-GNU-green)](LICENSE.txt)

The **Wikipedia Plagiarism Checker** is an advanced plagiarism detection tool that specializes in identifying similarities between user-provided text and Wikipedia content.  
Unlike traditional tools that only detect direct string matches, this system is designed to catch **paraphrasing, rewording, and semantically similar content**.  

It combines multiple detection methods:  
- **Semantic similarity** using Sentence Transformers.  
- **TF-IDF with cosine similarity** for statistical comparison.  
- **Fuzzy string matching** for approximate text overlaps.  

The system supports two powerful analysis modes:  
- A **static Wikipedia corpus** covering major domains such as science, literature, history, and technology.  
- A **dynamic Wikipedia corpus** generated on-the-fly, tailored to the user’s input text.  

With its **interactive Streamlit dashboard**, users can:  
- Upload or paste text for plagiarism analysis.  
- Tune chunking, thresholds, and detection strategies.  
- View plagiarism risk via charts, gauges, and visual breakdowns.  
- Export findings as **JSON reports** or **PDF summaries** for documentation.  

This makes the tool especially useful for:  
- **Students** ensuring originality of their work.  
- **Teachers & professors** detecting academic misconduct.  
- **Researchers** verifying proper citation practices.  
- **Content creators & professionals** validating authenticity.    

## 🚀 Try It
Run the app locally with Streamlit:
```bash
streamlit run plagiarism_checker.py
```

---

## ✨ Features
- **Hybrid Corpus Analysis**  
  - Static Wikipedia corpus (pre-built).  
  - Dynamic corpus (generated on-the-fly from user text).  
- **Multiple Detection Methods**  
  - Semantic similarity (Sentence Transformers).  
  - TF-IDF + cosine similarity.  
  - Fuzzy string matching.  
- **Interactive Streamlit Dashboard**  
  - Upload text or paste content for analysis.  
  - Configure thresholds & chunking methods.  
  - Visualizations: Pie charts, histograms, bar charts, gauges.  
- **Report Export**  
  - JSON report with full details.  
  - PDF report with summary, stats, and top matches.  

---

## 📂 Repository Structure
```
.
├── plagiarism_checker.py   # Main Streamlit app
├── wiki_generator.py       # Wikipedia-based corpus generator
├── requirements.txt        # Dependencies
├── static_wiki_index.faiss # Generated FAISS index (static corpus)
├── static_wiki_texts.npy   # Wikipedia articles (static corpus)
├── static_corpus_metadata.json
└── (dynamic corpus files created at runtime)
```

---

## ⚙️ Installation
You need to have python version **3.13** to run this project, please install it before moving ahead.

### 1. Clone the repository
```bash
git clone https://github.com/utsavish/wikipedia-plagiarism-checker
cd wikipedia-plagiarism-checker
```

### 2. Make a virtual enviroment and activate it
For Windows users
```powershell
python -3.13 -m venv venv
.\venv\Scripts\Activate
```
For Linux and Mac users
```bash
python3.13 -m venv venv
source ./venv/bin/Activate
```


### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run plagiarism_checker.py
```

---

## 🛠️ Usage
- Paste text or upload `.txt / .doc / .docx` files.  
- Configure detection methods (Semantic, TF-IDF, Fuzzy).  
- Adjust thresholds and chunking strategies.  
- Click **🔍 Analyze for Plagiarism**.  
- Download detailed **JSON** or **PDF** reports.  

---

## 📊 Example Outputs
- **Risk Level:** HIGH / MEDIUM / LOW  
- **Plagiarism %:** % of flagged chunks.  
- **Method Distribution:** How many matches came from each method.  
- **Top Matches:** Extracted sentences with highest similarity.  

---

## 🧩 Generating Corpora
You can generate or regenerate corpora with `wiki_generator.py`:

```bash
python wiki_generator.py
```

- **Static corpus:** Wikipedia articles from predefined categories.  
- **Dynamic corpus:** Generated from keywords in user text.  

---

## 🤝 Contributing
Contributions are welcome!  
- Bug fixes, new features, and optimizations are appreciated.  
- Feel free to open issues or submit pull requests.  

---

## 📜 License
Distributed under the GNU General Public License (GPL). See [LICENSE](LICENSE.txt) for more details.
