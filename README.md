# 🏢 Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Score](https://img.shields.io/badge/Macro%20F0.5-0.622-orange.svg?style=for-the-badge)](https://unstop.com)
[![Rank](https://img.shields.io/badge/Rank-Top%201%2C000%2B%20%2F%2089%2C399-purple.svg?style=for-the-badge)](https://unstop.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **An enterprise-grade, high-throughput Machine Learning pipeline for multi-source Business Entity Resolution (ER). Resolves, links, and deduplicates 1.7M+ noisy business records across three independent data sources ($S_1, S_2, S_3$).**

---

## 📌 Project Overview & Results

In large-scale commercial platforms, business identity data arrives from multiple independent sources with severe noise: legal suffix variations (*LLC*, *Pvt Ltd*, *SARL*), missing address fields, landmark-based descriptors (*"Near SBI ATM"*), and multilingual transliterations. 

This repository contains the complete Machine Learning system built for the **Amazon ML Challenge 2026**, achieving top-tier leaderboard performance evaluated on the precision-dominant **Macro $F_{0.5}$ metric**.

### 🏆 Key Performance Metrics
- **Final Macro $F_{0.5}$ Score:** **`0.622`**
- **Precision:** `78%` | **Recall:** `65%`
- **Global Placement:** **Top 1,000+** out of **89,399 participating teams**
- **Test Entities Processed:** `1,732,544` records ($100\%$ covered)
- **Matches Predicted:** `6,195,472` links across `1,596,333` entities ($92\%$)
- **Singletons Correctly Identified:** `136,211` non-matching entities ($8\%$)

---

## 🏗️ System Architecture

```
Raw Multi-Source Data (S1, S2, S3)
  ├── 1.7M+ Records across US, India, France
  └── Unstructured text: business_name, business_address
        │
        ▼
[ 1. Ingestion & Preprocessing ]
   ├── Unicode & Accent Normalization (strip_accents)
   ├── Legal Suffix Standardization (LLC, Pvt Ltd, Inc, SARL, SASU)
   └── Lossless Country Partitioning (US, India, France)
        │
        ▼
[ 2. Multi-Key Inverted Index Blocking Engine ]
   ├── Exact Clean Name (`NE`) & Token Permutations (`N1`, `N2`, `N12`, `NS2`)
   ├── Phonetic Consonant Skeletonization (`NSK`)
   ├── Postal / PIN Code (`PIN`) & High-Value Digit Anchors (`NUM4`, `AN`)
   └── Cross Name-Prefix & Street Number Keys (`W1_NUM`, `SK1_NUM`)
   └── ⚡ Candidate Search Reduction > 99.999% (100x speedup, >99.4% recall)
        │
        ▼
[ 3. 24-Dimensional Feature Engineering ]
   ├── Name Lexical & Word-Order Features (9)
   ├── Address & Geographic Similarities (8)
   ├── Numeric & PIN Conflict Resolution (4)
   └── Source Metadata & Indic Transliteration Flags (3)
        │
        ▼
[ 4. Class-Weighted LightGBM Classifier ]
   ├── 150 Estimators | Max Depth: 8 | Num Leaves: 31 | lr: 0.08
   └── Decision Threshold Optimization (τ = 0.65)
        │
        ▼
[ 5. High-Throughput Parallel Inference Engine (16 Workers) ]
   ├── Batch Streaming (< 3.5 GB RAM footprint)
   ├── candidate_pairs.tsv (Blocking Candidates)
   └── matching_results.tsv (Final High-Confidence Matches)
```

---

## 🧠 24 Engineered Features Breakdown

The system relies on 24 domain-crafted pairwise features rather than opaque deep models to deliver maximum precision and high runtime efficiency:

| # | Feature Name | Category | Description & Business Intuition |
|---|:---|:---|:---|
| **1** | `name_ratio` | Name Similarity | Normalized Levenshtein similarity on cleaned business names |
| **2** | `name_token_sort_ratio` | Name Similarity | Word-order invariant Levenshtein ratio (handles transposed tokens) |
| **3** | `name_token_set_ratio` | Name Similarity | Token set overlap ratio (handles subset names & added keywords) |
| **4** | `name_partial_ratio` | Name Similarity | Substring match ratio for truncated or prefixed enterprise names |
| **5** | `name_word_jaccard` | Name Similarity | Word-level token Jaccard similarity coefficient |
| **6** | `name_char3_jaccard` | Name Similarity | Character tri-gram Jaccard overlap (robust to typos and misspellings) |
| **7** | `name_exact_match` | Name Similarity | Binary flag (1.0/0.0) indicating exact normalized name equivalence |
| **8** | `name_len_diff_ratio` | Name Similarity | Relative character length disparity between compared names |
| **9** | `name_prefix_match` | Name Similarity | Indicator for exact 4+ character prefix alignment |
| **10** | `addr_ratio` | Address Similarity | Standardized address Levenshtein edit distance ratio |
| **11** | `addr_token_sort_ratio` | Address Similarity | Token sort ratio on address components (street, city, state reordering) |
| **12** | `addr_token_set_ratio` | Address Similarity | Shared token set ratio across address strings |
| **13** | `addr_partial_ratio` | Address Similarity | Substring matching score for truncated street addresses |
| **14** | `addr_word_jaccard` | Address Similarity | Full token Jaccard overlap on address components |
| **15** | `addr_distinctive_jaccard`| Address Similarity | Jaccard similarity on non-stopword distinctive address tokens |
| **16** | `addr_char3_jaccard` | Address Similarity | Character 3-gram address overlap |
| **17** | `addr_exact_match` | Address Similarity | Binary exact match flag on normalized address |
| **18** | `num_jaccard` | Numeric & PIN | Jaccard similarity of extracted numeric digit tokens |
| **19** | `num_common_count` | Numeric & PIN | Total count of shared numeric tokens (plot, sector, PIN) |
| **20** | `num_conflict_flag` | Numeric & PIN | Strict penalty indicator: numbers present in both but zero match |
| **21** | `pin_match` | Numeric & PIN | Tri-state postal code check (+1.0 match, -1.0 conflict, 0.0 missing) |
| **22** | `is_source2` | Metadata & Heuristics | Binary flag distinguishing Source 2 vs Source 3 targets |
| **23** | `indic_target_flag` | Metadata & Heuristics | Non-ASCII / Indic script presence flag for transliteration handling |
| **24** | `combined_heuristic` | Metadata & Heuristics | Weighted baseline heuristic (0.55 * name_tset + 0.45 * addr_tset) |

---

## ⚙️ Model Architecture & Hyperparameters

We utilize a tuned **LightGBM (Light Gradient Boosting Machine)** binary classifier. The objective is to score candidate pairs generated by the blocking stage.

```python
LGBMClassifier(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,     # Balanced loss weighting
    random_state=42,
    n_jobs=-1
)
```

### 🎯 Threshold Calibration ($\tau = 0.65$)
- **Evaluation Metric:** $\text{Macro } F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$
- **Why $\tau = 0.65$?** $F_{0.5}$ penalizes false merges (False Positives) $2\times$ more severely than missed links (False Negatives). In enterprise Master Data Management (MDM), merging two distinct businesses corrupts customer records. Shifting the decision boundary from default $0.50$ to **$0.65$** delivers an immediate $+0.10$ precision lift.

---

## 📁 Repository Structure

```
amazon-ml-challenge-2026/
├── src/
│   ├── __init__.py           # Package exports and version metadata
│   ├── main.py               # Main CLI entry point (train, inference, full)
│   ├── features.py           # Text cleaning and 24-feature extraction pipeline
│   ├── blocking.py           # Multi-key inverted index blocking engine
│   ├── model.py              # LightGBM classifier wrapper & Macro F_0.5 metrics
│   └── inference.py          # Parallel 16-worker partitioned inference pipeline
├── output/
│   ├── README.md             # Output specifications
│   ├── matching_results.tsv  # Final predicted entity matches (1.73M rows)
│   └── candidate_pairs.tsv   # Blocking candidate set
├── requirements.txt          # Pinned package dependencies
├── README.md                 # Primary project documentation
└── Documentation.md          # Comprehensive technical whitepaper
```

---

## 🚀 Quick Start & How to Run

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model
```bash
python src/main.py --mode train --train-dir dataset/train --model-path models/model.joblib --threshold 0.65
```

### 3. Run Inference on Test Set
```bash
python src/main.py --mode inference --test-dir dataset/test --output-dir output/ --model-path models/model.joblib --workers 16
```

### 4. Validate Output Files
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 🔧 Technologies & Libraries

| Component | Technology | Purpose |
|:---|:---|:---|
| **Programming Language** | Python 3.8+ | Core development stack |
| **Gradient Boosting** | LightGBM 4.0+ | Fast tabular decision tree classifier |
| **String & Metric Algorithms**| RapidFuzz, Fuzzywuzzy | Fast C-accelerated Levenshtein & token metrics |
| **Data Manipulation** | Pandas, NumPy | Streaming batch operations & array processing |
| **ML Framework** | Scikit-learn | Data partitioning, metrics, and cross-validation |
| **Model Serialization** | Joblib | High-efficiency tree model persistence |
| **Parallel Execution** | ThreadPoolExecutor | Multi-threaded candidate scoring (16 workers) |

---

## 📊 Summary of Insights & Key Learnings

1. **Blocking Determines Recall Ceiling:** Trillions of potential comparisons are filtered into high-probability candidates in under 15 minutes with $>99.4\%$ recall.
2. **Domain-Specific Signals Beat Model Complexity:** 24 handcrafted lexical, spatial, and postal features out-perform large transformer models while executing $50\times$ faster on CPUs.
3. **Metric-Aligned Decision Thresholding:** Calibrating the decision boundary directly against Macro $F_{0.5}$ optimizes precision without sacrificing viable links.
4. **Singletons Are Crucial:** Correctly predicting empty strings on singletons (8% of dataset) guarantees a full 1.0 macro score per record.

---

## 👨‍💻 Author

**Nandala Dheekshitha**  
- **Institution:** Institute of Aeronautical Engineering (IARE), Hyderabad  
- **Discipline:** Computer Science & Engineering / Information Technology  
- **Focus Areas:** Machine Learning, Full-Stack Development, Data Structures & Algorithms  
- **LinkedIn:** [linkedin.com/in/your-profile](#)  
- **GitHub:** [github.com/your-username](#)

---

## 📜 License

This project is released under the [MIT License](LICENSE). All code and documentation are original implementations created for the Amazon ML Challenge 2026.
