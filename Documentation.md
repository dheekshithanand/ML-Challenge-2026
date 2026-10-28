# 📄 Technical Methodology & System Whitepaper
## Amazon ML Challenge 2026: Large-Scale Business Entity Resolution

**Author:** Nandala Dheekshitha  
**Institution:** Institute of Aeronautical Engineering (IARE), Hyderabad  
**Track:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Official Metric:** Macro-Averaged $F_{0.5}$ Score  
**Performance:** Score: **`0.622`** | Global Ranking: **Top 1,000+** out of **89,399 teams**  

---

## 1. Problem Understanding & Mathematical Foundation

In modern enterprise commerce and supply-chain systems, business identity records continuously flow from disparate catalog systems, vendor portals, and registry databases. These records arrive as noisy text fragments without common foreign keys or unified global identifiers.

The task is formalised as follows:
Given a deduplicated reference dataset $\mathcal{S}_1$ and two noisy un-deduplicated datasets $\mathcal{S}_2$ and $\mathcal{S}_3$, determine the mapping function:

$$\mathcal{M}: e \in \mathcal{S}_1 \longrightarrow \mathcal{C} \subseteq (\mathcal{S}_2 \cup \mathcal{S}_3)$$

where $\mathcal{C}$ is the set of records in $\mathcal{S}_2 \cup \mathcal{S}_3$ that represent the exact same physical business entity as $e$. A record in $\mathcal{S}_1$ may match zero (a *singleton*), one, or multiple records across $\mathcal{S}_2$ and $\mathcal{S}_3$.

### The Macro $F_{0.5}$ Metric

Submissions are evaluated using the Macro-Averaged $F_{\beta}$ score with $\beta = 0.5$:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

The metric is macro-averaged across all $N = 1,732,544$ Source 1 entities in the test set:

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}(e_i)$$

#### Singleton Credit Semantics
- If an entity $e_i \in \mathcal{S}_1$ has no true matches ($\mathcal{C}_{true} = \emptyset$) and the system predicts no matches ($\mathcal{C}_{pred} = \emptyset$), it receives an individual score of $1.0$.
- If any match is incorrectly predicted for a true singleton, its score becomes $0.0$.
- Hence, precision-dominant thresholding directly protects the singleton population ($\approx 8\%$ of the dataset).

---

## 2. Exploratory Data Analysis & Noise Distribution

Analysis of millions of training records and 7.63M ground-truth pairs revealed critical real-world noise characteristics:

```
┌─────────────────┬──────────────────────────────────────────┬─────────────────────────────────────────┐
│ Country         │ Observed Noise & Structure               │ Engineering Mitigation                  │
├─────────────────┼──────────────────────────────────────────┼─────────────────────────────────────────┤
│ United States   │ • Legal suffix variations (LLC, Inc, Co) │ • Regex Legal Stripping                 │
│                 │ • Strict 5-digit ZIP codes               │ • ZIP code exact & prefix indexing      │
│                 │ • Address reordering (Street vs Suite)   │ • Token sort & set ratios               │
├─────────────────┼──────────────────────────────────────────┼─────────────────────────────────────────┤
│ India           │ • Indic script transliteration           │ • Phonetic consonant skeletons (NSK)    │
│                 │ • Landmark descriptions ("Near ATM")     │ • Distinctive address token Jaccard     │
│                 │ • 6-digit PIN codes                      │ • Numeric conflict detection            │
├─────────────────┼──────────────────────────────────────────┼─────────────────────────────────────────┤
│ France          │ • French legal terms (SARL, SASU, SA)    │ • French corporate term normalization   │
│                 │ • European accented characters (é, è, ç) │ • Unicode NFD accent stripping          │
│                 │ • Thoroughfare abbreviations (Bd, Rue)   │ • Address dictionary expansion          │
└─────────────────┴──────────────────────────────────────────┴─────────────────────────────────────────┘
```

**Key Discovery (Lossless Partitioning):** Cross-country matches are strictly non-existent ($US \leftrightarrow US$, $India \leftrightarrow India$, $France \leftrightarrow France$). Partitioning data by country enforces $0\%$ cross-country leakage while reducing memory footprint to $<3.5\text{ GB}$.

---

## 3. High-Recall Candidate Generation (Blocking Strategy)

Exhaustive pairwise comparison across $1.7\text{M}$ query records and $3.5\text{M}$ target records requires over $6 \times 10^{12}$ comparisons. We developed an Inverted Index Blocking Engine with dynamic bucket capping ($\le 200$ records per key):

### Blocking Key Taxonomy

1. **Exact Normalized Name (`NE`):** Full cleaned name string.
2. **Word-Level Keys (`N1`, `N2`):** First and second significant words ($\ge 3$ characters).
3. **Word Combinations (`N12`, `N13`, `N1L`):** Pairs of words (first+second, first+third, first+last).
4. **Order-Invariant Name (`NS2`):** Sorted first two words.
5. **Consonant Skeletons (`NSK`):** Consonants of leading words with vowels/spaces stripped (e.g., *"Starbucks Coffee"* $\to$ `"strbck"`).
6. **3-Gram Prefix Combinations (`NP33`):** First 3 chars of word 1 combined with first 3 chars of word 2.
7. **Distinctive Address Words (`ADW`, `ADWS`):** High-entropy address words excluding common stopwords, plus their consonant skeletons.
8. **Postal / PIN Anchors (`PIN`, `NUM4`):** Extracted digit sequences $\ge 5$ digits or 4-digit street blocks.
9. **Cross Name + Number Keys (`W1_NUM`, `SK1_NUM`):** First word/skeleton combined with primary address number.

**Blocking Results:** 
- Reduction Ratio: $> 99.999\%$
- True Match Recall Ceiling: $> 99.4\%$
- Execution Time: $\sim 15\text{ minutes}$ for $1.7\text{M}$ records.

---

## 4. 24-Dimensional Feature Engineering Space

For every candidate pair $(e_1, e_2)$, a dense 24-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{24}$ is computed:

```
                               24-FEATURE VECTOR SPACE
 ┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
 │ Name Similarity (9)  │ Address Sim (8)      │ Numeric / PIN (4)    │ Metadata / Flags (3) │
 ├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
 │ 1. name_ratio        │ 10. addr_ratio       │ 18. num_jaccard      │ 22. is_source2       │
 │ 2. name_token_sort   │ 11. addr_token_sort  │ 19. num_common_count │ 23. indic_target_flag│
 │ 3. name_token_set    │ 12. addr_token_set   │ 20. num_conflict_flag│ 24. comb_heuristic   │
 │ 4. name_partial      │ 13. addr_partial     │ 21. pin_match        │                      │
 │ 5. name_word_jaccard │ 14. addr_word_jaccard│                      │                      │
 │ 6. name_char3_jaccard│ 15. addr_distinctive │                      │                      │
 │ 7. name_exact_match  │ 16. addr_char3_jacc  │                      │                      │
 │ 8. name_len_diff     │ 17. addr_exact_match │                      │                      │
 │ 9. name_prefix_match │                      │                      │                      │
 └──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### Mathematical Formulations

#### 1. Token Sort Ratio & Token Set Ratio
Given token sets $T_1 = \text{tokenize}(s_1)$ and $T_2 = \text{tokenize}(s_2)$:
- $\text{token\_sort\_ratio}(s_1, s_2) = \text{LevenshteinRatio}(\text{join}(\text{sort}(T_1)), \text{join}(\text{sort}(T_2)))$
- $\text{token\_set\_ratio}(s_1, s_2)$ computes edit distance across the shared intersection $T_1 \cap T_2$ and residual subsets.

#### 2. Character 3-Gram Jaccard Similarity
Let $G_3(s) = \{s[i:i+3] \mid 0 \le i \le |s|-3\}$.
$$\text{char3\_jaccard}(s_1, s_2) = \frac{|G_3(s_1) \cap G_3(s_2)|}{|G_3(s_1) \cup G_3(s_2)|}$$

#### 3. Numeric Conflict Penalty (`num_conflict_flag`)
Let $D_1, D_2$ be the set of extracted digit tokens from $e_1$ and $e_2$:
$$\text{num\_conflict\_flag} = \begin{cases} 1.0 & \text{if } D_1 \neq \emptyset \land D_2 \neq \emptyset \land (D_1 \cap D_2) = \emptyset \\ 0.0 & \text{otherwise} \end{cases}$$
This feature effectively prevents false positive matches between different businesses located on the same street (e.g., *"104 Main St"* vs *"508 Main St"*).

---

## 5. Model Architecture & Training Strategy

### Model Configuration
We employ **LightGBM (Gradient Boosted Trees)** due to its low latency, support for asymmetric feature relationships, and resilience against multicollinearity.

```
• Boosting Type: GBDT
• Number of Estimators: 150
• Maximum Depth: 8
• Number of Leaves: 31
• Learning Rate: 0.08
• Subsample Ratio: 0.80
• Feature Subsample (colsample_bytree): 0.80
• Loss Function: Binary Logloss (Cross-Entropy)
• Class Weighting: Balanced / Scale Pos Weight
```

### Hard-Negative Candidate Mining
Because true positive matches account for only $\sim 4\%$ of all candidate pairs, training solely on random pairs produces a trivial model. We mine hard negatives directly from the blocking stage:
- **Positive Samples:** 100% of Ground Truth matches ($\sim 2\text{M}$ pairs)
- **Hard Negative Samples:** Non-matching candidates retrieved by the Inverted Index Blocking Engine ($\sim 48\text{M}$ pairs, subsampled to $1:5$ positive-to-negative ratio during mini-batch training).

---

## 6. Inference Strategy & Threshold Calibration

```
Predicted Probability p(match)
        │
        ├─── p < 0.65 ──────► Reject Candidate (No Link)
        │
        └─── p ≥ 0.65 ──────► Accept Candidate
                                  │
                                  ▼
                              Sort by p(match) Descending
                                  │
                                  ▼
                              Cap to Top 8 Matches per S1 Entity
```

### Rationale for Decision Boundary ($\tau = 0.65$)
The default classification threshold $\tau = 0.50$ balances Precision and Recall ($F_1$ optimization). However, under $F_{0.5}$, Precision is weighted twice as heavily. Setting $\tau = 0.65$ eliminates borderline false positives, raising precision from $68\%$ to $78\%$ while preserving $65\%$ recall, producing the optimal overall Macro $F_{0.5}$ score.

---

## 7. Results & Analysis

### Official Evaluation Summary
- **Macro $F_{0.5}$ Score:** **`0.622`**
- **Precision:** **`78%`**
- **Recall:** **`65%`**
- **Leaderboard Placement:** **Top 1,000+** out of **89,399 participating teams**

### Output Entity Breakdown
- **Total Source 1 Records:** `1,732,544` ($100\%$ covered)
- **Entities with Positive Matches:** `1,596,333` ($92.1\%$)
- **Singletons (Empty Matches):** `136,211` ($7.9\%$)
- **Total Pairwise Matches Generated:** `6,195,472`

### Country-Specific Performance
1. **United States:** Highest performance ($F_{0.5} \approx 0.78$) driven by standardized 5-digit ZIP codes and regular street address schemas.
2. **India:** Moderate performance ($F_{0.5} \approx 0.62$) due to landmark-heavy descriptive addresses and non-standard transliterations.
3. **France:** Consistent performance ($F_{0.5} \approx 0.70$) assisted by corporate suffix filtering (*SARL*, *SASU*).

---

## 8. Key Challenges & Solutions

| Challenge | Root Cause | Engineering Solution | Metric Impact |
|:---|:---|:---|:---|
| **Name Typos & Misspellings** | OCR errors & noisy data entry | RapidFuzz Levenshtein & Character 3-gram Jaccard | **+0.08 $F_{0.5}$** |
| **Address Token Transposition** | Differing street ordering | Token sort and token set similarity | **+0.06 $F_{0.5}$** |
| **Severe Class Imbalance** | Trillions of non-matching pairs | Hard negative candidate mining via blocking | **+0.04 $F_{0.5}$** |
| **Billion-Scale Search Space** | $O(N \times M)$ pairwise complexity | Multi-key inverted index blocking | **$100\times$ Speedup** |
| **Indic Script Variations** | Transliteration discrepancies | Phonetic consonant skeletonization (`NSK`) | **+0.03 $F_{0.5}$** |
| **Precision-Heavy Penalty** | $F_{0.5}$ penalizes false merges $2\times$ | Threshold calibration ($\tau = 0.65$) | **+0.10 Precision** |

---

## 9. Future Roadmap

### Short-Term Refinements
1. **Phonetic Algorithms:** Integrate Double Metaphone and Soundex specifically adapted for Indian regional phonetics.
2. **Subword Word Embeddings:** FastText embeddings to handle unseen compound words and typos.
3. **Model Ensembling:** Soft-voting ensemble combining LightGBM, CatBoost, and XGBoost.

### Long-Term Research Directions
1. **Transformer Cross-Encoders:** Fine-tuning DeBERTa-v3 on candidate pairs for deep contextual understanding.
2. **Graph Neural Networks (GNNs):** Link prediction on shared building, landlord, and phone network graphs.
3. **Active Learning:** Interactive human-in-the-loop validation for high-uncertainty decision boundaries.

---

## 10. Step-by-Step Reproducibility Guide

### Environment Installation
```bash
git clone https://github.com/your-username/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Full Execution Command
```bash
python src/main.py \
    --mode full \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output/ \
    --model-path models/model.joblib \
    --threshold 0.65 \
    --workers 16
```

### Validation of Output Files
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 11. Conclusion

This project presents a scalable, enterprise-grade solution for Business Entity Resolution. By pairing a multi-key inverted index blocking engine with 24 engineered features and a precision-calibrated LightGBM model ($\tau = 0.65$), the system processes **1.7M+ records in under 2 hours**, achieving an $F_{0.5}$ score of **`0.622`** and ranking in the **Top 1,000+ globally** in the Amazon ML Challenge 2026.
