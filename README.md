# Vehicle-Infotainment-Warranty-Analysis
# Advanced EDA and Text Mining 
## Overview
This project performs comprehensive Exploratory Data Analysis (EDA), Text Mining, and actionable insight generation on a dataset of 1,000 vehicle warranty records. All complaints relate to in-vehicle infotainment systems.
The analysis covers three major sections:
  1. Exploratory Data Analysis
  2. Text Mining using NLP
  3. Insights and Recommendations for Stakeholders

### Dataset
- Records: 1,000 warranty/repair events
- Model Year: MY2020 (single production cohort)
- Vehicle brands: 4 vehicle brands
- Build plants: 8 manufacturing locations

## Section 1 — Exploratory Data Analysis
### Data Quality
 - Data Cleaning
     1. 0 duplicate records
     2. 5% missing values handled with fillna
     3. Confirmed single production cohort
Key Findings
 - Number of cases warranty/repair cases
 - Peak Timing of cases - February 2020, ~6 months after build
 - Mean timt-to-time failure: 339 days
 - Which brand failure - ThunderVolt (48.8% cases) - dominant across all failure types
 - Which plant have highest case volume - Fort Wayne plant

### Visualizations produced
  - Complaint category bar chart
  - Vehicle make distribution
  - Yearly and monthly event frequency line charts
  - Build date vs. opened date lag histogram and boxplot
  - Make × Cluster heatmap
  - Build plant vs. issue type stacked bar chart

## Section 2 — Text Mining (NLP)
### I. Free-Text Columns Identified
### II. Text Cleaning Pipeline
  - Lowercasing
  - Special character removal (regex)
  - NLTK stopword removal
  - Combined into combined_text field for vectorization

### III. Entity Extraction
Issue Type Classification
Rule-based classification — structured ground-truth columns — rather than raw text keywords (which cause false positives due to generic service language).
### IV. Clustering — K-Means on TF-IDF
  - Vectorization: TF-IDF with 2,000 features, unigrams + bigrams
  - Optimal K: 5 (selected using Silhouette Score evaluation across K=2–7)
  - 5 clusters identified with unique names

## Section 3 — Insights and Recommendations
### Key Patterns Identified
  1. All failures trace to a single 37-day production batch
  2. Failure frequency followed a classic warranty ramp curve — ramping up 6 months post-build and declining by mid-2021
  3. ThunderVolt appears disproportionately across every cluster type, suggesting a systemic brand-level issue rather than a single component failure
  4. 59% of all failures are hardware/module faults — the radio module assembly is the single most common replaced component

## Tech Stack
Python 3.10+
pandas
numpy
matplotlib
seaborn
scikit-learn      # TF-IDF, KMeans, silhouette_score
nltk              # stopwords, text cleaning
re                # regex text processing

## Further Improvements

 - BERTopic / fine-tuned BERT for context-aware clustering — TF-IDF cannot distinguish "radio replaced" from "replace radio"
 - Language detection + translation for Spanish/French verbatims (currently flagged, not processed)
 - Predictive model — use make, plant, and cluster to predict time-to-failure for proactive warranty alerts
 - Cost-weighted recommendations — integrate repair cost data to prioritise by financial impact, not just case count
 - Finer clustering (K=8–10) — current 5 clusters may be grouping distinguishable sub-types together
