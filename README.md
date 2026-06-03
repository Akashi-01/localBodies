# Municipality-to-District Mapping Pipeline

A robust, data-driven Python pipeline designed to map Indian municipalities (Local Bodies) to their corresponding administrative districts using pincode-level spatial heuristics and multi-strategy fallback logic.

---

## 🕒 Commit History & Evolution Sequence

Below is the chronological sequence of development commits reconstructed directly from the project's repository logs.

| Commit Message & Purpose | Core Changes & Steps Taken |
| :--- | :--- |
| `Initial commit` | Added raw datasets and the initial exploration script (analysis.py). |
| `updated address dataset analysis` | Implemented core matching logic and categorizations (Full, Ambiguous, No Match) in Analysis2.py. Outputted municipality_district_mapping.csv. |
| `Advance address analysis` | Developed Advance_analysis.py to enrich mapping tables with state/local body metadata, outputting enriched_mapping.csv and ambiguous_analysis.csv. |
| `No_match_prediction(explainablity,inference,summary)` | Created no_matches_prediction.py focusing on the 135 `NO MATCH` cases. Introduced the multi-strategy candidate vote engine (S1/S2/S3) with confidence scores. |
| `mapping all the local body data based on unique, ambiguous and no match predictions` | Scaled the entire pipeline in final_data_mapping.py to process all 5,005 municipalities. Generated full-size intermediate datasets. |
| `finalized dataset` | Wrote Final_mapping.py to merge the three resolved segments (Exact, Ambiguous, AI Predicted) and run validation checks to create the master dataset. |
| `rechecked` | Conducted final audits, verification checks, and sanity test passes over final pipeline logs. |

---

## 📂 Project Structure (At Latest Commit)

At the final commit (`f9ca97e1`), the repository has the following structure:

```directory
c:/python/
├── README.md                              # This documentation file
└── localBodies/                           # Core project folder
    ├── .git/                              # Git metadata directory
    │
    │   # 🐍 Python Pipeline Scripts
    ├── analysis.py                        # Preliminary pincode analysis and helper checks
    ├── Analysis2.py                       # Core pincode-intersection matching logic
    ├── Advance_analysis.py                 # Enriched metadata joining & ambiguity statistics
    ├── no_matches_prediction.py           # Multi-strategy inference engine for missing pincodes
    ├── final_data_mapping.py              # Scaled master pipeline processing all records
    ├── Final_mapping.py                   # Synthesis, deduplication, and quality-assurance checks
    │
    │   # 📊 Raw Input Data
    ├── Lb.csv                             # Local Body registry (5,005 municipalities, pincodes, types)
    ├── villages.csv                       # Master Village-level dataset mapping pincodes to Districts (45MB)
    ├── [state_name].csv                   # State-wise split village data (e.g. up.csv, bihar.csv)
    │
    │   # 📈 Intermediate & Diagnostic Outputs
    ├── municipality_district_mapping.csv  # Basic status mapping from core matching
    ├── enriched_mapping.csv               # Enriched metadata (state/LB/type names) mapping
    ├── enriched_mapping_full.csv          # Flat-file multi-row mapping including names
    ├── ambiguous_analysis.csv             # Overlap percentage diagnostics (20-sample run)
    ├── ambiguous_analysis_full.csv        # Detailed overlap metrics for all 613 ambiguous cases
    ├── no_match_predictions.csv           # Diagnostic AI inferences for sample cases
    ├── no_match_ai_predictions_full.csv   # Structured inferences for all 135 unmapped cases
    ├── no_match_inference.csv             # Full inference predictions & top 3 candidates
    ├── no_match_explainability.csv        # Readability explanations for unmapped sample cases
    ├── no_match_summary.csv               # Aggregated inference metrics
    ├── one_to_one_matches.csv             # Direct 1-to-1 pincode matches
    ├── multi_mapping_cases.csv            # Identified overlapping pincodes
    │
    │   # 🏆 Final Gold Dataset
    └── final_municipality_district_mapping.csv # Exactly 5,005 clean, deduplicated mapped records
```

---

## ⚙️ How the Pipeline Logic Works

The project resolves mappings by grouping municipalities into three logical statuses:

### 1. FULL MATCH (4,257 Municipalities)
* **Definition**: All pincodes recorded for the municipality lie entirely within a single district.
* **Resolution**: Map directly to that district with **High Confidence** (`1.0`). If multiple districts claim the exact same pincode subset, the tie is broken deterministically by selecting the lowest `districtCode`.

### 2. AMBIGUOUS MATCH (613 Municipalities)
* **Definition**: Pincodes registered under the municipality overlap with multiple districts in the village dataset.
* **Resolution**: Re-derive overlap percentages for every overlapping district.
  * **High Disparity (>30% Gap)**: One district is strongly dominant (mapped with high confidence, flagged `NEARLY RESOLVABLE`).
  * **Moderate Disparity (10–30% Gap)**: Flagged as `MODERATE AMBIGUITY`.
  * **Low Disparity (<10% Gap)**: Highly contested border cases (flagged `HIGH AMBIGUITY`).

### 3. AI PREDICTION / NO MATCH (135 Municipalities)
* **Definition**: None of the municipality's pincodes exist as exact matches in `villages.csv` (typically due to data entry gaps).
* **Resolution**: Uses a multi-strategy heuristic engine to infer the district based on postal circle prefixes:
  * **S1**: Check matching 3-digit pincode prefixes in the **same state**.
  * **S2**: Check matching 3-digit pincode prefixes **across any state** (fallback).
  * **S3**: Check numeric pincode proximity (within $\pm1000$ range) in the same state.
* **Performance**: Achieves **>94.8%** overall reliability (High + Medium confidence resolution).

---

## 📊 Final Dataset Summary Statistics

Running `Final_mapping.py` performs rigorous quality checks, deduplicates the data prioritizing the strongest mappings, and outputs a clean list of exactly **5,005 unique municipalities** with the following characteristics:

* **Exact Matches (FULL MATCH)**: 85.1% (4,257 municipalities)
* **Ambiguous Matches (AMBIGUOUS MATCH)**: 12.2% (613 municipalities)
* **AI Inferences (AI PREDICTION)**: 2.7% (135 municipalities)
* **Validation Health**: 
  * `EXACT MATCH`: 85.1%
  * `NEARLY RESOLVABLE`: 7.0%
  * `STRONG MATCH` (AI): 2.4%
  * `MODERATE AMBIGUITY`: 2.9%
  * `HIGH AMBIGUITY`: 2.3%
  * `WEAK MATCH` (AI): 0.3%
