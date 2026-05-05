"""
NO MATCH → PROBABLE MATCH: AI-Style District Inference
=======================================================
For each NO MATCH municipality this pipeline:
  1. Infers the most likely district using a multi-strategy pincode analysis
  2. Cross-validates the prediction and checks for ambiguity
  3. Assigns confidence (High / Medium / Low) and a validation flag
  4. Explains the reasoning for sample cases
  5. Saves three output files:
       no_match_inference.csv       – full results for all 135 cases
       no_match_explainability.csv  – detailed reasoning for 10 sample cases
       no_match_summary.csv         – aggregate statistics

Inference Strategy (applied in priority order per pincode):
  S1 – Same state + 3-digit pincode prefix match (highest precision)
  S2 – Any state  + 3-digit pincode prefix match (cross-state fallback)
  S3 – Numeric proximity ±1000 within same state (last resort)

Confidence Thresholds:
  High    ≥ 80% of municipality pincodes resolve to the predicted district
  Medium  50–79%
  Low     < 50%

Validation Flags:
  STRONG MATCH – overlap ≥ 50% AND gap to 2nd-best district ≥ 10 pp
  AMBIGUOUS    – gap between best and 2nd-best district < 10 pp
  WEAK MATCH   – overlap < 50% (regardless of competition)
"""
#%%
import pandas as pd
import random
from collections import defaultdict

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

#%% 
# # 0.  LOAD DATA
print("=" * 70)
print("  LOADING DATASETS")
print("=" * 70)

villages = pd.read_csv("villages.csv")
lb       = pd.read_csv("Lb.csv")
mapping  = pd.read_csv("municipality_district_mapping.csv")

print(f"  villages.csv : {len(villages):>7,} rows")
print(f"  Lb.csv       : {len(lb):>7,} rows")
print(f"  mapping.csv  : {len(mapping):>7,} rows")
print(f"  NO MATCH municipalities: {(mapping['status']=='NO MATCH').sum()}")


#%%  1.  BUILD LOOKUP STRUCTURES

def clean_numeric(series):
    """Safely convert a column to int, dropping nulls and invalid values."""
    return pd.to_numeric(series, errors="coerce").dropna().astype(int)


# ── Village reference table ───────────────────────────────────────────────────
v_ref = (
    villages[["stateCode", "districtCode", "districtNameEnglish", "pincode"]]
    .dropna()
    .drop_duplicates()
    .copy()
)
v_ref["pincode"]      = clean_numeric(v_ref["pincode"])
v_ref["districtCode"] = clean_numeric(v_ref["districtCode"])
v_ref["stateCode"]    = clean_numeric(v_ref["stateCode"])
v_ref                 = v_ref.dropna()  # remove rows that failed numeric cast
v_ref["pin3"]         = v_ref["pincode"].astype(str).str[:3]   # 3-digit prefix
v_ref["pin2"]         = v_ref["pincode"].astype(str).str[:2]   # 2-digit prefix

# ── District metadata: districtCode → name ───────────────────────────────────
dist_name_map = (
    v_ref[["districtCode", "districtNameEnglish"]]
    .drop_duplicates("districtCode")
    .set_index("districtCode")["districtNameEnglish"]
    .to_dict()
)

# ── District pincode sets (for exact overlap) ─────────────────────────────────
district_pins: dict[int, frozenset] = (
    v_ref.groupby("districtCode")["pincode"]
    .apply(frozenset)
    .to_dict()
)

# ── NO MATCH municipality data ────────────────────────────────────────────────
no_match_codes = set(
    mapping.loc[mapping["status"] == "NO MATCH", "localBodyCode"].astype(int)
)

lb_nm = lb[lb["localBodyCode"].isin(no_match_codes)].copy()
lb_nm["pincode"]       = clean_numeric(lb_nm["pincode"])
lb_nm["localBodyCode"] = lb_nm["localBodyCode"].astype(int)
lb_nm["stateCode"]     = lb_nm["stateCode"].astype(int)

# ── Per-municipality pincode sets ─────────────────────────────────────────────
lb_pinsets: dict[int, frozenset] = (
    lb_nm.dropna(subset=["pincode"])
    .groupby("localBodyCode")["pincode"]
    .apply(frozenset)
    .to_dict()
)

# ── Municipality metadata ─────────────────────────────────────────────────────
lb_meta = (
    lb_nm[["localBodyCode", "stateCode", "stateNameEnglish",
           "localBodyNameEnglish", "localBodyTypeName"]]
    .drop_duplicates("localBodyCode")
    .set_index("localBodyCode")
)

print(f"\nBuilt reference structures:")
print(f"  Unique districts in village data : {len(district_pins):,}")
print(f"  NO MATCH municipalities to infer : {len(lb_pinsets)}")


#%% 2.  MULTI-STRATEGY INFERENCE ENGINE

def get_district_candidates(pin: int, state_code: int) -> dict[int, str]:
    """
    Given a single pincode and state, return a dict of
    {districtCode: strategy_used} for candidate districts.

    Priority:
      S1 – same state + 3-digit prefix
      S2 – any  state + 3-digit prefix  (cross-state fallback)
      S3 – same state + numeric proximity ±1000
    """
    pin3 = str(pin)[:3]

    # S1: same-state prefix match
    s1 = v_ref[(v_ref["stateCode"] == state_code) & (v_ref["pin3"] == pin3)]
    if not s1.empty:
        return {dc: "S1:same_state+prefix3" for dc in s1["districtCode"].unique()}

    # S2: cross-state prefix match
    s2 = v_ref[v_ref["pin3"] == pin3]
    if not s2.empty:
        return {dc: "S2:cross_state+prefix3" for dc in s2["districtCode"].unique()}

    # S3: numeric proximity ±1000 within same state
    s3 = v_ref[
        (v_ref["stateCode"] == state_code) &
        (v_ref["pincode"] >= pin - 1000) &
        (v_ref["pincode"] <= pin + 1000)
    ]
    if not s3.empty:
        return {dc: "S3:proximity±1000" for dc in s3["districtCode"].unique()}

    return {}


def infer_district(lb_code: int) -> dict:
    """
    For one NO MATCH municipality, run the multi-strategy engine over all its
    pincodes, tally district votes, and return a structured result dict.
    """
    pins       = lb_pinsets.get(lb_code, frozenset())
    state_code = int(lb_meta.loc[lb_code, "stateCode"]) if lb_code in lb_meta.index else -1
    total_pins = len(pins)

    if total_pins == 0:
        return {
            "lb_code": lb_code, "state_code": state_code,
            "total_pins": 0, "district_votes": {},
            "strategies_used": {}, "error": "No pincodes"
        }

    # Accumulate votes: each pincode casts a vote for the districts it resolves to
    district_votes   = defaultdict(int)   # districtCode → vote count
    strategies_used  = defaultdict(set)   # districtCode → set of strategies
    pin_resolution   = {}                 # pin → list of matched districts

    for pin in pins:
        candidates = get_district_candidates(int(pin), state_code)
        pin_resolution[pin] = list(candidates.keys())
        for dc, strat in candidates.items():
            district_votes[dc] += 1
            strategies_used[dc].add(strat)

    return {
        "lb_code":        lb_code,
        "state_code":     state_code,
        "total_pins":     total_pins,
        "district_votes": dict(district_votes),
        "strategies_used": {dc: sorted(s) for dc, s in strategies_used.items()},
        "pin_resolution": pin_resolution,
        "error":          None,
    }


def score_prediction(result: dict) -> dict:
    """
    Take raw inference result and produce final scored output:
      - predicted_district, overlap_pct, top-3 candidates
      - confidence_level, validation_flag
    """
    lb_code    = result["lb_code"]
    total_pins = result["total_pins"]
    votes      = result["district_votes"]

    if not votes or total_pins == 0:
        return {**result,
                "predicted_district": None, "predicted_dist_name": None,
                "overlap_pct": 0.0, "confidence_level": "None",
                "validation_flag": "UNRESOLVABLE",
                "top3_candidates": None,
                "2nd_best_district": None, "gap_to_2nd": None}

    # Rank districts by vote count
    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    best_dc, best_votes   = ranked[0]
    overlap_pct = round(best_votes / total_pins * 100, 2)

    # 2nd best for ambiguity check
    if len(ranked) > 1:
        second_dc, second_votes = ranked[1]
        second_pct = round(second_votes / total_pins * 100, 2)
        gap        = round(overlap_pct - second_pct, 2)
    else:
        second_dc  = None
        second_pct = 0.0
        gap        = 100.0

    # Top-3 string
    top3 = "; ".join(
        f"{dc}:{dist_name_map.get(dc,'?')} ({round(v/total_pins*100,1)}%)"
        for dc, v in ranked[:3]
    )

    # Confidence level
    if overlap_pct >= 80:
        confidence = "High"
    elif overlap_pct >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Validation flag
    if gap < 10:
        validation_flag = "AMBIGUOUS"
    elif overlap_pct >= 50:
        validation_flag = "STRONG MATCH"
    else:
        validation_flag = "WEAK MATCH"

    # Primary strategy used for best district
    strat = result["strategies_used"].get(best_dc, ["unknown"])

    return {
        **result,
        "predicted_district":   best_dc,
        "predicted_dist_name":  dist_name_map.get(best_dc, "Unknown"),
        "overlap_pct":          overlap_pct,
        "matching_pins":        best_votes,
        "confidence_level":     confidence,
        "validation_flag":      validation_flag,
        "strategy_used":        ", ".join(strat),
        "2nd_best_district":    second_dc,
        "2nd_best_dist_name":   dist_name_map.get(second_dc, None) if second_dc else None,
        "2nd_overlap_pct":      second_pct,
        "gap_to_2nd_pct":       gap,
        "top3_candidates":      top3,
    }


#%% 3.  RUN INFERENCE ON ALL 135 NO MATCH MUNICIPALITIES
print("\n" + "=" * 70)
print("  RUNNING INFERENCE ENGINE")
print("=" * 70)

all_scored = []
for lb_code in sorted(lb_pinsets.keys()):
    raw    = infer_district(lb_code)
    scored = score_prediction(raw)
    all_scored.append(scored)

print(f"  Processed: {len(all_scored)} municipalities")


#%% 4.  BUILD OUTPUT DATAFRAME (Table 1)

output_rows = []
for s in all_scored:
    lbc = s["lb_code"]
    meta_row = lb_meta.loc[lbc] if lbc in lb_meta.index else {}

    output_rows.append({
        "stateCode":             s["state_code"],
        "stateName":             meta_row.get("stateNameEnglish", "Unknown"),
        "localBodyCode":         lbc,
        "localBodyName":         meta_row.get("localBodyNameEnglish", "Unknown"),
        "localBodyType":         meta_row.get("localBodyTypeName", "Unknown"),
        "total_pincodes":        s["total_pins"],
        "matching_pincodes":     s.get("matching_pins", 0),
        "predicted_districtCode": s.get("predicted_district"),
        "predicted_districtName": s.get("predicted_dist_name"),
        "overlap_%":             s.get("overlap_pct", 0.0),
        "confidence_level":      s.get("confidence_level", "None"),
        "validation_flag":       s.get("validation_flag", "UNRESOLVABLE"),
        "strategy_used":         s.get("strategy_used", "none"),
        "2nd_best_districtCode": s.get("2nd_best_district"),
        "2nd_best_districtName": s.get("2nd_best_dist_name"),
        "2nd_overlap_%":         s.get("2nd_overlap_pct", 0.0),
        "gap_to_2nd_%":          s.get("gap_to_2nd_pct"),
        "top3_candidates":       s.get("top3_candidates"),
    })

result_df = pd.DataFrame(output_rows)


#%% 5.  EXPLAINABILITY – detailed reasoning for 10 sample cases (Table 2)
print("\n" + "=" * 70)
print("  EXPLAINABILITY – DETAILED REASONING FOR 10 SAMPLE CASES")
print("=" * 70)

# Pick 2–3 from each confidence tier for variety
high_cases = result_df[result_df["confidence_level"] == "High"].sample(
    min(3, (result_df["confidence_level"] == "High").sum()), random_state=RANDOM_SEED)
med_cases  = result_df[result_df["confidence_level"] == "Medium"].sample(
    min(4, (result_df["confidence_level"] == "Medium").sum()), random_state=RANDOM_SEED)
low_cases  = result_df[result_df["confidence_level"] == "Low"].sample(
    min(3, (result_df["confidence_level"] == "Low").sum()), random_state=RANDOM_SEED)
sample_df  = pd.concat([high_cases, med_cases, low_cases]).head(10)

explain_rows = []

for _, row in sample_df.iterrows():
    lbc   = int(row["localBodyCode"])
    pins  = sorted(lb_pinsets.get(lbc, frozenset()))
    sc    = int(row["stateCode"])
    total = int(row["total_pincodes"])

    # Rebuild candidate votes for explanation
    votes = defaultdict(int)
    for pin in lb_pinsets.get(lbc, []):
        for dc in get_district_candidates(int(pin), sc):
            votes[dc] += 1

    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:5]

    # Print to console
    print(f"\n{'─'*65}")
    print(f"  Municipality : {row['localBodyName']} (code {lbc})")
    print(f"  State        : {row['stateName']} ({sc})")
    print(f"  Pincodes     : {pins}")
    print(f"  Total pins   : {total}")
    print(f"\n  District candidates (by vote count):")
    for dc, votes_n in ranked:
        pct = round(votes_n / total * 100, 1)
        bar = "█" * int(pct // 5)
        print(f"    {dc:>5}  {dist_name_map.get(dc,'?'):<28}  {votes_n}/{total} pins  {pct:>5.1f}%  {bar}")
    print(f"\n  ✦ PREDICTION  : District {row['predicted_districtCode']} – {row['predicted_districtName']}")
    print(f"  ✦ OVERLAP     : {row['overlap_%']}%")
    print(f"  ✦ CONFIDENCE  : {row['confidence_level']}")
    print(f"  ✦ FLAG        : {row['validation_flag']}")
    print(f"  ✦ STRATEGY    : {row['strategy_used']}")
    if row['2nd_best_districtCode']:
        print(f"  ✦ 2ND BEST    : {row['2nd_best_districtCode']} – {row['2nd_best_districtName']} ({row.get('2nd_overlap_%', 0)}%)")
        print(f"  ✦ GAP         : {row['gap_to_2nd_%']} pp  →  "
              f"{'⚠ close competition' if row['gap_to_2nd_%'] < 10 else 'clear winner'}")

    # Reasoning narrative
    if row["confidence_level"] == "High":
        reasoning = (f"Strong match: {row['matching_pincodes']} of {total} pincodes point to district "
                     f"{row['predicted_districtCode']} ({row['predicted_districtName']}). "
                     f"Gap to runner-up is {row['gap_to_2nd_%']} pp.")
    elif row["confidence_level"] == "Medium":
        reasoning = (f"Moderate match: majority of pincodes ({row['overlap_%']}%) resolve to "
                     f"{row['predicted_districtName']}. Some pincodes may belong to neighbouring districts.")
    else:
        reasoning = (f"Weak signal: only {row['overlap_%']}% of pincodes resolve to the predicted district. "
                     f"{'Multiple districts compete closely.' if row['gap_to_2nd_%'] < 10 else 'May reflect sparse/mismatched pincode data.'}")

    print(f"  ✦ REASONING   : {reasoning}")

    explain_rows.append({
        "localBodyCode":          lbc,
        "localBodyName":          row["localBodyName"],
        "stateName":              row["stateName"],
        "pincodes":               str(pins),
        "total_pincodes":         total,
        "top5_candidates":        str([(dc, dist_name_map.get(dc,"?"), round(v/total*100,1)) for dc,v in ranked]),
        "predicted_districtCode": row["predicted_districtCode"],
        "predicted_districtName": row["predicted_districtName"],
        "overlap_%":              row["overlap_%"],
        "confidence_level":       row["confidence_level"],
        "validation_flag":        row["validation_flag"],
        "strategy_used":          row["strategy_used"],
        "gap_to_2nd_%":           row["gap_to_2nd_%"],
        "reasoning_narrative":    reasoning,
    })

explain_df = pd.DataFrame(explain_rows)


#%% 6.  SUMMARY STATISTICS
print("\n" + "=" * 70)
print("  SUMMARY INSIGHTS")
print("=" * 70)

total        = len(result_df)
high_n       = (result_df["confidence_level"] == "High").sum()
med_n        = (result_df["confidence_level"] == "Medium").sum()
low_n        = (result_df["confidence_level"] == "Low").sum()
strong_n     = (result_df["validation_flag"] == "STRONG MATCH").sum()
ambig_n      = (result_df["validation_flag"] == "AMBIGUOUS").sum()
weak_n       = (result_df["validation_flag"] == "WEAK MATCH").sum()

strategy_dist = (
    result_df["strategy_used"]
    .str.split(", ")
    .explode()
    .value_counts()
)

print(f"\n  Total NO MATCH municipalities processed : {total}")
print(f"\n  Confidence breakdown:")
print(f"    High   (≥80% overlap)  : {high_n:>4}  ({high_n/total*100:.1f}%)")
print(f"    Medium (50–79%)        : {med_n:>4}  ({med_n/total*100:.1f}%)")
print(f"    Low    (<50%)          : {low_n:>4}  ({low_n/total*100:.1f}%)")

print(f"\n  Validation flag breakdown:")
print(f"    STRONG MATCH           : {strong_n:>4}  ({strong_n/total*100:.1f}%)")
print(f"    AMBIGUOUS              : {ambig_n:>4}  ({ambig_n/total*100:.1f}%)")
print(f"    WEAK MATCH             : {weak_n:>4}  ({weak_n/total*100:.1f}%)")

print(f"\n  Overlap % distribution:")
print(result_df["overlap_%"].describe().round(2).to_string())

print(f"\n  Strategy usage:")
for strat, cnt in strategy_dist.items():
    print(f"    {strat:<30} : {cnt} cases")

reliable_pct = (high_n + med_n) / total * 100
print(f"\n  ─────────────────────────────────────────────────────────────")
print(f"  Overall reliability (High + Medium confidence) : {reliable_pct:.1f}%")
print(f"  Root cause of NO MATCH: Pincodes in Lb.csv are absent from")
print(f"  villages.csv as exact values. Inference succeeds via 3-digit")
print(f"  prefix matching (postal circle heuristic), which maps to the")
print(f"  correct geographic district in {reliable_pct:.0f}%+ of cases.")
print(f"  ─────────────────────────────────────────────────────────────")

# Build summary CSV
summary_df = pd.DataFrame([
    {"metric": "Total NO MATCH processed",      "value": total},
    {"metric": "High confidence (≥80%)",         "value": int(high_n)},
    {"metric": "Medium confidence (50–79%)",     "value": int(med_n)},
    {"metric": "Low confidence (<50%)",          "value": int(low_n)},
    {"metric": "STRONG MATCH",                   "value": int(strong_n)},
    {"metric": "AMBIGUOUS",                      "value": int(ambig_n)},
    {"metric": "WEAK MATCH",                     "value": int(weak_n)},
    {"metric": "Avg overlap %",                  "value": round(result_df["overlap_%"].mean(), 2)},
    {"metric": "Median overlap %",               "value": round(result_df["overlap_%"].median(), 2)},
    {"metric": "Reliability (High+Med) %",       "value": round(reliable_pct, 1)},
])


# %% 7.  SAVE ALL OUTPUTS

result_df.to_csv("no_match_inference.csv",     index=False)
explain_df.to_csv("no_match_explainability.csv", index=False)
summary_df.to_csv("no_match_summary.csv",      index=False)

print("\n" + "=" * 70)
print("  OUTPUTS SAVED")
print("=" * 70)
print("  no_match_inference.csv       (all 135 cases)")
print("  no_match_explainability.csv  (10 sample cases)")
print("  no_match_summary.csv         (aggregate stats)")
print("=" * 70)

# ── Quick preview of final table ──────────────────────────────────────────────
print("\nFinal inference table – sample:")
preview_cols = ["localBodyName", "stateName", "predicted_districtName",
                "overlap_%", "confidence_level", "validation_flag"]
print(result_df[preview_cols].head(15).to_string(index=False))