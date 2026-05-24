"""
Final Municipality–District Mapping: Clean, Deduplicated, Reporting-Ready
==========================================================================
Inputs:
  enriched_mapping.csv            – 6,775 rows / 5,005 unique municipalities
                                    (FULL MATCH, AMBIGUOUS MATCH, NO MATCH rows;
                                     multi-district LBs have multiple rows)
  ambiguous_analysis_full.csv     – 613 unique municipalities with full overlap detail
  no_match_ai_predictions_full.csv – 135 unique municipalities with AI predictions

Output:
  final_municipality_district_mapping.csv – exactly 5,005 rows, one per municipality

Design:
  • enriched_mapping is the master list (all 5,005 LB codes).
  • For FULL MATCH LBs: collapse multi-district rows → one row per LB (lowest districtCode).
  • For AMBIGUOUS LBs: use richer ambiguous_analysis row (top district, disparity, overlap %).
  • For NO MATCH LBs: use AI-prediction row (predicted district, confidence, flag).
  • Priority dedup (safety net): FULL MATCH > AMBIGUOUS > AI PREDICTION.
"""


#%%
# TASK 1 – LOAD
import pandas as pd
print("  TASK 1 – LOADING INPUT FILES \n")


enriched  = pd.read_csv("enriched_mapping.csv")
ambiguous = pd.read_csv("ambiguous_analysis_full.csv")
no_match  = pd.read_csv("no_match_ai_predictions_full.csv")

print(f"  enriched_mapping.csv              : {len(enriched):>5,} rows | "
      f"{enriched['localBodyCode'].nunique():,} unique municipalities")
print(f"  ambiguous_analysis_full.csv       : {len(ambiguous):>5,} rows | "
      f"{ambiguous['localBodyCode'].nunique():,} unique municipalities")
print(f"  no_match_ai_predictions_full.csv  : {len(no_match):>5,} rows | "
      f"{no_match['localBodyCode'].nunique():,} unique municipalities")

EXPECTED_TOTAL = 5_005

#%%
# TASK 2 & 3 – STANDARDISE + ASSIGN FINAL DISTRICT LOGIC
# Three clean segments are built; each has identical columns.

COLS = [
    "statecode", "statename",
    "districtcode", "district_name",
    "localbodycode", "localbody_name", "localbody_type",
    "status", "confidence_score", "disparity_category", "validation_flag",
    "top3_candidates",
]

# SEGMENT A – FULL MATCH  (4,257 unique municipalities)

print("\n" + "=" * 68)
print("  TASK 3a – SEGMENT: FULL MATCH")
print("=" * 68)

full_raw = (
    enriched[enriched["status"] == "FULL MATCH"]
    # Some LBs are covered by multiple districts; keep lowest districtCode
    # (deterministic; all listed districts are valid exact matches)
    .sort_values(["localBodyCode", "districtCode"])
    .drop_duplicates(subset=["localBodyCode"], keep="first")
    .reset_index(drop=True)
)

seg_full = pd.DataFrame({
    "statecode":          pd.to_numeric(full_raw["stateCode"],    errors="coerce").astype("Int64"),
    "statename":          full_raw["stateName"],
    "districtcode":       pd.to_numeric(full_raw["districtCode"], errors="coerce").astype("Int64"),
    "district_name":      full_raw["districtName"],
    "localbodycode":      full_raw["localBodyCode"].astype(int),
    "localbody_name":     full_raw["localBodyName"],
    "localbody_type":     full_raw["localBodyType"],
    "status":             "FULL MATCH",
    "confidence_score":   1.0,           # exact pincode coverage = certainty
    "disparity_category": pd.NA,
    "validation_flag":    "EXACT MATCH",
    "top3_candidates":    pd.NA,
})

print(f"  Rows in segment: {len(seg_full):,}")

# SEGMENT B – AMBIGUOUS MATCH  (613 unique municipalities)
print("\n" + "=" * 68)
print("  TASK 3b – SEGMENT: AMBIGUOUS MATCH")
print("=" * 68)

# Pull localBodyType from enriched (not in ambiguous file)
lb_type_map = (
    enriched[["localBodyCode", "localBodyType"]]
    .drop_duplicates("localBodyCode")
    .set_index("localBodyCode")["localBodyType"]
)

# Confidence label derived from top_overlap_%
def overlap_to_confidence(pct):
    if pct >= 80:   return 1.0
    if pct >= 50:   return round(pct / 100, 4)
    return round(pct / 100, 4)

# Validation flag derived from disparity category
DISP_TO_FLAG = {
    "HIGH DISPARITY":     "NEARLY RESOLVABLE",
    "MODERATE DISPARITY": "MODERATE AMBIGUITY",
    "LOW DISPARITY":      "HIGH AMBIGUITY",
}

seg_ambig = pd.DataFrame({
    "statecode":          pd.to_numeric(ambiguous["stateCode"],        errors="coerce").astype("Int64"),
    "statename":          ambiguous["stateName"],
    "districtcode":       pd.to_numeric(ambiguous["top_districtCode"], errors="coerce").astype("Int64"),
    "district_name":      ambiguous["top_districtName"],
    "localbodycode":      ambiguous["localBodyCode"].astype(int),
    "localbody_name":     ambiguous["localBodyName"],
    "localbody_type":     ambiguous["localBodyCode"].map(lb_type_map),
    "status":             "AMBIGUOUS MATCH",
    "confidence_score":   ambiguous["top_overlap_%"].apply(overlap_to_confidence),
    "disparity_category": ambiguous["disparity_category"],
    "validation_flag":    ambiguous["disparity_category"].map(DISP_TO_FLAG),
    "top3_candidates":    ambiguous["candidate_districts"].str.split(" | ").apply(
                              lambda x: " | ".join(x[:3]) if isinstance(x, list) else x
                          ),
})

print(f"  Rows in segment: {len(seg_ambig):,}")

# SEGMENT C – AI PREDICTION  (135 unique municipalities)
print("\n" + "=" * 68)
print("  TASK 3c – SEGMENT: AI PREDICTION (NO MATCH)")
print("=" * 68)

# Map confidence_level string → numeric score
CONF_MAP = {"HIGH": 1.0, "MEDIUM": 0.65, "LOW": 0.3}

seg_nm = pd.DataFrame({
    "statecode":          pd.to_numeric(no_match["stateCode"],              errors="coerce").astype("Int64"),
    "statename":          no_match["stateName"],
    "districtcode":       pd.to_numeric(no_match["predicted_districtCode"], errors="coerce").astype("Int64"),
    "district_name":      no_match["predicted_districtName"],
    "localbodycode":      no_match["localBodyCode"].astype(int),
    "localbody_name":     no_match["localBodyName"],
    "localbody_type":     no_match["localBodyType"],
    "status":             "AI PREDICTION",
    "confidence_score":   no_match["overlap_%"].apply(lambda x: round(x / 100, 4)),
    "disparity_category": pd.NA,
    "validation_flag":    no_match["validation_flag"],
    "top3_candidates":    no_match["top3_candidates"],
})

print(f"  Rows in segment: {len(seg_nm):,}")

#%%
# TASK 4 – CONCATENATE
print("\n" + "=" * 68)
print("  TASK 4 – CONCATENATE ALL SEGMENTS")
print("=" * 68)

combined = pd.concat(
    [seg_full, seg_ambig, seg_nm],
    ignore_index=True
)[COLS]

print(f"  Total rows after concat : {len(combined):,}")
print(f"  Unique localbodycodes   : {combined['localbodycode'].nunique():,}")

#%%
# TASK 5 – DEDUPLICATION (priority: FULL MATCH > AMBIGUOUS MATCH > AI PREDICTION)
print("\n" + "=" * 68)
print("  TASK 5 – DEDUPLICATION")
print("=" * 68)

PRIORITY = {"FULL MATCH": 0, "AMBIGUOUS MATCH": 1, "AI PREDICTION": 2}
combined["_priority"] = combined["status"].map(PRIORITY)

pre_dedup = len(combined)
combined = (
    combined
    .sort_values(["localbodycode", "_priority"])
    .drop_duplicates(subset=["localbodycode"], keep="first")
    .drop(columns=["_priority"])
    .reset_index(drop=True)
)
dropped = pre_dedup - len(combined)

if dropped:
    print(f"    Removed {dropped:,} duplicate rows (priority logic applied).")
else:
    print(f"    No duplicates found — all segments cover disjoint municipality sets.")

print(f"  Rows after dedup : {len(combined):,}")

#%%
# TASK 6 – FINAL COLUMN ORDER & TIDY TYPES

# Sort for readability: state → district → localbody
combined = combined.sort_values(
    ["statecode", "districtcode", "localbodycode"],
    na_position="last"
).reset_index(drop=True)

#%%
# TASK 7 – VALIDATION CHECKS
print("\n" + "=" * 68)
print("  TASK 7 – VALIDATION CHECKS")
print("=" * 68)

unique_lbs = combined["localbodycode"].nunique()
dup_check  = combined.duplicated("localbodycode").sum()

print(f"  Expected municipalities  : {EXPECTED_TOTAL:,}")
print(f"  Actual unique LB codes   : {unique_lbs:,}")
print(f"  Match expected total     : {' PASS' if unique_lbs == EXPECTED_TOTAL else '✗ FAIL'}")
print(f"  Duplicate LB codes       : {dup_check} {' PASS' if dup_check == 0 else '✗ FAIL'}")
print(f"  Null districtcode rows   : {combined['districtcode'].isna().sum()}")
print(f"  Null localbody_name rows : {combined['localbody_name'].isna().sum()}")

#%%
# TASK 8 – SAVE
OUT = "final_municipality_district_mapping.csv"
combined.to_csv(OUT, index=False)
print(f"\n  Saved → {OUT}")

#%%
# TASK 9 – SUMMARY
print("\n" + "=" * 68)
print("  TASK 9 – FINAL SUMMARY")
print("=" * 68)

total    = len(combined)
counts   = combined["status"].value_counts()
full_n   = counts.get("FULL MATCH",      0)
ambig_n  = counts.get("AMBIGUOUS MATCH", 0)
ai_n     = counts.get("AI PREDICTION",   0)

# Confidence bands
hi  = (combined["confidence_score"] >= 0.80).sum()
med = ((combined["confidence_score"] >= 0.50) & (combined["confidence_score"] < 0.80)).sum()
lo  = (combined["confidence_score"]  < 0.50).sum()

# Disparity breakdown (ambiguous only)
disp = combined["disparity_category"].value_counts()

# Validation flag breakdown
vflag = combined["validation_flag"].value_counts()

# State-level top 10
state_top = (
    combined.groupby(["statecode", "statename"])
    .size()
    .reset_index(name="municipalities")
    .sort_values("municipalities", ascending=False)
    .head(10)
)

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  FINAL MUNICIPALITY–DISTRICT MAPPING                         │
  ├──────────────────────────────────────────────────────────────┤
  │  Total unique municipalities    : {total:>5,}                    │
  │  Duplicate rows dropped         : {dropped:>5,}                    │
  ├──────────────────────────────────────────────────────────────┤
  │  STATUS BREAKDOWN                                            │
  │    FULL MATCH                   : {full_n:>5,}  ({full_n/total*100:>5.1f}%)           │
  │    AMBIGUOUS MATCH              : {ambig_n:>5,}  ({ambig_n/total*100:>5.1f}%)           │
  │    AI PREDICTION                : {ai_n:>5,}  ({ai_n/total*100:>5.1f}%)           │
  ├──────────────────────────────────────────────────────────────┤
  │  CONFIDENCE SCORE DISTRIBUTION                               │
  │    HIGH   (≥ 0.80)              : {hi:>5,}  ({hi/total*100:>5.1f}%)           │
  │    MEDIUM (0.50 – 0.79)         : {med:>5,}  ({med/total*100:>5.1f}%)           │
  │    LOW    (< 0.50)              : {lo:>5,}  ({lo/total*100:>5.1f}%)           │
  ├──────────────────────────────────────────────────────────────┤
  │  AMBIGUOUS DISPARITY BREAKDOWN ({ambig_n} cases)               │""")

for cat, cnt in disp.items():
    label = str(cat)[:42].ljust(42)
    print(f"  │    {label}: {cnt:>4}   ({cnt/ambig_n*100:>5.1f}%)    │")

print(f"""  ├──────────────────────────────────────────────────────────────┤
  │  VALIDATION FLAGS                                            │""")

for flag, cnt in vflag.items():
    label = str(flag)[:42].ljust(42)
    print(f"  │    {label}: {cnt:>4}   ({cnt/total*100:>5.1f}%)    │")

print(f"""  └──────────────────────────────────────────────────────────────┘""")

print("\n  Top 10 states by municipality count:")
print(state_top.rename(columns={"statecode":"State Code","statename":"State","municipalities":"Municipalities"})
      .to_string(index=False))

print("\n  Sample rows from final dataset:")
preview_cols = ["localbody_name", "statename", "district_name",
                "status", "confidence_score", "disparity_category", "validation_flag"]
print(combined[preview_cols].head(12).to_string(index=False))
#%%