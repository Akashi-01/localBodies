# """
# Municipality–District Mapping via Pincode Analysis
# ====================================================
# Inputs:
#   - villages.csv  →  village-level data with districtCode and pincode
#   - Lb.csv        →  local-body data with localBodyCode and pincode

# Logic:
#   FULL MATCH      → all pincodes of a municipality lie within exactly ONE district
#   AMBIGUOUS MATCH → pincodes span MORE THAN ONE district (each district covers some)
#   NO MATCH        → no district covers even a single pincode of the municipality
# """

#%%
import pandas as pd

# 1. LOAD DATA
print("Loading datasets …")

villages = pd.read_csv("villages.csv")
lb       = pd.read_csv("Lb.csv")

print(f"  villages.csv : {len(villages):,} rows")
print(f"  Lb.csv       : {len(lb):,} rows")

#%% 2. CLEAN – drop null pincodes, deduplicate
# Keep only the columns we actually need, then drop nulls and duplicates
villages_clean = (
    villages[["districtCode", "pincode"]]
    .dropna(subset=["districtCode", "pincode"])
    .drop_duplicates()
)

lb_clean = (
    lb[["localBodyCode", "pincode"]]
    .dropna(subset=["localBodyCode", "pincode"])
    .drop_duplicates()
)

# Ensure pincodes are integers (safe conversion – invalid strings → NaN → dropped)
villages_clean["pincode"] = pd.to_numeric(villages_clean["pincode"], errors="coerce")
lb_clean["pincode"]       = pd.to_numeric(lb_clean["pincode"],       errors="coerce")

villages_clean = villages_clean.dropna(subset=["pincode"])
lb_clean       = lb_clean.dropna(subset=["pincode"])

# Convert to int after NaN removal
villages_clean["pincode"]     = villages_clean["pincode"].astype(int)
villages_clean["districtCode"] = villages_clean["districtCode"].astype(int)
lb_clean["pincode"]            = lb_clean["pincode"].astype(int)
lb_clean["localBodyCode"]      = lb_clean["localBodyCode"].astype(int)

print(f"\nAfter cleaning:")
print(f"  Unique district codes : {villages_clean['districtCode'].nunique():,}")
print(f"  Unique local body codes: {lb_clean['localBodyCode'].nunique():,}")

#%% 3. BUILD MAPPING DICTIONARIES
#    district_pincodes : districtCode  → set of pincodes
#    lb_pincodes       : localBodyCode → set of pincodes
print("\nBuilding pincode sets …")

district_pincodes: dict[int, set] = (
    villages_clean
    .groupby("districtCode")["pincode"]
    .apply(set)
    .to_dict()
)

lb_pincodes: dict[int, set] = (
    lb_clean
    .groupby("localBodyCode")["pincode"]
    .apply(set)
    .to_dict()
)

#%% 4. MATCHING LOGIC
print("Running matching logic …")

results = []

for lb_code, lb_pins in lb_pincodes.items():

    # Find every district whose pincode set has ANY overlap with this municipality
    overlapping_districts = [
        dist_code
        for dist_code, dist_pins in district_pincodes.items()
        if lb_pins & dist_pins          # non-empty intersection
    ]

    if not overlapping_districts:
        # ── NO MATCH: no district shares even one pincode ──────────────────
        results.append({
            "localBodyCode":     lb_code,
            "status":            "NO MATCH",
            "matched_district(s)": None
        })

    else:
        # Check if any SINGLE district fully contains all pincodes
        full_match_districts = [
            dist_code
            for dist_code in overlapping_districts
            if lb_pins.issubset(district_pincodes[dist_code])
        ]

        if len(full_match_districts) >= 1:
            # ── FULL MATCH: at least one district covers all pincodes ───────
            # If multiple districts qualify (can happen when districts share
            # identical pincode supersets), we record them all comma-separated.
            results.append({
                "localBodyCode":      lb_code,
                "status":             "FULL MATCH",
                "matched_district(s)": (
                    full_match_districts[0]
                    if len(full_match_districts) == 1
                    else ", ".join(str(d) for d in full_match_districts)
                )
            })

        else:
            # ── AMBIGUOUS MATCH: pincodes straddle multiple districts ───────
            results.append({
                "localBodyCode":      lb_code,
                "status":             "AMBIGUOUS MATCH",
                "matched_district(s)": ", ".join(str(d) for d in overlapping_districts)
            })

#%% 5. BUILD OUTPUT DATAFRAME
result_df = pd.DataFrame(results, columns=["localBodyCode", "status", "matched_district(s)"])

#%% 6. SUMMARY STATISTICS
counts = result_df["status"].value_counts()

full_match_count      = counts.get("FULL MATCH",      0)
ambiguous_match_count = counts.get("AMBIGUOUS MATCH", 0)
no_match_count        = counts.get("NO MATCH",        0)
total                 = len(result_df)

print("       MUNICIPALITY–DISTRICT MAPPING SUMMARY")
print("-" * 50)
print(f"  Total municipalities processed : {total:>6,}")
print(f"  FULL MATCH                     : {full_match_count:>6,}  ({full_match_count/total*100:.1f}%)")
print(f"  AMBIGUOUS MATCH                : {ambiguous_match_count:>6,}  ({ambiguous_match_count/total*100:.1f}%)")
print(f"  NO MATCH                       : {no_match_count:>6,}  ({no_match_count/total*100:.1f}%)")
print("-" * 50)

# Sample rows for each category
for status in ["FULL MATCH", "AMBIGUOUS MATCH", "NO MATCH"]:
    subset = result_df[result_df["status"] == status]
    if not subset.empty:
        print(f"\nSample rows – {status}:")
        print(subset.head(5).to_string(index=False))

#%% 7. SAVE OUTPUT
output_path = "municipality_district_mapping.csv"
result_df.to_csv(output_path, index=False)
# %%
