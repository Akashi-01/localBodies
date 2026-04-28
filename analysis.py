#%%
import pandas as pd

# -------- Load Data --------
# Dataset 1: Local Body + Pincode mapping
df_local = pd.read_csv("Lb.csv")

# Dataset 2: Village dataset
df_village = pd.read_csv("villages.csv")


#%% -------- Pincode → District Mapping --------
pincode_district = (
    df_village.groupby("pincode")["districtCode"]
    .nunique()
    .reset_index()
)

#%% Unique district mapping (pincode -> only 1 district)
unique_pincode_district = set(
    pincode_district[pincode_district["districtCode"] == 1]["pincode"]
)
non_unique_pincode_district = set(
    pincode_district[pincode_district["districtCode"] > 1]["pincode"]
)


#%% --------Pincode → Local Body Mapping --------
pincode_localbody = (
    df_local.groupby("pincode")["localBodyCode"]
    .nunique()
    .reset_index()
)

#%% Unique local body mapping (only for already filtered pincodes)
unique_pincode_localbody = set(
    pincode_localbody[
        (pincode_localbody["pincode"].isin(unique_pincode_district)) &
        (pincode_localbody["localBodyCode"] == 1)
    ]["pincode"]
)


#%% -------- Final Matches (Extract Subdistrict etc.) --------
final_matches = df_village[
    df_village["pincode"].isin(unique_pincode_localbody)
][["pincode", "districtCode","districtNameEnglish", "subdistrictCode", "subdistrictNameEnglish", "villageCode" ,"villageNameEnglish"]]


#%% -------- Multi-mapping Cases --------
all_pincodes = set(df_village["pincode"].unique())

multi_mapping_cases = all_pincodes - unique_pincode_localbody


#%% -------- Output --------
print("Total one-to-one matches:", len(final_matches))
print("Total problematic (multi-mapping) pincodes:", len(multi_mapping_cases))

#%% Save results
final_matches.to_csv("one_to_one_matches.csv", index=False)
pd.DataFrame({"pincode": list(multi_mapping_cases)}).to_csv("multi_mapping_cases.csv", index=False)

# -------- STATISTICS REPORT --------

total_pincodes = df_village["pincode"].nunique()

one_to_one_count = len(unique_pincode_localbody)
multi_mapping_count = total_pincodes - one_to_one_count

one_to_one_pct = (one_to_one_count / total_pincodes) * 100
multi_pct = (multi_mapping_count / total_pincodes) * 100

# District issue
district_issue = len(
    pincode_district[pincode_district["districtCode"] > 1]
)

# Local body issue
localbody_issue = len(
    pincode_localbody[pincode_localbody["localBodyCode"] > 1]
)

print("\n Pincode Mapping Analysis Report\n")

print(f"Total pincodes analyzed: {total_pincodes}")

print(f"\n One-to-one mappings:")
print(f"Count: {one_to_one_count}")
print(f"Percentage: {one_to_one_pct:.2f}%")

print(f"\n Multiple mapping cases:")
print(f"Count: {multi_mapping_count}")
print(f"Percentage: {multi_pct:.2f}%")

print(f"\n Breakdown:")
print(f"Pincodes with multiple districts: {district_issue}")
print(f"Pincodes with multiple local bodies: {localbody_issue}")
# %%
