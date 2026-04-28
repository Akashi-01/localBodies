# import os 
# import pandas as pd
# import json
# data_folder = "C:/Users/Lenovo/OneDrive/Desktop/.py/localBodies/"
# dataframes = []
# for filename in os.listdir(data_folder):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(data_folder, filename)
#         df = pd.read_csv(file_path)
#         # df_clean = df.drop_duplicates()
#         # df_clean.to_csv(file_path, index = False)
        
#         dataframes.append(df)
        
#         final_df = pd.concat(dataframes , ignore_index=True)
#         final_df.to_csv('indiaLocations.csv', index=False)
