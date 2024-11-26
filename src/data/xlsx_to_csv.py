import pandas as pd 
  
file = pd.read_excel("./raw/neighbourhood-profiles-2021-158-model.xlsx") 
file.T.to_csv("./raw/neighbourhood_profiles.csv", index = None, header=False) 
