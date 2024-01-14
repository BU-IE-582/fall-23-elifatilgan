import pandas as pd
import json
from datetime import datetime

data_path = "C:/Users/efeme/Desktop/pred/predictions.xlsx"
df = pd.read_excel(data_path)
current_date = str(datetime.now().date())
json_path = "results_" + current_date + ".json"

res = {}
for name in list(df.columns):
    res[name] = list(df[name])

print(res)

with open(json_path, 'w') as json_file:
    json.dump(res, json_file, indent=2)