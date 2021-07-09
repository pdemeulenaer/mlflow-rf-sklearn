import requests
import json
import pandas as pd

url = 'http://192.168.0.143:8080/api'
#url = 'http://mlflow-rf-sklearn-pdemeulenaer-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/api/'

# data = [[5.7, 2.8, 4.1, 1.3],[5.8, 2.6, 4., 1.2],[5.8, 2.6, 4., 1.2]]
# j_data = json.dumps(data)
# headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
# r = requests.post(url, data=j_data, headers=headers)
# print(r, r.text)

r = requests.get(url)
# print(r.text)
# print(r.json())
j = r.json()

#df = pd.DataFrame([[d['v'] for d in x['c']] for x in j['rows']], columns=[d['label'] for d in j['cols']])
df = pd.DataFrame.from_dict(j)

print(df.head())                  