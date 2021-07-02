import requests
import json

url = 'http://0.0.0.0:8080/api/'

data = [[5.7, 2.8, 4.1, 1.3],[5.8, 2.6, 4., 1.2]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)