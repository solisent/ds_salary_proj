#making request from within python
import requests 
from FlaskAPI.data_input import data_in #since current wd is @parent of FlaskAPI

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

r = requests.get(URL,headers=headers, json=data) 

r.json()