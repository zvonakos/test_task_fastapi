import requests

url = "http://localhost:8000/proceed"
files = {"file": open("../app/Windows_live_square.JPG", "rb")}
data = {"k": 5}

response = requests.post(url, files=files, data=data)
