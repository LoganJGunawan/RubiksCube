import requests
url = "http://127.0.0.1:5000/post"
data = {'number':[1,2,3,4]}
response = requests.post(url,json=data)
if response.status_code == 200:
    print("Success")
    print(response.json())
else:
    print("Failure")