import requests

res = requests.post("http://localhost:8000/api/search", data={
    "text_query": "man",
    "limit": 5
})
print(res.status_code)
print(res.text)
