import requests
import json
key = 'sk-np-0NThAuUGOHbryDU3J0zjez1HJt07TV0ijlzjgWO0pwQ0'


response = requests.post(
    "https://www.neuronpedia.org/api/search-all",
    headers={
      "Content-Type": "application/json",
      "x-api-key": key,
    },
    json={
      "modelId": "gemma-2-2b",
      "sourceSet": "gemmascope-res-16k",
      "text": "mathematics quiz tutoring teaching contestant problem solving thinking reasoning critical analytical contest competition exams formulas equations think focus coding programming calculus differential subtraction addition reasoning high school exam test",
      "selectedLayers": [
        "25-gemmascope-res-16k"
      ],
      "sortIndexes": [],
      "ignoreBos": True,
      "densityThreshold": -1,
      "numResults": 50
    }
)
with open('saes.json', 'w') as fout:
    json.dump(response.json(), fout)