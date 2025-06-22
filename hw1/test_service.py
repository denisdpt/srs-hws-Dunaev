import requests
import json


with open("article.txt", encoding="utf-8") as f:
    sample_text = f.read()

BASE_URL = "http://localhost:8000"

def test_ner(text):
    resp = requests.post(
        f"{BASE_URL}/ner",
        json={"text": text}
    )
    resp.raise_for_status()
    return resp.json()

def test_keywords(text, top_n=5):
    resp = requests.post(
        f"{BASE_URL}/keywords",
        json={"text": text, "top_n": top_n}
    )
    resp.raise_for_status()
    return resp.json()

def test_embedding(text):
    resp = requests.post(
        f"{BASE_URL}/embedding",
        json={"text": text}
    )
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    print("=== NER ===")
    ner_result = test_ner(sample_text)
    print(json.dumps(ner_result, ensure_ascii=False, indent=2))

    print("\n=== Keywords ===")
    kw_result = test_keywords(sample_text, top_n=5)
    print(json.dumps(kw_result, ensure_ascii=False, indent=2))

    print("\n=== Embedding ===")
    emb_result = test_embedding(sample_text)
    print({
        "embedding": emb_result["embedding"][:10],
        "length": len(emb_result["embedding"])
    })
