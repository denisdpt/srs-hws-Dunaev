from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import ru_core_news_sm

nlp = ru_core_news_sm.load()
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
kw_model = KeyBERT(model=embed_model)

app = FastAPI(title="Habr Keyword & Embedding Service")

class TextRequest(BaseModel):
    text: str
    top_n: int = 10

@app.post("/ner")
async def named_entities(req: TextRequest):
    """Выдаёт список уникальных именованных сущностей."""
    doc = nlp(req.text)
    ents = sorted({ent.text for ent in doc.ents})
    return {"entities": ents}

@app.post("/keywords")
async def keywords(req: TextRequest):
    """Извлекает ключевые слова через KeyBERT."""
    kws = kw_model.extract_keywords(req.text, top_n=req.top_n)
    return {"keywords": [kw for kw, score in kws]}

@app.post("/embedding")
async def embedding(req: TextRequest):
    """Возвращает эмбеддинг текста."""
    vec = embed_model.encode(req.text)
    return {"embedding": vec.tolist()}

# uvicorn keyword_service:app --reload --host 0.0.0.0 --port 8000
