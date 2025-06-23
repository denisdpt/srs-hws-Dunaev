from flask import Flask, request, jsonify
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import os

# стоп-слова при старте
nltk.download('stopwords')
russian_stop_words = stopwords.words('russian')

app = Flask(__name__)

# Загружаем модель эмбеддингов
model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
kw_model = KeyBERT(model=model)

@app.route('/extract', methods=['POST'])
def extract_keywords():
    data = request.get_json(force=True)
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Нет текста для обработки'}), 400

    # Извлекаем 6 ключевых фраз (1–3 слова)
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=russian_stop_words,
        top_n=6,
        use_mmr=True,
        diversity=0.7
    )
    keywords = [kw[0] for kw in keywords]
    return jsonify({'keywords': keywords})

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port)