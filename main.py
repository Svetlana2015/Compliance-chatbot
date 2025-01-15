# Data Handling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Charger les données et le modèle
data = [
    { 
        "question": "Est-ce que je peux inviter un médecin à dîner ?", 
        "reponse": "Oui, mais seulement sous certaines conditions...",
        "aliases": ["Est-ce que je peux inviter un docteur a dîner ?", "Est-ce que je peux inviter un doctor à dîner ?", "Est-ce que je peux inviter un médecin à manger ?"]
    },
    { 
        "question": "Puis-je offrir un cadeau à un médecin ou à un professionnel de santé ?", 
        "reponse": "Les cadeaux doivent être modestes...",
        "aliases": ["Puis-je offrir un cadeau à un doctor ou à un professionnel de santé ?", "Puis-je offrir un petit cadeau à un doctor ou à un professionnel de santé ?"]
    },
    { 
        "question": "Est-il acceptable de financer la participation d’un médecin à un congrès ?", 
        "reponse": "Oui, cela est possible, mais cela doit être clairement lié à une activité scientifique ou professionnelle...",
        "aliases": ["non"]
    }
]
df = pd.DataFrame(data)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prétraitement
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    stop_words = set(stopwords.words('french'))
    words = sentence.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['question_normalized'] = df['question'].apply(preprocess_text)
df['aliases_normalized'] = df['aliases'].apply(lambda x: [preprocess_text(alias) for alias in x])

# Calcul de similarité
def find_similar_question(user_question, faq_df, model, threshold=0.95):
    user_question_normalized = preprocess_text(user_question)
    user_embedding = model.encode(user_question_normalized)
    faq_df['similarity'] = faq_df.apply(
        lambda row: max(
            util.pytorch_cos_sim(user_embedding, model.encode(row['question_normalized'])).item(),
            *[util.pytorch_cos_sim(user_embedding, model.encode(alias)).item() for alias in row['aliases_normalized']]
        ),
        axis=1
    )
    best_match = faq_df.loc[faq_df['similarity'].idxmax()]
    if best_match['similarity'] >= threshold:
        return best_match['reponse']
    else:
        return "Je n'ai pas de réponse exacte pour cette question, pouvez-vous reformuler ?"

# Route racine
@app.get("/")
async def root():
    return {"message": "Bienvenue dans l'API FAQ. Utilisez /faq/ pour poser vos questions."}

# Modèle pour la requête
class FAQRequest(BaseModel):
    question: str

# Route pour la FAQ
@app.post("/faq/")
async def get_faq_answer(request: FAQRequest):
    try:
        answer = find_similar_question(request.question, df, model)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
