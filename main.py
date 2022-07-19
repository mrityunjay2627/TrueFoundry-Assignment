from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

vector = load("tfidfvectorizer.joblib")
model = load("model.joblib")

# class get_review(BaseModel):
#     review :str


@app.get("/")
def read_root():
    return {"TrueFoundry": "Sentiment Analysis"}

@app.get("/prediction")
def get_prediction(review: str):
    airline_review = [review]
    review_vector = vector.transform(airline_review) # TF-IDF vectorizing
    prediction = model.predict(review_vector)
    

    return {"sentence":review,"prediction":prediction}
