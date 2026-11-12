from contextlib import asynccontextmanager
from datetime import date
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from tp3.common import ROOT

class Movie(BaseModel):
    model_config = ConfigDict(extra='forbid')
    release_date: date
    original_language: str = Field(min_length=2, max_length=10)
    overview: str = Field(default='', max_length=20000)

class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    movies: list[Movie] = Field(min_length=1, max_length=100)

@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(ROOT/'artifacts/films/pipeline.joblib')
    app.state.metadata = json.loads((ROOT/'artifacts/films/metadata.json').read_text())
    yield

app = FastAPI(title='TP3 — Estimation des notes de films', version='1.0.0', lifespan=lifespan,
    description='Estimation de la note moyenne dans la population des films retenus (au moins 30 votes).')

@app.get('/health')
def health():
    return {'status':'ok','model':app.state.metadata['model'],'task':'regression'}

@app.post('/predict')
def predict(request: Request):
    records = [m.model_dump(mode='json') for m in request.movies]
    ratings = np.clip(app.state.model.predict(pd.DataFrame(records)),0,10)
    return {'model':app.state.metadata['model'],
            'predictions':[{'estimated_rating':round(float(v),4)} for v in ratings]}
