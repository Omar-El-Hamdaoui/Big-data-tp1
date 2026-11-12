from contextlib import asynccontextmanager
from datetime import datetime
import json
import threading
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from tp3.common import ROOT, spark_session, local_uri
from tp3.accidents import engineer, raw_schema

class Accident(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    start_time: datetime = Field(description='Heure locale du lieu, comme dans le CSV source.')
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    state: str = Field(pattern=r'^[A-Z]{2}$')
    weather_timestamp: datetime | None = Field(default=None,description='Mesure antérieure à l’accident de deux heures au maximum ; sinon ignorée.')
    weather_condition: str | None = Field(default=None, max_length=100)
    temperature_f: float | None = Field(default=None,ge=-100,le=150)
    humidity_pct: float | None = Field(default=None,ge=0,le=100)
    pressure_in: float | None = Field(default=None,ge=20,le=35)
    visibility_mi: float | None = Field(default=None,ge=0,le=200)
    wind_speed_mph: float | None = Field(default=None,ge=0,le=250)
    precipitation_in: float | None = Field(default=None,ge=0,le=50)
    junction: bool | None = None
    traffic_signal: bool | None = None
    crossing: bool | None = None
    stop: bool | None = None

class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    accidents: list[Accident] = Field(min_length=1,max_length=100)

MAPPING = {'start_time':'Start_Time','latitude':'Start_Lat','longitude':'Start_Lng','state':'State',
    'weather_timestamp':'Weather_Timestamp','weather_condition':'Weather_Condition','temperature_f':'Temperature(F)','humidity_pct':'Humidity(%)',
    'pressure_in':'Pressure(in)','visibility_mi':'Visibility(mi)','wind_speed_mph':'Wind_Speed(mph)',
    'precipitation_in':'Precipitation(in)','junction':'Junction','traffic_signal':'Traffic_Signal',
    'crossing':'Crossing','stop':'Stop'}

def to_raw(item):
    values=item.model_dump()
    # Conservation de l'heure murale locale ; aucune conversion de fuseau implicite.
    values['start_time']=item.start_time.strftime('%Y-%m-%d %H:%M:%S')
    values['weather_timestamp']=item.weather_timestamp.strftime('%Y-%m-%d %H:%M:%S') if item.weather_timestamp else None
    return {raw: None if values[key] is None else str(values[key]) for key,raw in MAPPING.items()}

@asynccontextmanager
async def lifespan(app):
    from pyspark.ml import PipelineModel
    app.state.spark=spark_session('TP3_API_Accidents',hdfs=False)
    app.state.model=PipelineModel.load(local_uri(ROOT/'artifacts/accidents/pipeline'))
    app.state.metadata=json.loads((ROOT/'artifacts/accidents/metadata.json').read_text())
    app.state.lock=threading.Lock()
    try:
        yield
    finally:
        app.state.spark.stop()

app=FastAPI(title='TP3 — Sévérité des perturbations routières',version='1.0.0',lifespan=lifespan,
    description='Classification conditionnelle à un accident observé. Severity mesure son impact sur la circulation, pas le nombre de blessés.')

@app.get('/health')
def health():
    return {'status':'ok','model':app.state.metadata['model'],'task':'classification','classes':[1,2,3,4]}

@app.post('/predict')
def predict(request: Request):
    from pyspark.sql import functions as F
    # Identifiant ordinal explicite : l'ordre des lignes Spark n'est pas une garantie.
    with app.state.lock:
        raw=[to_raw(item) for item in request.accidents]
        from pyspark.sql.types import StructField,IntegerType
        schema=raw_schema().add(StructField('_request_id',IntegerType(),False))
        df=app.state.spark.createDataFrame([{**r,'_request_id':i} for i,r in enumerate(raw)],schema=schema)
        rows=app.state.model.transform(engineer(df)).select('_request_id','prediction','probability').orderBy('_request_id').collect()
    return {'model':app.state.metadata['model'],'predictions':[
        {'severity':int(r.prediction)+1,'scores':{str(i+1):round(float(v),6) for i,v in enumerate(r.probability)}} for r in rows],
        'score_note':'Scores du modèle pondéré ; probabilités non calibrées.'}
