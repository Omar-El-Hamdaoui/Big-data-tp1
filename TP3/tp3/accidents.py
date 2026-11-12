"""Contrat de variables et transformations Spark disponibles à l'inférence."""
NUMERIC = ['Start_Lat', 'Start_Lng', 'Temperature_F', 'Humidity_pct',
           'Pressure_in', 'Visibility_mi', 'Wind_Speed_mph', 'Precipitation_in',
           'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
           'Junction_num', 'Traffic_Signal_num', 'Crossing_num', 'Stop_num']
CATEGORICAL = ['State', 'Weather_Group']
# Exclusion : End_Time, durée, End_Lat/Lng, Distance(mi), Description et Source.
# Ils décrivent le résultat / l'étendue de l'événement ou le fournisseur de l'étiquette.
RAW_NUMERIC = {'Start_Lat':'Start_Lat','Start_Lng':'Start_Lng',
    'Temperature(F)':'Temperature_F','Humidity(%)':'Humidity_pct','Pressure(in)':'Pressure_in',
    'Visibility(mi)':'Visibility_mi','Wind_Speed(mph)':'Wind_Speed_mph','Precipitation(in)':'Precipitation_in'}
RAW_FEATURES = ['Start_Time', *RAW_NUMERIC, 'State', 'Weather_Condition',
                'Junction', 'Traffic_Signal', 'Crossing', 'Stop', 'Weather_Timestamp']

def raw_schema(include_target=False):
    from pyspark.sql.types import StructType, StructField, StringType
    names = RAW_FEATURES + (['ID', 'Severity'] if include_target else [])
    return StructType([StructField(c, StringType(), True) for c in names])

def engineer(df):
    from pyspark.sql import functions as F
    import math
    df = df.withColumn('_time', F.expr('try_cast(Start_Time as timestamp)'))
    df = df.withColumn('_weather_time', F.expr('try_cast(Weather_Timestamp as timestamp)'))
    weather_ok = (F.col('_weather_time') <= F.col('_time')) & (F.col('_weather_time') >= F.col('_time') - F.expr('INTERVAL 2 HOURS'))
    for name in ['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)',
                 'Wind_Speed(mph)','Precipitation(in)','Weather_Condition']:
        df = df.withColumn(name, F.when(weather_ok, F.col(name)))
    for old, new in RAW_NUMERIC.items():
        df = df.withColumn(new, F.expr(f'try_cast(`{old}` as double)'))
    # Valeurs hors domaine physique représentées comme manquantes (règles fixes).
    bounds = {'Start_Lat':(-90,90),'Start_Lng':(-180,180),'Temperature_F':(-100,150),
              'Humidity_pct':(0,100),'Pressure_in':(20,35),'Visibility_mi':(0,200),
              'Wind_Speed_mph':(0,250),'Precipitation_in':(0,50)}
    for col,(lo,hi) in bounds.items():
        df = df.withColumn(col,F.when(F.col(col).between(lo,hi),F.col(col)))
    for col in ['Junction','Traffic_Signal','Crossing','Stop']:
        df = df.withColumn(col+'_num', F.when(F.lower(F.col(col).cast('string'))=='true',1.0)
                          .when(F.lower(F.col(col).cast('string'))=='false',0.0))
    for prefix,value,period in [('hour',F.hour('_time'),24),('weekday',F.dayofweek('_time')-1,7),
                                ('month',F.month('_time')-1,12)]:
        df = df.withColumn(prefix+'_sin', F.sin(value*(2*math.pi/period)))
        df = df.withColumn(prefix+'_cos', F.cos(value*(2*math.pi/period)))
    weather = F.lower(F.coalesce(F.col('Weather_Condition'), F.lit('unknown')))
    df = df.withColumn('Weather_Group',
        F.when(weather.rlike('thunder|tornado|squall'),'storm')
         .when(weather.rlike('snow|sleet|ice|wintry|hail'),'snow_ice')
         .when(weather.rlike('rain|drizzle'),'rain')
         .when(weather.rlike('fog|mist|haze|smoke'),'low_visibility')
         .when(weather.rlike('cloud|overcast'),'cloudy')
         .when(weather.rlike('fair|clear'),'clear').otherwise('other_unknown'))
    return df.withColumn('State', F.coalesce(F.col('State'),F.lit('unknown')))

def spark_metrics(predictions):
    """Calcul exact de macro-F1 à partir de 16 comptes au maximum ; pas de collect des lignes."""
    import numpy as np
    import pandas as pd
    counts = predictions.groupBy('label','prediction').count().collect()
    cm = np.zeros((4,4),dtype=np.int64)
    for row in counts:
        cm[int(row['label']),int(row['prediction'])] += row['count']
    tp=np.diag(cm); support=cm.sum(axis=1); predicted=cm.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros(4),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(4),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=(precision+recall)!=0)
    table=pd.DataFrame({'Severity':[1,2,3,4],'precision':precision,'recall':recall,'f1':f1,'support':support})
    metrics={'macro_f1':float(f1.mean()),'accuracy':float(tp.sum()/cm.sum()),
             'balanced_accuracy':float(recall.mean()),'weighted_f1':float(np.dot(f1,support)/cm.sum()),
             'n':int(cm.sum())}
    return metrics,table,cm
