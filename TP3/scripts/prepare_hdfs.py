from pathlib import Path
import sys,csv,time,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tp3.common import spark_session,save_json,local_uri,acquire,open_dataset,DATASETS
from pyspark.sql import functions as F,types as T
path=acquire('accidents')
spark=spark_session('TP3_Ingestion')
uri='hdfs://localhost:19000/user/omar/tp3_us_accidents_v13/raw_parquet'
fs=spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
if not fs.exists(spark._jvm.org.apache.hadoop.fs.Path(uri+'/_SUCCESS')):
    with open_dataset(path) as f: cols=next(csv.reader(f))
    schema=T.StructType([T.StructField(c,T.StringType(),True) for c in cols])
    begin=time.perf_counter()
    raw=(spark.read.schema(schema).option('header',True).option('escape','"').option('mode','FAILFAST').csv(local_uri(path)))
    print('Ingestion started',flush=True)
    (raw.withColumn('year',F.year(F.expr('try_cast(Start_Time as timestamp)'))).coalesce(8)
        .write.mode('overwrite').partitionBy('year').parquet(uri))
    seconds=time.perf_counter()-begin
    size=fs.getContentSummary(spark._jvm.org.apache.hadoop.fs.Path(uri)).getLength()
    save_json({'seconds':seconds,'csv_bytes':DATASETS['accidents']['bytes'],'parquet_bytes':size},root/'reports/accidents_ingestion.json')
    print('INGESTED',seconds,size,flush=True)
print('Rows',spark.read.parquet(uri).count(),flush=True)
spark.stop()
