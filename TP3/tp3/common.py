from pathlib import Path
import hashlib
import gzip
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'films': {'ref': 'sahilk22/trending-movies-analysis-up-to-2026', 'version': 1,
              'file': 'Treanding_movies.csv', 'folder': 'movies', 'bytes': 3471451},
    'accidents': {'ref': 'sobhanmoosavi/us-accidents', 'version': 13,
                  'file': 'US_Accidents_March23.csv', 'folder': 'accidents', 'bytes': 3058183727},
}

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def acquire(name):
    """Réutiliser la copie locale, sinon télécharger une version précise via kagglehub."""
    spec = DATASETS[name]
    path = ROOT / 'data/raw' / spec['folder'] / spec['file']
    if not path.exists() and path.with_suffix('.csv.gz').exists():
        path = path.with_suffix('.csv.gz')
    if not path.exists():
        os.environ.setdefault('KAGGLEHUB_CACHE', str(ROOT / 'data/kagglehub'))
        import kagglehub
        source = Path(kagglehub.dataset_download(
            f"{spec['ref']}/versions/{spec['version']}", path=spec['file']))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lien physique : pas de deuxième copie du CSV de 3 Go sur le même disque.
        try:
            os.link(source, path)
        except OSError:
            shutil.copyfile(source, path)
    digest = hashlib.sha256()
    total = 0
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            total += len(chunk)
            digest.update(chunk)
    if total != spec['bytes']:
        raise ValueError(f'Fichier incomplet ou version inattendue : {path}')
    save_json({**spec, 'sha256': digest.hexdigest(), 'local_path': str(path.relative_to(ROOT)),
               'stored_bytes': path.stat().st_size, 'compression': 'gzip' if path.suffix == '.gz' else 'none'},
              ROOT / f'reports/provenance_{name}.json')
    return path

def spark_session(name='TP3_Accidents', hdfs=True):
    # Java 21 est utilisé avec Spark 4.2.0. Le Python du noyau est aussi celui des workers.
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    from pyspark.sql import SparkSession
    local = ROOT / 'data/spark_tmp'
    local.mkdir(parents=True, exist_ok=True)
    session = (SparkSession.builder.appName(name)
        .master(os.environ.get('TP3_SPARK_MASTER', 'local[2]'))
        .config('spark.driver.memory', '1536m')
        .config('spark.sql.shuffle.partitions', '8')
        .config('spark.default.parallelism', '8')
        .config('spark.sql.session.timeZone', 'UTC')
        .config('spark.ui.showConsoleProgress', 'false')
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.local.dir', str(local))
        .config('spark.hadoop.fs.defaultFS', os.environ.get('TP3_HDFS_URI', 'hdfs://localhost:19000') if hdfs else 'file:///')
        .config('spark.hadoop.dfs.replication', '1')
        .getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    return session


def local_uri(path):
    """URI Hadoop locale : conserver les espaces, que Path.as_uri encoderait en %20."""
    return 'file://' + str(Path(path).resolve())


def open_dataset(path):
    """Lire du CSV texte, qu'il soit conservé brut ou compressé sans perte."""
    return gzip.open(path, 'rt', newline='') if Path(path).suffix == '.gz' else open(path, newline='')
