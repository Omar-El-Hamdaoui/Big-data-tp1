# TP3 — Data Science et Big Data

Ce dossier contient les deux notebooks exécutés. Ils incluent leur code, les graphiques et les sorties affichées. Les fonctions nécessaires sont directement dans chaque notebook : aucun dossier de code auxiliaire n'est requis.

## Small data

`trending-movies-analysis-up-to-2026_small_data.ipynb` analyse le jeu de films avec pandas et scikit-learn.

## Big data

`us-accidents_big_data.ipynb` traite les accidents avec Hadoop/HDFS et Spark, puis classe leur sévérité.

## Exécution

Python 3.12 et Java 21 sont utilisés. Installer les dépendances dans un environnement Python :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4 pandas==2.3.3 scipy==1.18.1 scikit-learn==1.7.2 matplotlib==3.11.2 seaborn==0.13.2 joblib==1.6.0 pyspark==4.2.0 kagglehub==1.0.3 ipykernel==7.4.0 fastapi==0.162.5 uvicorn==0.54.0 httpx==0.28.1
.venv/bin/python -m ipykernel install --user --name tp3 --display-name 'Python (TP3)'
```

Ouvrir le notebook voulu dans Jupyter ou VS Code et choisir le noyau **Python (TP3)**. Au premier lancement, KaggleHub télécharge le dataset; une authentification Kaggle peut être demandée.

La partie accidents nécessite Hadoop/HDFS configuré et démarré avant l'exécution. Avec une installation locale :

```bash
hdfs --daemon start namenode
hdfs --daemon start datanode
hdfs dfsadmin -safemode wait
```

Le notebook utilise Spark en mode local (`local[2]`) et écrit ses fichiers temporaires sous `data/` et dans HDFS. Le chemin HDFS par défaut est `/user/omar/tp3_us_accidents_v13`; l'adapter si nécessaire. Ne pas reformater un NameNode existant.

## API locale

Chaque notebook définit et vérifie sa propre API FastAPI à la fin. Après l'exécution de ses cellules, démarrer le serveur dans une nouvelle cellule avec :

```python
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8001)  # films
```

Pour l'API accidents, utiliser le port `8002`. La documentation interactive sera disponible sur `http://127.0.0.1:8001/docs` ou `http://127.0.0.1:8002/docs`. Les modèles sont créés localement en exécutant les notebooks et ne sont pas stockés sur GitHub.

## Sources

- [Trending Movies Analysis up to 2026](https://www.kaggle.com/datasets/sahilk22/trending-movies-analysis-up-to-2026)
- [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — licence CC BY-NC-SA 4.0, usage académique.
- Moosavi et al., [A Countrywide Traffic Accident Dataset](https://arxiv.org/abs/1906.05409), 2019.
