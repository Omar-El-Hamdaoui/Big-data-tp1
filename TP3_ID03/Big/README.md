# TP3 — Big Data : accidents US avec Hadoop/Spark

**Omar El Hamdaoui — ID03.**

Notebook exécuté : [`us-accidents_big_data.ipynb`](us-accidents_big_data.ipynb). Il contient l’acquisition, l’ingestion HDFS, l’EDA distribuée, les transformations, Spark ML, l’évaluation graphique et l’API locale FastAPI.

## Exécuter

Python 3.12, Java 21 et Hadoop/HDFS configuré sont nécessaires. Installer les dépendances Python :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4 pandas==2.3.3 scipy==1.18.1 scikit-learn==1.7.2 matplotlib==3.11.2 seaborn==0.13.2 joblib==1.6.0 pyspark==4.2.0 kagglehub==1.0.3 ipykernel==7.4.0 fastapi==0.162.5 uvicorn==0.54.0 httpx==0.28.1
.venv/bin/python -m ipykernel install --user --name tp3-big --display-name 'Python (TP3 Big Data)'
```

Démarrer HDFS sans reformater le NameNode, ouvrir le notebook dans Jupyter/VS Code, choisir le noyau ci-dessus et exécuter toutes les cellules. KaggleHub récupère `sobhanmoosavi/us-accidents` à la première exécution; une authentification Kaggle peut être nécessaire.

Le notebook utilise Spark local (`local[2]`) et l’espace HDFS `/user/omar/tp3_us_accidents_v13`; adapter ce chemin à l’utilisateur HDFS de la machine.

Après exécution, démarrer l’API locale dans une nouvelle cellule :

```python
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8002)
```

Documentation : http://127.0.0.1:8002/docs.

Données, caches et modèles générés ne sont pas versionnés.

## Sources

- [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — CC BY-NC-SA 4.0, usage académique.
- Moosavi et al., [A Countrywide Traffic Accident Dataset](https://arxiv.org/abs/1906.05409), 2019.
