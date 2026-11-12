# TP3 — Data Science et Big Data

Le dépôt garde les deux notebooks exécutés, leur code partagé, les API locales et les modèles nécessaires à leur démarrage. Les graphiques et résultats sont déjà visibles dans les notebooks.

## Contenu

- `TP3_Partie_1_Films.ipynb` : analyse et estimation de notes de films avec pandas et scikit-learn.
- `TP3_Partie_2_US_Accidents_Spark.ipynb` : traitement HDFS/Spark et classification de la sévérité des accidents.
- `tp3/` : fonctions utilisées par les notebooks et les API.
- `api/` : deux API FastAPI et leurs requêtes d’exemple.
- `artifacts/` : modèles entraînés, nécessaires aux API pour démarrer sans réentraînement.
- `requirements.txt` : dépendances Python.

Les fichiers de résultats et les images exportées séparément ne sont pas versionnés : les sorties et figures sont intégrées aux notebooks. Les données Kaggle ne sont pas incluses.

## Installation et notebooks

Python 3.12 et Java 21 sont utilisés. Hadoop/HDFS doit être installé et configuré pour la partie Spark.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name tp3 --display-name 'Python (TP3)'
```

Ouvrir les notebooks dans Jupyter ou VS Code et sélectionner le noyau **Python (TP3)**. Au premier lancement, KaggleHub télécharge les jeux de données; un compte Kaggle configuré peut être nécessaire. La partie accidents nécessite HDFS actif avant l’exécution du notebook :

```bash
bash scripts/start_hdfs.sh
```

Le mode Spark est local (`local[2]`) et le notebook écrit les données préparées dans HDFS. Le chemin par défaut est `/user/omar/tp3_us_accidents_v13`; l’adapter si l’utilisateur HDFS est différent. Ne pas reformater un NameNode existant.

## API locales

Depuis le dossier `TP3`, lancer chaque service dans son propre terminal :

```bash
bash scripts/start_api.sh films
bash scripts/start_api.sh accidents
```

| API | Documentation | Santé |
|---|---|---|
| Films | `http://127.0.0.1:8001/docs` | `http://127.0.0.1:8001/health` |
| Accidents | `http://127.0.0.1:8002/docs` | `http://127.0.0.1:8002/health` |

Les modèles `artifacts/` sont conservés pour que les API fonctionnent après clonage, sans exécuter l’entraînement complet. Les données brutes, environnements virtuels et résultats temporaires restent locaux.

## Sources des données

- [Trending Movies Analysis up to 2026](https://www.kaggle.com/datasets/sahilk22/trending-movies-analysis-up-to-2026)
- [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — licence CC BY-NC-SA 4.0, usage académique.
- Moosavi et al., [A Countrywide Traffic Accident Dataset](https://arxiv.org/abs/1906.05409), 2019.
