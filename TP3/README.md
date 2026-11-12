# TP3 — Data Science et Big Data

Deux parties complémentaires, en français, de l'acquisition au déploiement local.

| Livrable | Sujet | Technologie |
|---|---|---|
| `TP3_Partie_1_Films.ipynb` | Estimation de la note moyenne d'un film | pandas / scikit-learn ; distributions, comparaison de modèles, erreurs par langue et coefficients TF-IDF |
| `TP3_Partie_2_US_Accidents_Spark.ipynb` | Classification de l'impact d'un accident sur la circulation | Hadoop/HDFS / Spark ML ; dérive annuelle, compromis accuracy/macro-F1, matrice par classe, carte de densité et coefficients |

## Ouvrir les notebooks sur cette machine

Ouvrir le dossier TP3 dans VS Code/Jupyter et sélectionner le noyau **Python (TP3)**.
Les sorties livrées permettent aussi de lire les résultats sans relancer les calculs.
Chaque notebook contient six figures sauvegardées ; les PNG sont également exportés dans
`reports/visualisations/` pour les intégrer facilement à un rapport ou à une présentation.
Les fichiers `tp3/` doivent rester à côté des notebooks : ils contiennent les transformations partagées avec les API.

Sur cette machine, le noyau **Python (TP3)** utilise un environnement local `.venv`.
Pour une nouvelle installation, les commandes ci-dessous recréent un environnement autonome.

```bash
cd TP3
.venv/bin/python scripts/run_notebook.py TP3_Partie_1_Films.ipynb
bash scripts/start_hdfs.sh
.venv/bin/python scripts/run_notebook.py TP3_Partie_2_US_Accidents_Spark.ipynb
```

Le script d'exécution sauvegarde les sorties même lorsqu'une cellule échoue, pour faciliter le diagnostic.
Il ne crée pas de faux résultats. Les journaux de la préparation sont dans `logs/`.

## Recréer un environnement autonome sur une autre machine

Python 3.12 et Java 21 sont utilisés. Installer Hadoop/HDFS séparément pour la partie 2.
La liste `requirements.txt` fige les versions utilisées ; `requirements-lock.txt` conserve l'inventaire complet.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name tp3 --display-name 'Python (TP3)'
```

Configurer `JAVA_HOME`, `HADOOP_HOME` et `HADOOP_CONF_DIR` pour son installation Hadoop.
HDFS est configuré ici sur `hdfs://localhost:19000` ; la variable `TP3_HDFS_URI` peut modifier cette adresse.
Les données sont écrites uniquement dans `/user/omar/tp3_us_accidents_v13`.
Adapter ce chemin dans le notebook si un autre utilisateur HDFS est requis.
**Ne jamais reformater le NameNode existant pour relancer ce TP.**

Spark fonctionne en `local[2]`, avec une mémoire pilote de 1536 Mo et huit partitions de shuffle.
Ce réglage convient à la machine utilisée ; ce n'est pas un cluster multi-machines.
`TP3_SPARK_MASTER` permet de changer le master. Un vrai cluster exige aussi une configuration
réseau et des dépendances accessibles aux workers ; changer cette seule variable ne suffit pas.

## Données et stockage

- Films : `sahilk22/trending-movies-analysis-up-to-2026`, version 1, `Treanding_movies.csv`.
- Accidents : `sobhanmoosavi/us-accidents`, version 13, `US_Accidents_March23.csv`.
- Les données ne sont pas incluses dans Git. Elles sont téléchargées automatiquement depuis KaggleHub au premier lancement ; un compte Kaggle configuré peut être requis. Les copies locales sont dans `data/raw/`. Le CSV accidents est conservé en `.csv.gz` sans perte, après vérification SHA-256. Spark le lit directement si une réingestion est nécessaire. `tp3.common.acquire` vérifie les octets décompressés et utilise `kagglehub` si la copie manque.
- Empreintes SHA-256 et versions : `reports/provenance_*.json`.
- HDFS : Parquet source, Parquet des variables et prédictions du test. Aucun second CSV brut n'est copié dans HDFS.
- Les archives temporaires de téléchargement sont supprimées après extraction validée.
- Prévoir environ 6 à 8 Go libres pour une installation autonome avec les données, les modèles et les fichiers temporaires.

Le notebook Spark réutilise les jeux Parquet déjà écrits. Après modification du code de transformation,
utiliser un nouveau chemin `FEATURE_PARQUET` (par exemple un suffixe `_v2`) pour ne pas réutiliser
un ancien cache. Les scripts de téléchargement dans `scripts/` documentent la récupération initiale ;
ils ne sont pas nécessaires lorsque les CSV sont déjà présents.

## Protocole ML

**Films** : déduplication par ID ; films sortis au 22 octobre 2026 et au moins 30 votes.
Entrées : date de sortie, langue, synopsis. Split 70/15/15 par film, seed 42.
Comparaison moyenne / Ridge TF-IDF / Random Forest / Histogram Gradient Boosting.
Choix sur MAE validation, réentraînement train+validation, puis test réservé.
Les notes, votes et popularité ne sont pas des entrées du modèle. Ce n'est pas une simulation de prévision pré-sortie.

**Accidents** : EDA et transformations sur les 7 728 394 lignes sources.
Train 2016–2020, validation 2021 entière, test 2022–mars 2023 entier.
L'entraînement utilise 5 % des ID par hachage reproductible, puis le modèle final est ajusté
sur 5 % de 2016–2021. C'est un choix de calcul explicite, pas un remplacement du dataset par un petit fichier.
`TP3_TRAIN_FRACTION=1` permet un ajustement complet, plus coûteux.
Régression logistique et Random Forest avec pondération des classes comparées ; pondération complète/modérée/légère choisie sur macro-F1 de validation. La matrice de confusion, les performances par année, les cartes de densité et les coefficients par classe sont tracés dans le notebook.
Les variables révélant l'issue de l'accident sont exclues ; les mesures météo futures ou trop anciennes sont masquées.
Les temps et effectifs réels sont dans les notebooks et `reports/`.

## Déployer les deux API locales

Exécuter chaque commande dans un terminal distinct, à la racine du TP3 :

```bash
bash scripts/start_api.sh films
bash scripts/start_api.sh accidents
```

| Service | Documentation | Santé |
|---|---|---|
| Films | http://127.0.0.1:8001/docs | http://127.0.0.1:8001/health |
| Accidents | http://127.0.0.1:8002/docs | http://127.0.0.1:8002/health |

```bash
curl -s http://127.0.0.1:8001/predict -H 'Content-Type: application/json' --data-binary @api/example_films.json
curl -s http://127.0.0.1:8002/predict -H 'Content-Type: application/json' --data-binary @api/example_accidents.json
```

Les modèles légers exportés sont inclus dans `artifacts/`, avec le code `tp3/`, afin que les API locales puissent démarrer sans réentraîner les modèles.
L'API accidents ne dépend plus de HDFS et démarre son propre Spark local. Son premier appel est plus lent.
Limiter chaque requête à 100 éléments. Arrêter avec `Ctrl+C`.
Les tests intégrés vérifient le rechargement, la cohérence notebook/API et le refus des entrées invalides.
Le déploiement choisi est **local**, conformément au périmètre demandé ; aucun service en ligne n'est créé.

## Interpréter les résultats

La note des films est une estimation de moyenne, pas une recommandation personnalisée ni une note certaine.
`Severity` décrit les perturbations de circulation ; il ne mesure ni les blessures, ni la probabilité de survenue d'un accident.
Les scores des classifieurs pondérés ne sont pas calibrés comme probabilités de population.
Les changements de fournisseur et de couverture limitent la généralisation du dataset accidents.
Les tableaux par classe et la comparaison à une référence simple sont essentiels.

## Sources

- [Films — dataset et licence CC0](https://www.kaggle.com/datasets/sahilk22/trending-movies-analysis-up-to-2026)
- [US Accidents — CC BY-NC-SA 4.0, usage académique](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- [Moosavi et al., A Countrywide Traffic Accident Dataset, 2019](https://arxiv.org/abs/1906.05409)
- [Moosavi et al., Accident Risk Prediction based on Heterogeneous Sparse Data, 2019](https://arxiv.org/abs/1909.09638)
- Inspiration de présentation : `../TP 2/etude_accidents_pyspark.ipynb` ; ses données australiennes et résultats ne sont pas réutilisés.
