from pathlib import Path
import json, textwrap, uuid
ROOT=Path(__file__).resolve().parents[1]
def md(s): return {'cell_type':'markdown','metadata':{},'id':uuid.uuid4().hex[:8],'source':textwrap.dedent(s).strip()}
def code(s): return {'cell_type':'code','metadata':{},'id':uuid.uuid4().hex[:8],'source':textwrap.dedent(s).strip(),'execution_count':None,'outputs':[]}
def write(name,cells):
 nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (TP3)','language':'python','name':'tp3'},'language_info':{'name':'python','version':'3.12.3'}},'nbformat':4,'nbformat_minor':5}
 (ROOT/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
setup='''
from pathlib import Path
import sys, os, json, time
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'tp3/common.py').exists())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib','inline')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from tp3.common import acquire, save_json
SEED = 42
sns.set_theme(style='whitegrid', palette='colorblind')
pd.set_option('display.max_columns', 20)
print('Racine :', ROOT)
'''
cells=[md('''
# TP3 — Partie 1 : des données de films à une API de prédiction
**Jeu : Trending Movies Dataset (Up to 2026) — version 1.**

**Question :** peut-on estimer la note moyenne d'un film à partir de sa date de sortie, de sa langue et de son synopsis ?
Nous traitons une **régression supervisée**, sur les films ayant au moins 30 votes et déjà sortis à la date du snapshot.
Le résultat est une estimation de note observée dans ce catalogue, pas une prévision garantie du succès commercial.

Parcours : acquisition → EDA → prétraitement → transformation → comparaison ML → évaluation finale → API locale.
Les cellules sont exécutables dans l'ordre avec le noyau **Python (TP3)**. Installation et commandes : `README.md`.
'''),md('''
## 1. Environnement et acquisition reproductible
Le téléchargement demandé utilise `kagglehub`. Nous figeons la version 1 pour pouvoir reproduire les résultats.
Une copie déjà présente est réutilisée ; taille et empreinte SHA-256 sont enregistrées.
Lors de la préparation initiale, le même fichier a été récupéré par l'API publique Kaggle pendant l'installation des dépendances.
'''),code(setup),code('''
path = acquire('films')
raw = pd.read_csv(path)
print(f'{len(raw):,} lignes, {raw.shape[1]} colonnes ; {path.stat().st_size / 1e6:.2f} Mo')
display(raw.head(3))
display(pd.read_json(ROOT/'reports/provenance_films.json', typ='series').to_frame('Valeur'))
'''),md('''
## 2. EDA : structure, qualité et population
Le fichier contient les notes, les votes, la popularité, les dates, la langue et les synopsis.
Il ne contient pas de colonne de genre : aucune analyse par genre n'est inventée.
L'identifiant `id` est la clé métier. La colonne d'index exportée n'est pas une variable explicative.
'''),code('''
audit = pd.DataFrame({'type':raw.dtypes.astype(str),'manquants':raw.isna().sum(),
                      'uniques':raw.nunique(dropna=True)})
audit['manquants_pct'] = 100 * audit['manquants'] / len(raw)
display(audit)
print('Lignes strictement identiques :', raw.duplicated().sum())
print('Identifiants répétés :', raw['id'].duplicated().sum())
display(raw[['vote_average','vote_count','popularity']].describe().round(3))
'''),code('''
fig,axes=plt.subplots(1,3,figsize=(15,4))
sns.histplot(raw['vote_average'],bins=30,ax=axes[0]); axes[0].set_title('Notes : zéro peut signifier aucun vote')
sns.histplot(np.log1p(raw['vote_count']),bins=35,ax=axes[1]);axes[1].set_title('Nombre de votes (log1p)')
raw['original_language'].value_counts().head(10).plot.bar(ax=axes[2]);axes[2].set_title('10 langues les plus représentées')
plt.tight_layout();plt.show()
'''),md('''
## 3. Prétraitement et périmètre de la cible
Règles fixées **avant le split**, sans apprendre de paramètres sur le test :
- retirer l'index exporté et les doublons de film (première occurrence conservée, sans choisir selon la note) ;
- convertir les dates et les nombres, garder une cible dans [0, 10] ;
- garder les films sortis au plus tard le **22 octobre 2026** (date de publication du snapshot) et ayant **au moins 30 votes**.

Une note nulle sans vote n'est pas une mauvaise critique. Le seuil de 30 réduit cette instabilité, mais restreint
la population : les conclusions ne couvrent pas les films sans audience. Les dates invalides sont exclues car
elles empêchent de vérifier le périmètre. Les synopsis manquants deviendront des chaînes vides dans le pipeline.
'''),code('''
SNAPSHOT = pd.Timestamp('2026-10-22')
df = raw.loc[:, ~raw.columns.str.startswith('Unnamed')].copy()
for col in ['vote_average','vote_count','popularity']:
    df[col] = pd.to_numeric(df[col],errors='coerce')
df['release_date'] = pd.to_datetime(df['release_date'],errors='coerce')
steps=[{'étape':'Brut','lignes':len(df)}]
df=df.drop_duplicates(subset=['id'],keep='first').copy()
steps.append({'étape':'Identifiants uniques','lignes':len(df)})
df=df.loc[df['vote_average'].between(0,10) & (df['vote_count']>=30)
          & df['release_date'].notna() & (df['release_date']<=SNAPSHOT)].copy()
steps.append({'étape':'Notes exploitables et films sortis','lignes':len(df)})
df=df.reset_index(drop=True)
display(pd.DataFrame(steps))
assert df['id'].is_unique and df['vote_average'].notna().all()
quality={'raw_rows':len(raw),'eligible_rows':len(df),'duplicate_ids':int(raw['id'].duplicated().sum())}
save_json(quality,ROOT/'reports/films_quality.json')
'''),code('''
fig,axes=plt.subplots(1,3,figsize=(15,4))
sns.histplot(df['vote_average'],bins=30,ax=axes[0]); axes[0].set_title('Notes après filtrage')
df['release_date'].dt.year.value_counts().sort_index().plot(ax=axes[1]);axes[1].set_title('Années de sortie représentées')
sns.scatterplot(x=np.log1p(df['vote_count']),y=df['vote_average'],s=10,alpha=.2,ax=axes[2]);axes[2].set_title('Votes et notes : relation descriptive');axes[2].set_xlabel('log(1 + nombre de votes)')
plt.tight_layout();plt.show()
display(Markdown(f'Le nettoyage conserve **{len(df):,} films**. La note médiane est **{df.vote_average.median():.2f}/10**. '
                'La surreprésentation de films populaires limite la généralisation à tout le cinéma.'))
'''),md('''
## 4. Séparation apprentissage / validation / test
Split aléatoire reproductible **70 % / 15 % / 15 %**, après déduplication des identifiants.
Ce choix répond à une généralisation à d'autres films du même snapshot. Il ne simule pas une prévision historique :
les notes et synopsis ont été collectés à une seule date, y compris pour les films anciens.

`vote_average` est la cible. `vote_count` sert seulement au périmètre. `popularity`, les votes, le titre, l'ID et
l'index sont exclus des entrées : on n'utilise pas la réception actuelle du public pour prétendre la prévoir.
La colonne `adult`, constante dans ce fichier, n'apporte rien. Le test restera hors du choix de modèle.
'''),code('''
from sklearn.model_selection import train_test_split
from tp3.films import RAW_FEATURES, NUMERIC, CATEGORICAL, movie_features
train_idx,hold_idx=train_test_split(df.index,test_size=.30,random_state=SEED)
valid_idx,test_idx=train_test_split(hold_idx,test_size=.50,random_state=SEED)
X_train,y_train=df.loc[train_idx,RAW_FEATURES],df.loc[train_idx,'vote_average']
X_valid,y_valid=df.loc[valid_idx,RAW_FEATURES],df.loc[valid_idx,'vote_average']
X_test,y_test=df.loc[test_idx,RAW_FEATURES],df.loc[test_idx,'vote_average']
assert not(set(df.loc[train_idx,'id']) & set(df.loc[test_idx,'id']))
assert not(set(df.loc[valid_idx,'id']) & set(df.loc[test_idx,'id']))
display(pd.DataFrame({'jeu':['train','validation','test'],'lignes':[len(train_idx),len(valid_idx),len(test_idx)]}))
# Les identifiants de split permettent de reproduire et auditer la partition.
split=pd.Series('test',index=df.index)
split.loc[train_idx]='train';split.loc[valid_idx]='validation'
df[['id']].assign(split=split).to_csv(ROOT/'reports/films_splits.csv',index=False)
'''),md('''
## 5. Transformation et pipelines sans fuite
La fonction commune au notebook et à l'API extrait année/mois et longueur du synopsis. Elle est déterministe.
Les paramètres appris — médianes, échelles, catégories, vocabulaire TF-IDF — sont ajustés **sur l'apprentissage seulement**.

- **Ridge + TF-IDF** : référence linéaire régularisée exploitant les mots du synopsis (texte limité à 4 000 termes).
- **Random Forest** et **Histogram Gradient Boosting** : interactions non linéaires sur les variables tabulaires.
- **Moyenne** : référence minimale indispensable pour vérifier l'utilité du ML.

On conserve un nombre réduit de configurations : Ridge avec deux régularisations, une forêt et un boosting.
L'absence de normalisation spécifique pour les arbres ne change pas leur ordre de coupure ; le même pipeline
tabulaire standardisé est conservé pour simplifier la comparaison.
'''),code('''
import inspect
print(inspect.getsource(movie_features))
display(movie_features(X_train).head())
'''),code('''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score

def make_pipeline(model,text=False):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=text)
    transforms=[('numeric',numeric,NUMERIC),('language',categorical,CATEGORICAL)]
    if text:
        transforms.append(('synopsis',TfidfVectorizer(max_features=4000,min_df=3,ngram_range=(1,2),sublinear_tf=True),'overview'))
    return Pipeline([('features',FunctionTransformer(movie_features)),
                     ('preprocess',ColumnTransformer(transforms,sparse_threshold=1.0 if text else 0.0)),
                     ('model',model)])

candidates={
    'Moyenne':make_pipeline(DummyRegressor(strategy='mean')),
    'Ridge TF-IDF alpha=10':make_pipeline(Ridge(alpha=10,solver='lsqr'),text=True),
    'Ridge TF-IDF alpha=100':make_pipeline(Ridge(alpha=100,solver='lsqr'),text=True),
    'Random Forest':make_pipeline(RandomForestRegressor(n_estimators=160,max_depth=14,min_samples_leaf=8,n_jobs=2,random_state=SEED)),
    'Histogram Gradient Boosting':make_pipeline(HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=10,learning_rate=.06,random_state=SEED)),
}

def regression_metrics(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':root_mean_squared_error(y,p),'R2':r2_score(y,p)}
'''),md('''
## 6. Apprentissage et choix sur validation
Critère principal : **MAE**, exprimée en points de note et facile à interpréter. RMSE pénalise davantage les grosses erreurs ;
R² compare à une prédiction constante (il peut être négatif). On borne les prédictions à [0, 10] pour tous les modèles,
comme le fera l'API. Aucun résultat de test n'intervient dans cette sélection.
'''),code('''
validation_rows=[]
for name,pipeline in candidates.items():
    begin=time.perf_counter()
    pipeline.fit(X_train,y_train)
    pred=np.clip(pipeline.predict(X_valid),0,10)
    validation_rows.append({'modèle':name,**regression_metrics(y_valid,pred),'secondes':time.perf_counter()-begin})
validation=pd.DataFrame(validation_rows).sort_values(['MAE','RMSE']).reset_index(drop=True)
display(validation.round(4))
validation.to_csv(ROOT/'reports/films_validation.csv',index=False)
best_name=validation.iloc[0]['modèle']
print('Sélection sur validation :',best_name)
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
ordered=validation.sort_values('MAE',ascending=True)
colors=['#d95f02' if name==best_name else '#4c78a8' for name in ordered['modèle']]
axes[0].barh(ordered['modèle'],ordered['MAE'],color=colors)
axes[0].invert_yaxis();axes[0].set(xlabel='MAE sur validation (points / 10)',title='Moins est mieux')
for y,value in enumerate(ordered['MAE']): axes[0].text(value+.006,y,f'{value:.3f}',va='center',fontsize=9)
ordered_r2=validation.sort_values('R2',ascending=True)
colors=['#d95f02' if name==best_name else '#4c78a8' for name in ordered_r2['modèle']]
axes[1].barh(ordered_r2['modèle'],ordered_r2['R2'],color=colors);axes[1].axvline(0,color='black',lw=1)
axes[1].set(xlabel='R² sur validation',title='Plus est mieux ; sous zéro = moyenne battue')
plt.suptitle('Comparaison des modèles — validation seulement',fontweight='bold')
plt.tight_layout();plt.show()
'''),md('''
### Lire la comparaison
Les deux graphiques utilisent **uniquement la validation** : ils justifient le choix avant d'ouvrir le test.
La couleur orange marque le modèle retenu. La référence de prédiction moyenne donne une base lisible ; R² négatif
signifie que la moyenne aurait fait mieux sur cette validation.
'''),md('''
## 7. Réentraînement et évaluation finale
Le modèle choisi est réentraîné sur **train + validation**, sans changer ses hyperparamètres.
Le test est ensuite ouvert une seule fois. Une référence moyenne réentraînée sur le même ensemble est aussi mesurée.
Un bootstrap du test fournit un intervalle indicatif de MAE ; il ne couvre pas l'incertitude du choix de modèle
ni le biais de sélection du catalogue.
'''),code('''
X_fit=pd.concat([X_train,X_valid]);y_fit=pd.concat([y_train,y_valid])
final_model=clone(candidates[best_name]).fit(X_fit,y_fit)
test_pred=np.clip(final_model.predict(X_test),0,10)
baseline_pred=np.full(len(y_test),y_fit.mean())
test_metrics=regression_metrics(y_test,test_pred)
display(pd.DataFrame([{'modèle':best_name,**test_metrics},
                     {'modèle':'Moyenne train+validation',**regression_metrics(y_test,baseline_pred)}]).round(4))
rng=np.random.default_rng(SEED)
errors=np.abs(y_test.to_numpy()-test_pred)
boot=np.array([rng.choice(errors,size=len(errors),replace=True).mean() for _ in range(500)])
ci=np.quantile(boot,[.025,.975])
print(f'IC bootstrap 95 % de la MAE : [{ci[0]:.3f}, {ci[1]:.3f}]')
save_json({'selected_model':best_name,'test':test_metrics,'mae_bootstrap_ci95':ci.tolist(),
           'baseline_test':regression_metrics(y_test,baseline_pred),'n_test':len(y_test)},ROOT/'reports/films_test.json')
'''),code('''
fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].scatter(y_test,test_pred,s=12,alpha=.35);axes[0].plot([0,10],[0,10],'k--');axes[0].set(xlabel='Note observée',ylabel='Note estimée',title='Test : observations et estimations')
sns.histplot(y_test.to_numpy()-test_pred,bins=30,ax=axes[1]);axes[1].set_title('Résidus sur le test')
axes[2].scatter(test_pred,y_test.to_numpy()-test_pred,s=12,alpha=.35);axes[2].axhline(0,color='k',linestyle='--');axes[2].set(xlabel='Note estimée',ylabel='Résidu',title='Biais selon la note')
plt.tight_layout();plt.show()
# Diagnostic par langue, sans modifier le modèle après lecture du test.
diagnostics=pd.DataFrame({'language':X_test.original_language,'abs_error':errors})
language_errors=(diagnostics.groupby('language').agg(n=('abs_error','size'),MAE=('abs_error','mean'))
                .query('n >= 20').sort_values('MAE',ascending=False))
display(language_errors)
show_languages=language_errors.head(12).sort_values('MAE')
fig,ax=plt.subplots(figsize=(9,5.5))
ax.barh(show_languages.index.astype(str),show_languages['MAE'],color='#4c78a8')
ax.set(xlabel='MAE sur le test (points / 10)',title='Erreurs par langue — groupes avec au moins 20 films')
for y,(value,n) in enumerate(zip(show_languages['MAE'],show_languages['n'])):
    ax.text(value+.01,y,f'n={int(n)}',va='center',fontsize=8)
plt.tight_layout();plt.show()
# Ridge permet une visualisation compacte des associations apprises, y compris des mots du synopsis.
if hasattr(final_model.named_steps['model'],'coef_'):
    feature_names=final_model.named_steps['preprocess'].get_feature_names_out()
    coefficients=np.asarray(final_model.named_steps['model'].coef_).ravel()
    coef_table=pd.DataFrame({'variable':feature_names,'coefficient':coefficients})
    top_coef=pd.concat([coef_table.nsmallest(12,'coefficient'),coef_table.nlargest(12,'coefficient')]).drop_duplicates('variable')
    top_coef=top_coef.sort_values('coefficient')
    fig,ax=plt.subplots(figsize=(10,8))
    colors=np.where(top_coef['coefficient']>0,'#d95f02','#4c78a8')
    ax.barh(top_coef['variable'].str.replace('synopsis__','texte: ',regex=False).str.replace('numeric__','num: ',regex=False).str.replace('language__','langue: ',regex=False),top_coef['coefficient'],color=colors)
    ax.axvline(0,color='black',lw=1)
    ax.set(xlabel='Coefficient Ridge (note prédite)',title='24 associations les plus fortes du modèle films')
    plt.tight_layout();plt.show()
    display(top_coef.sort_values('coefficient',key=lambda v:v.abs(),ascending=False).head(16))
'''),md('''
### Interprétation et limites
L'erreur est à comparer à celle du prédicteur moyen. Les caractéristiques disponibles expliquent seulement une partie
de l'appréciation du public : casting, budget, distribution et genres ne sont pas présents.
Une estimation centrale peut masquer de fortes erreurs sur les notes extrêmes.
Le synopsis peut avoir été révisé après sortie ; nous ne revendiquons donc pas une évaluation pré-sortie.
Les diagnostics de test servent à décrire les limites, pas à choisir un nouveau modèle.
'''),code('''
base_mae=mean_absolute_error(y_test,baseline_pred)
gain=100*(base_mae-test_metrics['MAE'])/base_mae
display(Markdown(f'**Résultat mesuré :** {best_name}, MAE **{test_metrics["MAE"]:.3f}/10**, '
                f'RMSE **{test_metrics["RMSE"]:.3f}**, R² **{test_metrics["R2"]:.3f}**. '
                f'Écart relatif de MAE face à la moyenne : **{gain:+.1f} %** de réduction.'))
'''),md('''
## 8. Déploiement : pipeline complet et API FastAPI locale
Le pipeline sérialisé contient les transformations et le modèle, évitant un encodage différent à l'inférence.
L'API reçoit les trois entrées brutes et renvoie une note estimée. Les schémas d'entrée refusent les champs
inconnus et les lots vides. Les scores ne constituent pas une recommandation personnalisée.
'''),code('''
import joblib
out=ROOT/'artifacts/films';out.mkdir(parents=True,exist_ok=True)
joblib.dump(final_model,out/'pipeline.joblib')
metadata={'model':best_name,'target':'vote_average','features':RAW_FEATURES,'snapshot':'2026-10-22',
          'minimum_votes':30,'seed':SEED,'validation_mae':float(validation.iloc[0]['MAE']),
          'test_metrics':test_metrics,'clip':[0,10]}
save_json(metadata,out/'metadata.json')
example=X_test.iloc[0].copy()
example['release_date']=pd.Timestamp(example['release_date']).strftime('%Y-%m-%d')
example['overview']='' if pd.isna(example['overview']) else str(example['overview'])
payload={'movies':[example.to_dict()]}
save_json(payload,ROOT/'api/example_films.json')
loaded=joblib.load(out/'pipeline.joblib')
np.testing.assert_allclose(loaded.predict(X_test.iloc[:10]),final_model.predict(X_test.iloc[:10]))
print('Pipeline sauvegardé, rechargé et prédictions identiques.')
'''),code('''
from fastapi.testclient import TestClient
from api.films_api import app
with TestClient(app) as client:
    response=client.post('/predict',json=payload)
    assert response.status_code==200,response.text
    expected=float(np.clip(final_model.predict(pd.DataFrame(payload['movies']))[0],0,10))
    assert abs(response.json()['predictions'][0]['estimated_rating']-expected)<1e-3
    assert client.post('/predict',json={'movies':[]}).status_code==422
    assert client.post('/predict',json={'movies':[dict(payload['movies'][0],unknown=1)]}).status_code==422
    print('Health :',client.get('/health').json())
    print('Prédiction API :',response.json())
print('Contrat, rechargement et cohérence notebook/API vérifiés.')
'''),md('''
### Lancer et essayer le service
Dans un terminal à la racine du TP3 :
```bash
bash scripts/start_api.sh films
```
Documentation interactive : **http://127.0.0.1:8001/docs**.
Dans un second terminal :
```bash
curl -s http://127.0.0.1:8001/predict -H 'Content-Type: application/json' --data-binary @api/example_films.json
```
Arrêt : `Ctrl+C`. L'exemple JSON provient du test et ne contient pas la note cible.

## 9. Bilan et références
Nous avons construit une chaîne reproductible et mesuré ses limites. Une amélioration future demanderait des
variables plus riches et un nouveau protocole de validation, pas des ajustements sur le test déjà consulté.

- [Dataset original et licence CC0](https://www.kaggle.com/datasets/sahilk22/trending-movies-analysis-up-to-2026)
- [kagglehub : téléchargement versionné](https://github.com/Kaggle/kagglehub)
- [Scikit-learn : pipelines et prévention des fuites](https://scikit-learn.org/1.7/common_pitfalls.html)
- [FastAPI : tests d'application](https://fastapi.tiangolo.com/tutorial/testing/)

Le code de cette partie est original. Les notebooks publics trouvés sur d'autres datasets de films n'ont pas été
utilisés comme source de résultats ni confondus avec ce fichier.
''')]
write('TP3_Partie_1_Films.ipynb',cells)
