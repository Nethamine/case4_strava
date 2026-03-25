## Data inladen & Importeren
import os
import gzip
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from fitparse import FitFile
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import time
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  VOORTGANGSLOGGER  –  print stap + timing
# ─────────────────────────────────────────────
_stap_teller = 0
_stap_starttijd = None
_script_start = time.time()

def stap(beschrijving: str):
    """Print een genummerde stap-header met tijdstempel."""
    global _stap_teller, _stap_starttijd
    # Sluit vorige stap af met duur
    if _stap_starttijd is not None:
        duur = time.time() - _stap_starttijd
        print(f'  ✓ klaar in {duur:.1f}s\n')
    _stap_teller += 1
    _stap_starttijd = time.time()
    elapsed = time.time() - _script_start
    print(f'{"═"*60}')
    print(f'  STAP {_stap_teller}  │  {beschrijving}')
    print(f'         │  t = {elapsed:.1f}s sinds start')
    print(f'{"═"*60}')

def stap_klaar():
    """Sluit de laatste stap af met timing."""
    global _stap_starttijd
    if _stap_starttijd is not None:
        duur = time.time() - _stap_starttijd
        totaal = time.time() - _script_start
        print(f'  ✓ klaar in {duur:.1f}s  (totaal: {totaal:.1f}s)')
        _stap_starttijd = None


stap('Imports laden')
print('  Alle imports geslaagd')

# %%
stap('Configuratie inlezen')

# Optie 1: geef een lijst van losse bestanden mee
FIT_FILES = []

# Optie 2: laad een hele map in (zet op None om te deactiveren)
FIT_DIRECTORY = ['/workspaces/case4_strava/data/StravaJan/activities_dump_download', 
                 '/workspaces/case4_strava/data/StravaJan', 
                 '/workspaces/case4_strava/data/StravaPieter', 
                 '/workspaces/case4_strava/data/StravaPieter/Alle activiteiten', 
                 '/workspaces/case4_strava/data/StravaRobin', 
                 '/workspaces/case4_strava/data/StravaRobin/Alle activiteiten'
                 ]         # bijv. 'data/StravaRobin'
FILE_EXTENSIONS = ('.fit', '.fit.gz')
print(f'  FIT_FILES:     {FIT_FILES}')
print(f'  FIT_DIRECTORY: {FIT_DIRECTORY}')
print(f'  Extensies:     {FILE_EXTENSIONS}')

# ─────────────────────────────────────────────


# %%
stap('Hulpfuncties definiëren')

def _open_fit(filepath: str) -> FitFile:
    """Open een .fit of .fit.gz bestand en retourneer een FitFile object."""
    if filepath.endswith('.gz'):
        tmp = tempfile.NamedTemporaryFile(suffix='.fit', delete=False)
        with gzip.open(filepath, 'rb') as f_in:
            shutil.copyfileobj(f_in, tmp)
        tmp.close()
        return FitFile(tmp.name), tmp.name   # (fitfile, pad_om_later_te_verwijderen)
    return FitFile(filepath), None


def detect_sport(filepath: str) -> str | None:
    """
    Lees de 'sport' uit de session- of activity-berichten van een .fit bestand.
    Geeft een genormaliseerde lowercase string terug, of None als niet gevonden.
    """
    fitfile, tmp_path = _open_fit(filepath)
    sport = None
    try:
        # Probeer eerst 'session' berichten (meest betrouwbaar)
        for msg in fitfile.get_messages(['session', 'sport', 'activity']):
            for field in msg:
                if field.name == 'sport' and field.value:
                    sport = str(field.value).lower().strip()
                    break
            if sport:
                break
    finally:
        if tmp_path:
            os.remove(tmp_path)
    return sport


def load_fit_file(filepath: str) -> pd.DataFrame:
    """
    Lees een .fit of .fit.gz bestand in en retourneer een DataFrame
    met alle beschikbare velden per tijdstap (record messages).
    """
    fitfile, tmp_path = _open_fit(filepath)
    records = []
    try:
        for record in fitfile.get_messages('record'):
            data = {}
            for field in record:
                data[field.name] = field.value
            records.append(data)
    finally:
        if tmp_path:
            os.remove(tmp_path)

    df = pd.DataFrame(records)
    return df


def collect_filepaths(
    file_list: list[str] | None = None,
    directory: str | list[str] | None = None,  # ← ook lijst toegestaan
    extensions: tuple[str, ...] = ('.fit', '.fit.gz'),
) -> list[str]:
    paths = set()

    if file_list:
        for p in file_list:
            if os.path.isfile(p):
                paths.add(os.path.abspath(p))
            else:
                print(f'[WAARSCHUWING] Bestand niet gevonden, overgeslagen: {p}')

    # Normaliseer naar lijst (ook als het een enkele string is)
    directories = [directory] if isinstance(directory, str) else (directory or [])

    for d in directories:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if any(fname.endswith(ext) for ext in extensions):
                    paths.add(os.path.abspath(os.path.join(d, fname)))
        else:
            print(f'[WAARSCHUWING] Map niet gevonden: {d}')

    return sorted(paths)


def find_two_matching_files(
    file_list: list[str] | None = None,
    directory: str | None = None,
    extensions: tuple[str, ...] = ('.fit', '.fit.gz'),
    target_sport: str | None = None,
) -> tuple[list[pd.DataFrame], str]:
    """
    Scan de beschikbare .fit bestanden één voor één totdat er exact twee
    bestanden met dezelfde sport gevonden zijn. Zodra dat het geval is,
    worden die twee direct teruggegeven zonder de rest te verwerken.

    Strategie:
      - Detecteer per bestand de sport (lichtgewicht: leest alleen metadata)
      - Groepeer gevonden sporten in een dict  {sport -> [pad, ...]}
      - Zodra een sport twee paden heeft → laad die twee en stop
      - Als `target_sport` opgegeven is, worden alleen bestanden van die
        sport geteld; het eerste paar wordt teruggegeven

    Geeft terug:
        dataframes  – lijst van precies twee DataFrames
        used_sport  – de gedetecteerde/gebruikte sport
    """
    all_paths = collect_filepaths(file_list, directory, extensions)

    if not all_paths:
        raise FileNotFoundError('Geen geldige .fit bestanden gevonden.')

    print(f'\n{len(all_paths)} bestand(en) gevonden. Zoeken naar twee met dezelfde sport...\n')

    ref_sport = target_sport.lower().strip() if target_sport else None

    # sport → lijst van paden die we al tegengekomen zijn
    sport_buckets: dict[str, list[str]] = {}

    for path in all_paths:
        naam = os.path.basename(path)
        sport = detect_sport(path)
        label = sport if sport else '(onbekend)'
        print(f'  {naam:50s}  →  sport: {label}')

        if sport is None:
            print(f'    [OVERGESLAGEN] sport onbekend')
            continue

        # Als een specifieke sport gevraagd is, sla anderen over
        if ref_sport and sport != ref_sport:
            print(f'    [OVERGESLAGEN] sport "{sport}" ≠ gevraagd "{ref_sport}"')
            continue

        sport_buckets.setdefault(sport, []).append(path)

        # Hebben we nu twee bestanden voor deze sport?
        if len(sport_buckets[sport]) == 2:
            gevonden_sport = sport
            paar = sport_buckets[sport]
            print(f'\n✓ Twee bestanden gevonden voor sport "{gevonden_sport}":')
            for p in paar:
                print(f'    {os.path.basename(p)}')

            # Laad de twee bestanden
            dataframes = []
            for run_id, pad in enumerate(paar, start=1):
                try:
                    df = load_fit_file(pad)
                    df['run_id'] = run_id
                    df['source_file'] = os.path.basename(pad)
                    dataframes.append(df)
                    print(f'[GELADEN]  run {run_id}: {os.path.basename(pad)}  ({len(df)} tijdstappen)')
                except Exception as e:
                    raise RuntimeError(f'Fout bij inladen van {pad}: {e}') from e

            return dataframes, gevonden_sport

    # Geen enkel paar gevonden
    beschikbaar = {s: len(p) for s, p in sport_buckets.items()}
    raise RuntimeError(
        f'Kon geen twee bestanden met dezelfde sport vinden.\n'
        f'Gevonden sporten (aantal bestanden): {beschikbaar}\n'
        f'Voeg meer .fit bestanden toe of pas FIT_DIRECTORY aan.'
    )

print('  Functies gedefinieerd: _open_fit, detect_sport, load_fit_file,')
print('    collect_filepaths, find_two_matching_files')


# %%
stap('FIT-bestanden inladen & sport detecteren')

raw_dataframes, detected_sport = find_two_matching_files(
    file_list=FIT_FILES,
    directory=FIT_DIRECTORY,
    extensions=FILE_EXTENSIONS,
    target_sport=None,    # None = automatisch eerste paar met zelfde sport
                          # bijv. 'running' om specifiek op te filteren
)

if not raw_dataframes:
    raise RuntimeError('Geen bruikbare bestanden geladen. Controleer paden en sportfilter.')


# %%
stap('Data opschonen (filters, interpolatie, timestamps)')

def clean_fit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opschonen van ruwe fit data:
    - Zet timestamp als index
    - Verwijder rijen zonder snelheid of hartslag
    - Filter onrealistische waarden (stilstand, outliers)
    - Interpoleer kleine gaten
    """
    df = df.copy()

    # Bewaar meta-kolommen die niet numeriek zijn
    meta_cols = ['run_id', 'source_file']
    meta = {col: df[col].iloc[0] for col in meta_cols if col in df.columns}

    # Timestamp als index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

    # Verwijder rijen zonder kernvariabelen
    kernvariabelen = [col for col in ['enhanced_speed', 'heart_rate', 'cadence']
                      if col in df.columns]
    df = df.dropna(subset=kernvariabelen)

    # Verwijder stilstand (snelheid < 0.5 m/s ≈ 1.8 km/h)
    if 'enhanced_speed' in df.columns:
        df = df[df['enhanced_speed'] > 0.5]

    # Filter onrealistische hartslag
    if 'heart_rate' in df.columns:
        df = df[(df['heart_rate'] > 40) & (df['heart_rate'] < 220)]

    # Interpoleer kleine gaten (max 5 seconden), alleen numerieke kolommen
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].interpolate(method='time', limit=5)

    # Herstel meta-kolommen
    for col, val in meta.items():
        df[col] = val

    # Reset naar numerieke index (seconden vanaf start)
    df['elapsed_seconds'] = (df.index - df.index[0]).total_seconds().astype(int)
    df = df.reset_index()

    return df


clean_dataframes = []
for df_raw in raw_dataframes:
    naam = df_raw['source_file'].iloc[0]
    df_c = clean_fit_data(df_raw)
    df_c['duration'] = df_c['elapsed_seconds']
    df_c['progress'] = df_c['duration'] / df_c['duration'].max()
    clean_dataframes.append(df_c)
    print(f'{naam}: {len(df_raw)} → {len(df_c)} rijen na opschonen')

df_all = pd.concat(clean_dataframes, ignore_index=True)
print(f'\nTotaal gecombineerd: {len(df_all)} tijdstappen uit {len(clean_dataframes)} activiteiten')


# %%
stap('Feature engineering (per activiteit, rolling windows)')

def engineer_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Bereken vermoeidheids-features op basis van de tijdreeks.
    window: grootte van het rolling venster in seconden
    """
    df = df.copy()

    # Cumulatieve features
    if 'distance' in df.columns:
        df['cumulative_distance'] = df['distance']
    elif 'enhanced_speed' in df.columns:
        df['cumulative_distance'] = df['enhanced_speed'].cumsum()

    if 'power' in df.columns:
        df['cumulative_power'] = df['power'].cumsum()

    # Rolling gemiddelden
    for col in ['enhanced_speed', 'heart_rate', 'cadence', 'power']:
        if col in df.columns:
            df[f'{col}_rolling'] = df[col].rolling(window=window, min_periods=1).mean()

    # Hartslag-drift (stijgt dit → drift/vermoeidheid)
    if 'heart_rate' in df.columns and 'enhanced_speed' in df.columns:
        df['hr_speed_ratio'] = df['heart_rate'] / (df['enhanced_speed'] + 1e-5)
        df['hr_drift'] = df['hr_speed_ratio'].rolling(window=window, min_periods=1).mean()

    # Cadans-trend (afname over tijd)
    if 'cadence' in df.columns:
        df['cadence_trend'] = df['cadence'].rolling(window=window, min_periods=1).mean()

    # Relatieve positie in activiteit (0.0 – 1.0)
    df['progress'] = df['elapsed_seconds'] / df['elapsed_seconds'].max()

    return df


# ── FIX: Features PER activiteit berekenen ──────────────────
# Rolling features (hr_drift, cadence_trend, cumsum) mogen niet
# over de grens van twee activiteiten heen lopen. Daarom berekenen
# we ze per run_id en voegen ze daarna weer samen.

feature_parts = []
for run_id, grp in df_all.groupby('run_id'):
    grp_feat = engineer_features(grp)
    feature_parts.append(grp_feat)
    print(f'  run {run_id}: {len(grp_feat)} rijen, '
          f'{len([c for c in grp_feat.columns if "_rolling" in c or "drift" in c or "trend" in c])} rolling features')

df_features = pd.concat(feature_parts, ignore_index=True)
print(f'Features beschikbaar: {list(df_features.columns)}')


# %%
stap('Model configuratie')

FEATURE_COLS = [col for col in [
    'heart_rate', 'cadence', 'heart_rate_rolling', 'cadence_rolling',
    'hr_drift', 'cadence_trend', 'hr_speed_ratio',
    'cumulative_distance', 'cumulative_power',
    'elapsed_seconds', 'progress'
] if col in df_features.columns]

TARGET_COL  = 'enhanced_speed'
DREMPEL_PCT = -10    # >10% langzamer dan voorspeld = muur

print(f'Gebruikte features: {FEATURE_COLS}')
print(f'Target:             {TARGET_COL}')
print(f'Validatie:          Leave-one-activity-out cross-validation')


# %%
stap('Leave-one-activity-out cross-validation')
# Per fold: train op ALLE andere activiteiten, test op de
# volledige doelactiviteit. Dit voorkomt data leakage: het model
# heeft nooit data van de te voorspellen activiteit gezien.

modellen_def = {
    'Lineaire Regressie': lambda: Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Random Forest':      lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':  lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
}

alle_run_ids = sorted(df_features['run_id'].unique())
print(f'Leave-one-out CV over {len(alle_run_ids)} activiteiten\n')

# Verzamel scores per model over alle folds
cv_scores = {naam: {'MAE': [], 'RMSE': [], 'R2': []} for naam in modellen_def}

# Bewaar per fold de beste voorspellingen voor visualisatie later
fold_resultaten = {}   # run_id → dict met voorspellingen

for test_run_id in alle_run_ids:
    # Split: alles behalve deze activiteit = train
    df_train_fold = df_features[df_features['run_id'] != test_run_id].copy()
    df_test_fold  = df_features[df_features['run_id'] == test_run_id].copy()

    naam_test = df_test_fold['source_file'].iloc[0]
    print(f'─── Fold: test op run {test_run_id} ({naam_test}) ───')
    print(f'    Train: {len(df_train_fold)} stappen van '
          f'{df_train_fold["run_id"].nunique()} andere activiteit(en)')
    print(f'    Test:  {len(df_test_fold)} stappen')

    X_train_fold = df_train_fold[FEATURE_COLS].fillna(0)
    y_train_fold = df_train_fold[TARGET_COL]
    X_test_fold  = df_test_fold[FEATURE_COLS].fillna(0)
    y_test_fold  = df_test_fold[TARGET_COL]

    fold_model_scores = {}
    for model_naam, model_factory in modellen_def.items():
        model = model_factory()          # vers model per fold
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        mae  = mean_absolute_error(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        r2   = r2_score(y_test_fold, y_pred)

        cv_scores[model_naam]['MAE'].append(mae)
        cv_scores[model_naam]['RMSE'].append(rmse)
        cv_scores[model_naam]['R2'].append(r2)

        fold_model_scores[model_naam] = {
            'model': model, 'MAE': mae, 'y_pred': y_pred
        }
        print(f'    {model_naam:25s} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}')

    # Bewaar voorspellingen van het beste model in deze fold
    beste_in_fold = min(fold_model_scores, key=lambda m: fold_model_scores[m]['MAE'])
    fold_resultaten[test_run_id] = {
        'naam':       naam_test,
        'df_test':    df_test_fold,
        'y_pred':     fold_model_scores[beste_in_fold]['y_pred'],
        'model_naam': beste_in_fold,
        'model':      fold_model_scores[beste_in_fold]['model'],
    }
    print()


# %%
stap('CV-resultaten samenvatten')

print('═══ Gemiddelde scores over alle folds ═══')
for model_naam, scores in cv_scores.items():
    avg_mae  = np.mean(scores['MAE'])
    avg_rmse = np.mean(scores['RMSE'])
    avg_r2   = np.mean(scores['R2'])
    print(f'{model_naam:25s} → MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | R²: {avg_r2:.4f}')

beste_naam = min(cv_scores, key=lambda m: np.mean(cv_scores[m]['MAE']))
print(f'\nBeste model (laagste gemiddelde MAE): {beste_naam}')


# %%
stap('Muur-detectie per activiteit')

activiteit_resultaten = {}

for run_id, fold in fold_resultaten.items():
    naam    = fold['naam']
    df_tst  = fold['df_test']
    y_pred  = fold['y_pred']

    df_res = df_tst[['elapsed_seconds', 'enhanced_speed',
                      'heart_rate', 'cadence', 'source_file']].copy()
    df_res['speed_predicted'] = y_pred
    df_res['afwijking']       = df_res['enhanced_speed'] - df_res['speed_predicted']
    df_res['afwijking_pct']   = (df_res['afwijking'] / df_res['speed_predicted']) * 100

    # Muur-detectie
    muur_kandidaten = df_res[df_res['afwijking_pct'] < DREMPEL_PCT]
    if not muur_kandidaten.empty:
        muur_tijdstap = muur_kandidaten.iloc[0]['elapsed_seconds']
        muur_minuut   = int(muur_tijdstap // 60)
        print(f'[{naam}] Muur gedetecteerd rond minuut {muur_minuut} ({int(muur_tijdstap)}s)')
    else:
        muur_tijdstap = None
        print(f'[{naam}] Geen duidelijke muur – consistent pacing!')

    activiteit_resultaten[run_id] = {
        'naam':          naam,
        'df_full':       df_tst,         # volledige activiteit (was testset in deze fold)
        'df_result':     df_res,
        'muur_tijdstap': muur_tijdstap,
        'model_naam':    fold['model_naam'],
    }


# %%
stap('Visualisatie genereren (Plotly grafieken)')

KLEUREN = ['steelblue', 'darkorange', 'mediumpurple', 'seagreen']

for run_id, res in activiteit_resultaten.items():
    kleur         = KLEUREN[(run_id - 1) % len(KLEUREN)]
    naam          = res['naam']
    df_res        = res['df_result']
    muur_tijdstap = res['muur_tijdstap']
    model_naam    = res['model_naam']

    fig = go.Figure()

    # Werkelijke snelheid (hele activiteit)
    fig.add_trace(go.Scatter(
        x=df_res['elapsed_seconds'] / 60,
        y=df_res['enhanced_speed'],
        name='Werkelijk',
        line=dict(color=kleur, width=1.5)
    ))

    # Voorspelling (getraind op ANDERE activiteiten)
    fig.add_trace(go.Scatter(
        x=df_res['elapsed_seconds'] / 60,
        y=df_res['speed_predicted'],
        name='Voorspeld (leave-one-out)',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Muur
    if muur_tijdstap:
        fig.add_vline(x=muur_tijdstap / 60, line_dash='solid', line_color='red',
                      annotation_text='Muur')

    fig.update_layout(
        title=f'Pacing Model – {naam}<br>'
              f'<sup>Model: {model_naam} (leave-one-out)  |  sport: {detected_sport}</sup>',
        xaxis_title='Tijd (minuten)',
        yaxis_title='Snelheid (m/s)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=450,
    )
    fig.show()
    print(f'  Grafiek {run_id}/{len(activiteit_resultaten)}: {naam}')

# %%
stap('Feature importance berekenen')
# Train een finaal model op ALLE data voor de feature importance plot.
# (De CV-scores hierboven geven de eerlijke generalisatie-meting.)

finaal_model_def = modellen_def[beste_naam]
finaal_model = finaal_model_def()
X_all = df_features[FEATURE_COLS].fillna(0)
y_all = df_features[TARGET_COL]
finaal_model.fit(X_all, y_all)

# Haal het onderliggende model op (bij Pipeline zit het in .named_steps)
_model_voor_importance = (finaal_model.named_steps['model']
                          if hasattr(finaal_model, 'named_steps')
                          else finaal_model)

if hasattr(_model_voor_importance, 'feature_importances_'):
    importances = pd.Series(_model_voor_importance.feature_importances_, index=FEATURE_COLS)
    importances.sort_values().plot(kind='barh', figsize=(8, 5),
                                   title=f'Feature Importance – {beste_naam} (alle data)')
    plt.tight_layout()
    plt.show()
elif hasattr(_model_voor_importance, 'coef_'):
    importances = pd.Series(np.abs(_model_voor_importance.coef_), index=FEATURE_COLS)
    importances.sort_values().plot(kind='barh', figsize=(8, 5),
                                   title=f'Coëfficiënten (abs) – {beste_naam} (alle data)')
    plt.tight_layout()
    plt.show()

# %%
stap_klaar()
totaal = time.time() - _script_start
print(f'\n{"═"*60}')
print(f'  SCRIPT VOLTOOID  │  {_stap_teller} stappen in {totaal:.1f}s')
print(f'{"═"*60}')