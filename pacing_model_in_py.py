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

import warnings
warnings.filterwarnings('ignore')

print('Imports geslaagd')

# %%
# ─────────────────────────────────────────────
#  CONFIGURATIE  –  pas hier aan
# ─────────────────────────────────────────────

# Optie 1: geef een lijst van losse bestanden mee
FIT_FILES = [
    'data/StravaRobin/Midwintermarathon_Apeldoorn.fit',
    'data/StravaRobin/Turnhout_Gravel.fit',
]

# Optie 2: laad een hele map in (zet op None om te deactiveren)
FIT_DIRECTORY = None          # bijv. 'data/StravaRobin'
FILE_EXTENSIONS = ('.fit', '.fit.gz')

# ─────────────────────────────────────────────


# %%
# ─────────────────────────────────────────────
#  HULPFUNCTIES
# ─────────────────────────────────────────────

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
    directory: str | None = None,
    extensions: tuple[str, ...] = ('.fit', '.fit.gz'),
) -> list[str]:
    """
    Combineer een handmatige lijst én een map tot één lijst van unieke paden.
    """
    paths = set()

    if file_list:
        for p in file_list:
            if os.path.isfile(p):
                paths.add(os.path.abspath(p))
            else:
                print(f'[WAARSCHUWING] Bestand niet gevonden, overgeslagen: {p}')

    if directory:
        if os.path.isdir(directory):
            for fname in os.listdir(directory):
                # Controleer op .fit.gz eerst (endswith checkt de langste extensie)
                if any(fname.endswith(ext) for ext in extensions):
                    paths.add(os.path.abspath(os.path.join(directory, fname)))
        else:
            print(f'[WAARSCHUWING] Map niet gevonden: {directory}')

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


# %%
# ─────────────────────────────────────────────
#  BESTANDEN INLADEN
# ─────────────────────────────────────────────

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
# ─────────────────────────────────────────────
#  DATA OPSCHONEN
# ─────────────────────────────────────────────

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
# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

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


df_features = engineer_features(df_all)
print(f'Features beschikbaar: {list(df_features.columns)}')


# %%
# ─────────────────────────────────────────────
#  MODEL TRAINEN
# ─────────────────────────────────────────────

FEATURE_COLS = [col for col in [
    'heart_rate', 'cadence', 'heart_rate_rolling', 'cadence_rolling',
    'hr_drift', 'cadence_trend', 'hr_speed_ratio',
    'cumulative_distance', 'cumulative_power',
    'elapsed_seconds', 'progress'
] if col in df_features.columns]

TARGET_COL  = 'enhanced_speed'
TRAIN_RATIO = 0.90   # eerste 90% van elke activiteit = training
DREMPEL_PCT = -10    # >10% langzamer dan voorspeld = muur

print(f'Gebruikte features: {FEATURE_COLS}')
print(f'Target:             {TARGET_COL}')
print(f'Train/test split:   {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)} per activiteit')


# %%
# ─────────────────────────────────────────────
#  90/10 SPLIT PER ACTIVITEIT
# ─────────────────────────────────────────────
# Train  = eerste 90% van ELKE activiteit gecombineerd
# Test   = laatste 10% van ELKE activiteit afzonderlijk

train_parts = []
test_parts  = {}   # run_id → DataFrame

for run_id, grp in df_features.groupby('run_id'):
    grp      = grp.sort_values('elapsed_seconds').copy()
    cutoff   = grp['elapsed_seconds'].max() * TRAIN_RATIO
    naam     = grp['source_file'].iloc[0]

    trn = grp[grp['elapsed_seconds'] <= cutoff]
    tst = grp[grp['elapsed_seconds'] >  cutoff]

    train_parts.append(trn)
    test_parts[run_id] = tst

    print(f'  run {run_id} ({naam}): '
          f'{len(trn)} train-stappen | {len(tst)} test-stappen '
          f'(split @ {int(cutoff)}s)')

df_train = pd.concat(train_parts, ignore_index=True)
X_train  = df_train[FEATURE_COLS].fillna(0)
y_train  = df_train[TARGET_COL]
print(f'\nTotale trainingsset: {len(X_train)} tijdstappen '
      f'(90% van alle {len(test_parts)} activiteiten gecombineerd)')


# %%
# ─────────────────────────────────────────────
#  MODELLEN TRAINEN  (één model op gecombineerde trainingsset)
# ─────────────────────────────────────────────

modellen = {
    'Lineaire Regressie': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

# Evalueer modellen op de gecombineerde testset (alle laatste 10%)
df_test_all = pd.concat(test_parts.values(), ignore_index=True)
X_test_all  = df_test_all[FEATURE_COLS].fillna(0)
y_test_all  = df_test_all[TARGET_COL]

resultaten = {}
for naam, model in modellen.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_all)
    resultaten[naam] = {
        'model':  model,
        'MAE':    mean_absolute_error(y_test_all, y_pred),
        'RMSE':   np.sqrt(mean_squared_error(y_test_all, y_pred)),
        'R2':     r2_score(y_test_all, y_pred),
    }
    print(f'{naam:25s} → MAE: {resultaten[naam]["MAE"]:.4f} | '
          f'RMSE: {resultaten[naam]["RMSE"]:.4f} | R²: {resultaten[naam]["R2"]:.4f}')

beste_naam  = min(resultaten, key=lambda x: resultaten[x]['MAE'])
beste_model = resultaten[beste_naam]['model']
print(f'\nBeste model: {beste_naam}')


# %%
# ─────────────────────────────────────────────
#  RESULTATEN & MUUR-DETECTIE  (per activiteit)
# ─────────────────────────────────────────────

activiteit_resultaten = {}

for run_id, df_tst in test_parts.items():
    naam   = df_tst['source_file'].iloc[0]
    X_tst  = df_tst[FEATURE_COLS].fillna(0)
    y_pred = beste_model.predict(X_tst)

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
        'df_train':      df_train[df_train['run_id'] == run_id],
        'df_result':     df_res,
        'muur_tijdstap': muur_tijdstap,
    }


# %%
# ─────────────────────────────────────────────
#  VISUALISATIE  (één grafiek per activiteit)
# ─────────────────────────────────────────────

KLEUREN = ['steelblue', 'darkorange', 'mediumpurple', 'seagreen']

for run_id, res in activiteit_resultaten.items():
    kleur        = KLEUREN[(run_id - 1) % len(KLEUREN)]
    naam         = res['naam']
    df_trn       = res['df_train']
    df_res       = res['df_result']
    muur_tijdstap = res['muur_tijdstap']
    cutoff_s     = df_trn['elapsed_seconds'].max()

    fig = go.Figure()

    # Trainingsgedeelte (90%)
    fig.add_trace(go.Scatter(
        x=df_trn['elapsed_seconds'] / 60,
        y=df_trn['enhanced_speed'],
        name='Werkelijk (train 90%)',
        line=dict(color=kleur, width=1.5)
    ))

    # Testgedeelte werkelijk (laatste 10%)
    fig.add_trace(go.Scatter(
        x=df_res['elapsed_seconds'] / 60,
        y=df_res['enhanced_speed'],
        name='Werkelijk (test 10%)',
        line=dict(color='orange', width=1.5)
    ))

    # Voorspelling testgedeelte
    fig.add_trace(go.Scatter(
        x=df_res['elapsed_seconds'] / 60,
        y=df_res['speed_predicted'],
        name='Voorspeld (test 10%)',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Splitlijn
    fig.add_vline(x=cutoff_s / 60, line_dash='dot', line_color='gray',
                  annotation_text='90% split')

    # Muur
    if muur_tijdstap:
        fig.add_vline(x=muur_tijdstap / 60, line_dash='solid', line_color='red',
                      annotation_text='Muur')

    fig.update_layout(
        title=f'Pacing Model – {naam}<br>'
              f'<sup>Model: {beste_naam}  |  sport: {detected_sport}</sup>',
        xaxis_title='Tijd (minuten)',
        yaxis_title='Snelheid (m/s)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=450,
    )
    fig.show()

# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────
if hasattr(beste_model, 'feature_importances_'):
    importances = pd.Series(beste_model.feature_importances_, index=FEATURE_COLS)
    importances.sort_values().plot(kind='barh', figsize=(8, 5),
                                   title=f'Feature Importance – {beste_naam}')
    plt.tight_layout()
    plt.show()
