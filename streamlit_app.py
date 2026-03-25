"""
Strava Pacing Model – Streamlit App
Gebruik: streamlit run pacing_model_streamlit.py
"""

import os
import gzip
import hashlib
import shutil
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fitparse import FitFile
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
#  PAGINA-CONFIGURATIE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pacing Model",
    page_icon=":running:",
    layout="wide",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS  –  sportief donker thema
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;800&family=Barlow:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}
.hero-header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1f2e 50%, #0d1a2e 100%);
    border-bottom: 2px solid #00c8ff;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #ffffff;
    line-height: 1;
    margin: 0;
}
.hero-subtitle {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 400;
    font-size: 1.1rem;
    color: #00c8ff;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.metric-card {
    background: #161b27;
    border: 1px solid #252d3d;
    border-top: 3px solid #00c8ff;
    border-radius: 4px;
    padding: 1rem 1.5rem;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7a99;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.metric-unit { font-size: 0.8rem; color: #6b7a99; margin-left: 0.3rem; }
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #00c8ff;
    border-left: 3px solid #00c8ff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem;
}
.muur-badge {
    display: inline-block;
    background: #3d1515;
    border: 1px solid #c0392b;
    border-radius: 3px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #e74c3c;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.geen-muur-badge {
    display: inline-block;
    background: #0d2e1a;
    border: 1px solid #27ae60;
    border-radius: 3px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #2ecc71;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.upload-hint {
    text-align: center;
    color: #4a5568;
    font-size: 0.9rem;
    padding: 2rem;
    border: 1px dashed #252d3d;
    border-radius: 6px;
    background: #10131c;
}
.cache-hit {
    display: inline-block;
    background: #0d2a1a;
    border: 1px solid #1a6b3a;
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    color: #2ecc71;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] {
    background-color: #10131c;
    border-right: 1px solid #1e2535;
}
.stButton > button {
    background: #00c8ff;
    color: #0d0f14;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 3px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #33d4ff; color: #0d0f14; }
div[data-testid="stFileUploader"] {
    background: #10131c;
    border: 1px dashed #252d3d;
    border-radius: 6px;
    padding: 0.5rem;
}
.stProgress > div > div { background-color: #00c8ff !important; }
.stSelectbox > div > div { background-color: #161b27; border-color: #252d3d; }
hr { border-color: #1e2535; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HERO HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Pacing Model</div>
    <div class="hero-subtitle">Strava FIT-bestand analyse &nbsp;·&nbsp; Muur-detectie &nbsp;·&nbsp; Leave-one-out CV</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  HULPFUNCTIES  (low-level, niet gecached)
# ──────────────────────────────────────────────

def _open_fit(filepath: str):
    if filepath.endswith('.gz'):
        tmp = tempfile.NamedTemporaryFile(suffix='.fit', delete=False)
        with gzip.open(filepath, 'rb') as f_in:
            shutil.copyfileobj(f_in, tmp)
        tmp.close()
        return FitFile(tmp.name), tmp.name
    return FitFile(filepath), None


def _detect_sport_from_path(filepath: str) -> str | None:
    fitfile, tmp_path = _open_fit(filepath)
    sport = None
    try:
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


def _load_records_from_path(filepath: str) -> pd.DataFrame:
    fitfile, tmp_path = _open_fit(filepath)
    records = []
    try:
        for record in fitfile.get_messages('record'):
            data = {field.name: field.value for field in record}
            records.append(data)
    finally:
        if tmp_path:
            os.remove(tmp_path)
    return pd.DataFrame(records)


# ──────────────────────────────────────────────
#  LAAG 1 CACHE: per bestand  →  schone DataFrame
#  Sleutel: bestandsnaam + sha256 van bytes
#  Vervalt nooit zolang de app draait (ttl=None)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def parse_uploaded_file(filename: str, file_hash: str, file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    """
    Schrijf bytes naar tmp-bestand, lees FIT, schoon op.
    Gecached op (filename, file_hash) – zelfde bestand = gratis.
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, 'wb') as f:
        f.write(file_bytes)

    sport = _detect_sport_from_path(tmp_path)
    df    = _load_records_from_path(tmp_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if df.empty:
        return df, (sport or 'onbekend')

    # Opschonen
    df = _clean_fit_data(df)

    # Gooi lege of te kleine activiteiten weg (minder dan 30 rijen = onbruikbaar)
    if len(df) < 30:
        return pd.DataFrame(), (sport or 'onbekend')

    return df, (sport or 'onbekend')


def _clean_fit_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Timestamp als index — alleen als de kolom aanwezig en parseerbaar is
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp').sort_index()

    # Verwijder rijen zonder kernvariabelen
    kernvars = [c for c in ['enhanced_speed', 'heart_rate', 'cadence'] if c in df.columns]
    if kernvars:
        df = df.dropna(subset=kernvars)

    if df.empty:
        return df  # lege activiteit — wordt verderop overgeslagen

    if 'enhanced_speed' in df.columns:
        df = df[df['enhanced_speed'] > 0.5]
    if 'heart_rate' in df.columns:
        df = df[(df['heart_rate'] > 40) & (df['heart_rate'] < 220)]

    if df.empty:
        return df

    # Interpoleer alleen als index een DatetimeIndex is
    num_cols = df.select_dtypes(include=['number']).columns
    if isinstance(df.index, pd.DatetimeIndex):
        df[num_cols] = df[num_cols].interpolate(method='time', limit=5)
        df['elapsed_seconds'] = (df.index - df.index[0]).total_seconds().astype(int)
    else:
        df[num_cols] = df[num_cols].interpolate(limit=5)
        df['elapsed_seconds'] = range(len(df))

    df = df.reset_index()
    return df


# ──────────────────────────────────────────────
#  LAAG 2 CACHE: feature engineering
#  Sleutel: bestandshashes + rolling_window
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_feature_matrix(
    file_hashes: tuple[str, ...],
    rolling_window: int,
    _clean_dfs: list[pd.DataFrame],          # underscore = niet gehasht door Streamlit
) -> pd.DataFrame:
    """
    Gecached op (file_hashes, rolling_window).
    Zelfde bestanden + zelfde window = gratis.
    """
    parts = []
    for i, df in enumerate(_clean_dfs, start=1):
        df = df.copy()
        df['run_id'] = i
        df['duration'] = df['elapsed_seconds']
        df['progress']  = df['elapsed_seconds'] / df['elapsed_seconds'].max()
        parts.append(_engineer_features(df, window=rolling_window))
    return pd.concat(parts, ignore_index=True)


def _engineer_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df.copy()
    if 'distance' in df.columns:
        df['cumulative_distance'] = df['distance']
    elif 'enhanced_speed' in df.columns:
        df['cumulative_distance'] = df['enhanced_speed'].cumsum()
    if 'power' in df.columns:
        df['cumulative_power'] = df['power'].cumsum()

    for col in ['enhanced_speed', 'heart_rate', 'cadence', 'power']:
        if col in df.columns:
            df[f'{col}_rolling'] = df[col].rolling(window=window, min_periods=1).mean()

    if 'heart_rate' in df.columns and 'enhanced_speed' in df.columns:
        df['hr_speed_ratio'] = df['heart_rate'] / (df['enhanced_speed'] + 1e-5)
        df['hr_drift'] = df['hr_speed_ratio'].rolling(window=window, min_periods=1).mean()

    if 'cadence' in df.columns:
        df['cadence_trend'] = df['cadence'].rolling(window=window, min_periods=1).mean()

    df['progress'] = df['elapsed_seconds'] / df['elapsed_seconds'].max()
    return df


# ──────────────────────────────────────────────
#  LAAG 3 CACHE: CV + modellen
#  Sleutel: bestandshashes + rolling_window
#  (drempel_pct zit NIET in de sleutel: muur-detectie
#   is razendsnel en hoeft niet opnieuw te trainen)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_cv(
    file_hashes: tuple[str, ...],
    rolling_window: int,
    max_rijen: int,
    snelle_modus: bool,
    _df_features: pd.DataFrame,              # underscore = niet gehasht
) -> dict:
    """
    Leave-one-out CV + feature importance.
    Gecached op (file_hashes, rolling_window, max_rijen, snelle_modus).
    Drempel aanpassen hertraint het model NIET.
    """
    # Downsample per activiteit voor snelheid
    parts = []
    for rid, grp in _df_features.groupby('run_id'):
        if len(grp) > max_rijen:
            grp = grp.iloc[::len(grp)//max_rijen].head(max_rijen)
        parts.append(grp)
    _df_features = pd.concat(parts, ignore_index=True)

    if snelle_modus:
        modellen_def = {
            'Random Forest': lambda: RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1),
        }
    else:
        modellen_def = {
            'Lineaire Regressie': lambda: Pipeline([
                ('scaler', StandardScaler()), ('model', LinearRegression())]),
            'Random Forest':      lambda: RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting':  lambda: GradientBoostingRegressor(
                n_estimators=50, random_state=42),
        }

    feature_cols = [c for c in [
        'heart_rate', 'cadence', 'heart_rate_rolling', 'cadence_rolling',
        'hr_drift', 'cadence_trend', 'hr_speed_ratio',
        'cumulative_distance', 'cumulative_power',
        'elapsed_seconds', 'progress',
    ] if c in _df_features.columns]

    target_col   = 'enhanced_speed'
    alle_run_ids = sorted(_df_features['run_id'].unique())

    cv_scores      = {n: {'MAE': [], 'RMSE': [], 'R2': []} for n in modellen_def}
    fold_resultaten = {}

    for test_run_id in alle_run_ids:
        df_train = _df_features[_df_features['run_id'] != test_run_id].copy()
        df_test  = _df_features[_df_features['run_id'] == test_run_id].copy()

        # Sla fold over als train of test te klein is
        if len(df_train) < 10 or len(df_test) < 2:
            continue

        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col]
        X_test  = df_test[feature_cols].fillna(0)
        y_test  = df_test[target_col]

        # Sla over als er onvoldoende variatie is in de target
        if y_train.nunique() < 2:
            continue

        fold_scores = {}
        for mnaam, mfactory in modellen_def.items():
            try:
                m = mfactory()
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)
                cv_scores[mnaam]['MAE'].append(mae)
                cv_scores[mnaam]['RMSE'].append(rmse)
                cv_scores[mnaam]['R2'].append(r2)
                fold_scores[mnaam] = {'MAE': mae, 'y_pred': y_pred.tolist()}
            except Exception:
                continue

        if not fold_scores:
            continue

        beste = min(fold_scores, key=lambda m: fold_scores[m]['MAE'])
        fold_resultaten[test_run_id] = {
            'naam':       df_test['source_file'].iloc[0] if 'source_file' in df_test.columns else f'run_{test_run_id}',
            'df_test':    df_test,
            'y_pred':     fold_scores[beste]['y_pred'],
            'model_naam': beste,
        }

    # Filter modellen zonder scores (alle folds overgeslagen)
    cv_scores = {k: v for k, v in cv_scores.items() if v['MAE']}
    if not cv_scores:
        raise RuntimeError(
            "Geen enkele fold kon worden getraind. "
            "Controleer of de FIT-bestanden voldoende rijdata bevatten "
            "(enhanced_speed, heart_rate, cadence)."
        )
    if not fold_resultaten:
        raise RuntimeError(
            "Geen resultaten beschikbaar. Mogelijk hebben alle activiteiten "
            "te weinig bruikbare datapunten na opschonen."
        )

    beste_naam = min(cv_scores, key=lambda m: np.mean(cv_scores[m]['MAE']))

    # Finaal model voor feature importance
    finaal_model = modellen_def[beste_naam]()
    X_all = _df_features[feature_cols].fillna(0)
    y_all = _df_features[target_col]
    finaal_model.fit(X_all, y_all)

    inner = (finaal_model.named_steps['model']
             if hasattr(finaal_model, 'named_steps') else finaal_model)

    importances = None
    if hasattr(inner, 'feature_importances_'):
        importances = pd.Series(inner.feature_importances_, index=feature_cols).sort_values()
    elif hasattr(inner, 'coef_'):
        importances = pd.Series(np.abs(inner.coef_), index=feature_cols).sort_values()

    return {
        'cv_scores':       cv_scores,
        'beste_naam':      beste_naam,
        'fold_resultaten': fold_resultaten,
        'feature_cols':    feature_cols,
        'importances':     importances,
    }


# ──────────────────────────────────────────────
#  MUUR-DETECTIE  (geen cache nodig: microsnel)
# ──────────────────────────────────────────────

def detect_muur(fold_resultaten: dict, drempel_pct: int) -> dict:
    resultaten = {}
    for run_id, fold in fold_resultaten.items():
        df_tst = fold['df_test']
        y_pred = np.array(fold['y_pred'])

        df_res = df_tst[['elapsed_seconds', 'enhanced_speed',
                          'heart_rate' if 'heart_rate' in df_tst.columns else 'elapsed_seconds',
                          'cadence'    if 'cadence'    in df_tst.columns else 'elapsed_seconds',
                          ]].copy()
        # Haal alleen bestaande kolommen op
        cols = ['elapsed_seconds', 'enhanced_speed']
        for opt in ['heart_rate', 'cadence', 'source_file']:
            if opt in df_tst.columns:
                cols.append(opt)
        df_res = df_tst[cols].copy()

        df_res['speed_predicted'] = y_pred
        df_res['afwijking']       = df_res['enhanced_speed'] - df_res['speed_predicted']
        df_res['afwijking_pct']   = (df_res['afwijking'] / df_res['speed_predicted']) * 100

        kandidaten    = df_res[df_res['afwijking_pct'] < drempel_pct]
        muur_tijdstap = kandidaten.iloc[0]['elapsed_seconds'] if not kandidaten.empty else None

        resultaten[run_id] = {
            'naam':          fold['naam'],
            'df_result':     df_res,
            'muur_tijdstap': muur_tijdstap,
            'model_naam':    fold['model_naam'],
        }
    return resultaten


# ──────────────────────────────────────────────
#  REPO-DATA LADEN  (relatief pad, werkt op
#  Streamlit Cloud en Codespaces)
# ──────────────────────────────────────────────

# Bepaal repo-root: probeer meerdere bekende locaties
# Repo-root = map waar dit script staat
_BASE = os.path.dirname(os.path.abspath(__file__))

REPO_DIRS = [
    os.path.join(_BASE, "data", "StravaJan"),
    os.path.join(_BASE, "data", "StravaJan", "activities_dump_download"),
    os.path.join(_BASE, "data", "StravaPieter"),
    os.path.join(_BASE, "data", "StravaPieter", "Alle activiteiten"),
    os.path.join(_BASE, "data", "StravaRobin"),
    os.path.join(_BASE, "data", "StravaRobin", "Alle activiteiten"),
]
FIT_EXTENSIONS = ('.fit', '.fit.gz')


def scan_repo_files() -> list[dict]:
    """
    Doorzoek alle REPO_DIRS en geef een gesorteerde lijst van
    {'path': ..., 'name': ..., 'athlete': ...} terug.
    """
    seen = set()
    found = []
    for d in REPO_DIRS:
        if not os.path.isdir(d):
            continue
        parts = d.replace("\\", "/").split("/")
        strava_part = next((p for p in parts if p.startswith("Strava")), "Onbekend")
        athlete = strava_part.replace("Strava", "")
        for fname in sorted(os.listdir(d)):
            fname_lower = fname.lower()
            if fname_lower.endswith('.fit') or fname_lower.endswith('.fit.gz'):
                full = os.path.join(d, fname)   # geen abspath, gewoon join
                if full not in seen and os.path.isfile(full):
                    seen.add(full)
                    found.append({'path': full, 'name': fname, 'athlete': athlete})
    return found


# ──────────────────────────────────────────────
#  WRAPPER: RepoFile  –  gedraagt zich als
#  UploadedFile zodat de rest van de code
#  ongewijzigd blijft
# ──────────────────────────────────────────────

class RepoFile:
    """Lichtgewicht wrapper om een pad op disk als 'uploaded file' aan te bieden.
    Lazy: bytes worden pas ingelezen bij eerste aanroep van read() of getvalue()."""
    def __init__(self, path: str):
        self._path = path
        self.name  = os.path.basename(path)
        self._bytes: bytes | None = None

    def _load(self):
        if self._bytes is None:
            if not os.path.isfile(self._path):
                raise FileNotFoundError(
                    f"FIT-bestand niet gevonden: {self._path!r}\n"
                    f"_BASE={_BASE!r}"
                )
            with open(self._path, 'rb') as f:
                self._bytes = f.read()

    def read(self) -> bytes:
        self._load()
        return self._bytes

    def seek(self, _):
        pass

    def getvalue(self) -> bytes:
        self._load()
        return self._bytes


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
repo_files = scan_repo_files()
has_repo_data = len(repo_files) >= 2

with st.sidebar:
    st.markdown('<div class="section-label">Databron</div>', unsafe_allow_html=True)

    if has_repo_data:
        bron = st.radio(
            "Bron",
            ["Repo-data (automatisch)", "Eigen bestanden uploaden"],
            label_visibility="collapsed",
        )
    else:
        bron = "Eigen bestanden uploaden"
        st.caption("Geen repo-data gevonden in /data/Strava*")

    if bron == "Repo-data (automatisch)":
        # Selecteer welke atleten mee te nemen
        atleten = sorted({f['athlete'] for f in repo_files})
        gekozen_atleten = st.multiselect(
            "Atleten",
            atleten,
            default=atleten,
        )
        sportfilter = st.text_input(
            "Sportfilter (optioneel)",
            placeholder="bijv. running",
        )
        uploaded_files = [
            RepoFile(f['path'])
            for f in repo_files
            if f['athlete'] in gekozen_atleten
        ]
        st.caption(f"{len(uploaded_files)} FIT-bestanden geselecteerd uit repo")
    else:
        uploaded_files = st.file_uploader(
            "Upload .fit of .fit.gz bestanden",
            type=['fit', 'gz'],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        sportfilter = st.text_input(
            "Sportfilter (optioneel)",
            placeholder="bijv. running",
        )

    st.markdown('<div class="section-label">Model-instellingen</div>', unsafe_allow_html=True)
    drempel_pct = st.slider(
        "Muur-drempel (%)",
        min_value=-30, max_value=-1, value=-10, step=1,
        help="Hoe ver de werkelijke snelheid onder de voorspelling moet vallen.",
    )
    rolling_window = st.slider(
        "Rolling window (sec)",
        min_value=10, max_value=300, value=60, step=10,
    )
    max_activiteiten = st.slider(
        "Max. activiteiten",
        min_value=2, max_value=50, value=10, step=1,
        help="Beperk het aantal activiteiten voor snellere analyse. Bestanden worden willekeurig geselecteerd.",
    )
    max_rijen = st.slider(
        "Max. rijen per activiteit",
        min_value=100, max_value=3000, value=500, step=100,
        help="Downsample elke activiteit voor snellere berekening. 500 rijen is doorgaans voldoende.",
    )
    snelle_modus = st.checkbox(
        "Snelle modus (alleen Random Forest)",
        value=True,
        help="Traint alleen Random Forest. Schakel uit voor vergelijking van alle drie modellen.",
    )

    st.markdown("---")
    run_btn = st.button("▶  Analyse uitvoeren")

# ──────────────────────────────────────────────
#  UPLOAD-CHECK
# ──────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div class="upload-hint">
        <h3 style="color:#4a5568;font-family:'Barlow Condensed',sans-serif;margin:0 0 0.5rem;">
            Geen bestanden geladen
        </h3>
        <p style="margin:0;">Selecteer atleten uit de repo-data of upload eigen .fit bestanden via de zijbalk.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Selecteer minimaal **twee** FIT-bestanden voor leave-one-out validatie.")
    st.stop()

# ──────────────────────────────────────────────
#  BESTANDSOVERZICHT  +  hashes berekenen
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">Geladen bestanden</div>', unsafe_allow_html=True)

file_bytes_list = [uf.read() for uf in uploaded_files]
file_hashes     = tuple(
    hashlib.sha256(b).hexdigest()[:16] for b in file_bytes_list
)

# Reset file pointers (Streamlit hergebruikt het object)
for uf in uploaded_files:
    uf.seek(0)

# Controleer welke bestanden al in cache zitten
from streamlit import cache_data as _cd  # nodig voor cache-check via _cd.get_stats()

cols_files = st.columns(min(len(uploaded_files), 4))
for i, (uf, fhash) in enumerate(zip(uploaded_files, file_hashes)):
    with cols_files[i % len(cols_files)]:
        size_kb = len(file_bytes_list[i]) / 1024
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bestand {i+1}</div>
            <div style="font-size:0.85rem;color:#c8d0e0;word-break:break-all;">{uf.name}</div>
            <div style="margin-top:0.4rem;font-size:0.75rem;color:#6b7a99;">
                {size_kb:.0f} KB &nbsp;·&nbsp;
                <span style="font-family:monospace;color:#3a4a6a;">{fhash}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  ANALYSE  –  drie gecachede lagen
# ──────────────────────────────────────────────
if run_btn:
    import random as _random
    # Steekproef als er meer bestanden zijn dan het maximum
    if len(uploaded_files) > max_activiteiten:
        uploaded_files = _random.sample(uploaded_files, max_activiteiten)
    n_files  = len(uploaded_files)
    n_models = 1 if snelle_modus else 3
    # Stap-gewichten:  parse(n) + features(1) + cv_folds(n * n_models) + muur(1)
    total_stappen = n_files + 1 + (n_files * n_models) + 1
    stap_nu = [0]  # list zodat inner functie kan muteren zonder nonlocal

    pbar  = st.progress(0, text="Start…")
    plog  = st.empty()

    def tick(label: str):
        stap_nu[0] += 1
        pct = int(stap_nu[0] / total_stappen * 100)
        pbar.progress(min(pct, 99), text=label)
        plog.markdown(
            f'<span style="color:#6b7a99;font-size:0.8rem;">▸ {label}</span>',
            unsafe_allow_html=True,
        )

    try:
        # ── Laag 1: parse per bestand ──────────────────
        clean_dfs   = []
        sport_labels = []
        cached_count = 0

        for i, (uf, fhash, fbytes) in enumerate(
                zip(uploaded_files, file_hashes, file_bytes_list), start=1):

            # Detecteer cache-hit door de functie te roepen vóór én ná
            # (Streamlit cache geeft bij hit instant terug)
            import time as _time
            t0 = _time.perf_counter()
            df_clean, sport = parse_uploaded_file(uf.name, fhash, fbytes)
            dt = _time.perf_counter() - t0
            was_cached = dt < 0.15   # <150 ms → cache-hit

            if df_clean.empty:
                tick(f"Bestand {i}/{n_files}: {uf.name} – overgeslagen (geen bruikbare data)")
                continue
            df_clean['source_file'] = uf.name
            clean_dfs.append(df_clean)
            sport_labels.append(sport)
            cached_count += int(was_cached)

            label = (f"Bestand {i}/{n_files}: {uf.name}"
                     + (" (cache)" if was_cached else " – parsen..."))
            tick(label)

        detected_sport = sport_labels[0]

        # ── Laag 2: feature engineering ───────────────
        tick("Feature engineering…")
        df_features = build_feature_matrix(file_hashes, rolling_window, clean_dfs)

        # Voeg source_file toe als die er nog niet in zit
        if 'source_file' not in df_features.columns:
            src_map = {i+1: uf.name for i, uf in enumerate(uploaded_files)}
            df_features['source_file'] = df_features['run_id'].map(src_map)

        # ── Laag 3: CV (met eigen stap-teller) ────────
        # We kunnen de interne loop niet aftikken vanuit de cache,
        # dus toon een enkelvoudige "CV loopt…" boodschap
        for fold_i in range(n_files):
            for _ in range(n_models):
                tick(f"CV fold {fold_i+1}/{n_files} – modellen trainen…")

        cv_resultaat = run_cv(file_hashes, rolling_window, max_rijen, snelle_modus, df_features)

        # ── Muur-detectie (altijd vers, razendsnel) ───
        tick("Muur-detectie…")
        act_res = detect_muur(cv_resultaat['fold_resultaten'], drempel_pct)

        pbar.progress(100, text="Analyse klaar!")
        plog.empty()

        st.session_state['results'] = {
            **cv_resultaat,
            'activiteit_resultaten': act_res,
            'detected_sport':        detected_sport,
            'drempel_pct':           drempel_pct,
            'cached_count':          cached_count,
            'n_files':               n_files,
        }

    except Exception as e:
        pbar.empty()
        plog.empty()
        st.error(f"Fout tijdens analyse: {e}")
        st.exception(e)
        st.stop()

# Drempel veranderd zonder opnieuw te klikken → muur-detectie live bijwerken
elif 'results' in st.session_state:
    prev = st.session_state['results']
    if prev.get('drempel_pct') != drempel_pct:
        # CV-resultaten zijn gecached; alleen muur opnieuw
        act_res = detect_muur(prev['fold_resultaten'], drempel_pct)
        st.session_state['results'] = {**prev,
                                        'activiteit_resultaten': act_res,
                                        'drempel_pct': drempel_pct}

# ──────────────────────────────────────────────
#  RESULTATEN WEERGEVEN
# ──────────────────────────────────────────────
if 'results' not in st.session_state:
    st.info("Klik op **Analyse uitvoeren** in de zijbalk om te starten.")
    st.stop()

results    = st.session_state['results']
act_res    = results['activiteit_resultaten']
cv_scores  = results['cv_scores']
beste_naam = results['beste_naam']
sport      = results['detected_sport']
cached_cnt = results.get('cached_count', 0)
n_files    = results.get('n_files', len(act_res))

# Cache-info banner
if cached_cnt > 0:
    st.markdown(
        f'<span class="cache-hit">{cached_cnt}/{n_files} bestand(en) uit cache geladen – '
        f'geen herverwerking nodig</span>',
        unsafe_allow_html=True,
    )

# ── KPI-balk ──
st.markdown('<div class="section-label">Samenvatting</div>', unsafe_allow_html=True)
avg_mae = np.mean(cv_scores[beste_naam]['MAE'])
avg_r2  = np.mean(cv_scores[beste_naam]['R2'])
n_muur  = sum(1 for r in act_res.values() if r['muur_tijdstap'] is not None)

kpi_cols = st.columns(5)
kpi_data = [
    ("Sport",        sport.capitalize(), ""),
    ("Activiteiten", str(n_files),       ""),
    ("Beste model",  beste_naam.replace(" ", "\u00a0"), ""),
    ("Gem. MAE",     f"{avg_mae:.3f}",   "m/s"),
    ("Gem. R²",      f"{avg_r2:.3f}",    ""),
]
for col, (label, val, unit) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}<span class="metric-unit">{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ── CV-scoretabel ──
st.markdown('<div class="section-label">Cross-validatie scores</div>', unsafe_allow_html=True)
cv_rows = [{
    'Model':      mnaam,
    'MAE (m/s)':  f"{np.mean(s['MAE']):.4f}",
    'RMSE (m/s)': f"{np.mean(s['RMSE']):.4f}",
    'R²':         f"{np.mean(s['R2']):.4f}",
    'Beste':      'ja' if mnaam == beste_naam else '',
} for mnaam, s in cv_scores.items()]
st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

# ── Per activiteit: grafieken ──
KLEUREN = ['#00c8ff', '#ff6b35', '#b084ff', '#2ecc71']

st.markdown('<div class="section-label">Pacing grafieken per activiteit</div>', unsafe_allow_html=True)

for run_id, res in act_res.items():
    kleur  = KLEUREN[(run_id - 1) % len(KLEUREN)]
    naam   = res['naam']
    df_res = res['df_result']
    muur   = res['muur_tijdstap']

    with st.expander(f"{naam}", expanded=True):
        if muur:
            muur_min = int(muur // 60)
            st.markdown(
                f'<span class="muur-badge">Muur gedetecteerd – minuut {muur_min} ({int(muur)}s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="geen-muur-badge">Geen muur – consistent pacing</span>',
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2)

        # Snelheid
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['elapsed_seconds'] / 60, y=df_res['enhanced_speed'],
                name='Werkelijk', line=dict(color=kleur, width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=df_res['elapsed_seconds'] / 60, y=df_res['speed_predicted'],
                name='Voorspeld', line=dict(color='#ffffff', width=2, dash='dash'),
            ))
            if muur:
                fig.add_vline(x=muur / 60, line_color='#e74c3c',
                              annotation_text='Muur', annotation_font_color='#e74c3c')
            fig.update_layout(
                title=dict(text='Snelheid (m/s)', font=dict(size=13, color='#c8d0e0')),
                xaxis_title='Tijd (min)', yaxis_title='m/s',
                plot_bgcolor='#10131c', paper_bgcolor='#10131c',
                font=dict(color='#8899aa'),
                legend=dict(orientation='h', y=1.12, font=dict(size=10)),
                height=300, margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
                yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Afwijking
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_res['elapsed_seconds'] / 60, y=df_res['afwijking_pct'],
                marker_color=[
                    '#e74c3c' if v < drempel_pct else ('#2ecc71' if v >= 0 else '#f39c12')
                    for v in df_res['afwijking_pct']
                ],
            ))
            fig2.add_hline(y=drempel_pct, line_dash='dot', line_color='#e74c3c',
                           annotation_text=f'Drempel {drempel_pct}%',
                           annotation_font_color='#e74c3c')
            fig2.update_layout(
                title=dict(text='Afwijking t.o.v. voorspelling (%)', font=dict(size=13, color='#c8d0e0')),
                xaxis_title='Tijd (min)', yaxis_title='%',
                plot_bgcolor='#10131c', paper_bgcolor='#10131c',
                font=dict(color='#8899aa'), showlegend=False,
                height=300, margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
                yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Hartslag
        if 'heart_rate' in df_res.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_res['elapsed_seconds'] / 60, y=df_res['heart_rate'],
                line=dict(color='#e74c3c', width=1.2),
                fill='tozeroy', fillcolor='rgba(231,76,60,0.08)',
            ))
            if muur:
                fig3.add_vline(x=muur / 60, line_color='#e74c3c')
            fig3.update_layout(
                title=dict(text='Hartslag (bpm)', font=dict(size=13, color='#c8d0e0')),
                xaxis_title='Tijd (min)', yaxis_title='bpm',
                plot_bgcolor='#10131c', paper_bgcolor='#10131c',
                font=dict(color='#8899aa'), showlegend=False,
                height=220, margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
                yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
            )
            st.plotly_chart(fig3, use_container_width=True)

# ── Feature importance ──
if results.get('importances') is not None:
    st.markdown('<div class="section-label">Feature importance</div>', unsafe_allow_html=True)
    imp = results['importances']
    fig_imp = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h', marker_color='#00c8ff',
    ))
    fig_imp.update_layout(
        title=dict(text=f'{beste_naam} – getraind op alle data',
                   font=dict(size=13, color='#c8d0e0')),
        plot_bgcolor='#10131c', paper_bgcolor='#10131c',
        font=dict(color='#8899aa'),
        height=350, margin=dict(l=150, r=20, t=50, b=40),
        xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', title='Belang'),
        yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#2a3248;font-size:0.75rem;letter-spacing:0.08em;">'
    'STRAVA PACING MODEL &nbsp;·&nbsp; LEAVE-ONE-OUT CV &nbsp;·&nbsp; MUUR-DETECTIE'
    '</p>',
    unsafe_allow_html=True,
)