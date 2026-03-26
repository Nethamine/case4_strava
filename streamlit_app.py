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
    margin-bottom: 0;
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

/* ── INTRO SECTION ── */
.intro-section {
    background: linear-gradient(180deg, #0d1a2e 0%, #0d0f14 100%);
    border-bottom: 1px solid #1e2535;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.intro-section::before {
    content: '';
    position: absolute;
    top: -60px; right: -80px;
    width: 340px; height: 340px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.intro-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.5rem;
    margin-top: 2rem;
}
.intro-story {
    grid-column: 1 / 3;
}
.intro-story-lead {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.55rem;
    color: #ffffff;
    line-height: 1.3;
    margin: 0 0 1rem 0;
}
.intro-story-lead em {
    color: #00c8ff;
    font-style: normal;
}
.intro-story-body {
    font-size: 0.92rem;
    color: #8899aa;
    line-height: 1.75;
    max-width: 580px;
}
.intro-story-body strong {
    color: #c8d8e8;
    font-weight: 500;
}
.intro-steps {
    grid-column: 3 / 4;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.7rem 0.9rem;
    background: rgba(0,200,255,0.04);
    border: 1px solid #1a2535;
    border-radius: 4px;
    transition: border-color 0.2s;
}
.step-item:hover {
    border-color: rgba(0,200,255,0.25);
}
.step-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    color: #00c8ff;
    line-height: 1;
    min-width: 24px;
    opacity: 0.7;
}
.step-text {
    font-size: 0.82rem;
    color: #8899aa;
    line-height: 1.5;
}
.step-text strong {
    display: block;
    color: #c8d8e8;
    font-size: 0.85rem;
    margin-bottom: 0.1rem;
}
.intro-divider {
    width: 40px;
    height: 2px;
    background: #00c8ff;
    margin: 1.2rem 0;
    opacity: 0.5;
}
.intro-tag {
    display: inline-block;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #00c8ff;
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 2px;
    padding: 0.2rem 0.55rem;
    margin-right: 0.4rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}
.intro-collapse-hint {
    font-size: 0.72rem;
    color: #2a3a4a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1.5rem;
    cursor: default;
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
.speed-box-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #f39c12;
    border-left: 3px solid #f39c12;
    padding-left: 0.75rem;
    margin: 1.2rem 0 0.6rem;
    display: block;
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
#  INTRO / VERHAAL SECTIE
# ──────────────────────────────────────────────
with st.expander("**Over dit dashboard**", expanded=True):
    col_verhaal, col_stappen = st.columns([2, 1], gap="large")

    with col_verhaal:
        st.markdown("**`MACHINE LEARNING` · `SPORTANALYSE` · `PACING` · `MUURDETECTIE`**")
        st.markdown("### Elke sporter kent het gevoel wel. Ben ik goed aan het sporten? Ben ik effectief bezig?")
        st.markdown("---")
        st.markdown(
            "Herken je deze vragen? Komt dit zelf ook bij jou op?"
            " Dan is dit dashboard gemaakt voor jou!"
            " In dit dashboard gaan we de bovenstaande vragen beantwoorden op een manier waar jij wat aan hebt."
            " Het enige wat je hoeft te doen is je eigen FIT-bestanden uploaden en de rest wordt voor je gedaan!"
        )


    with col_stappen:
        for num, titel, tekst in [
            ("1", "Upload FIT-bestanden",          "Je kan je eigen .fit of .fit.gz bestanden van dezelfde sport uploaden via de zijbalk!"),
            ("2", "Model traint automatisch",      "Het model ontdekt jouw sport-patroon op basis van hartslag, cadans en snelheid."),
            ("3", "Voorspelling vs werkelijkheid", "De grafiek toont waar jouw tempo afweek van de verwachting."),
            ("4", "Muurdetectie",                  "Zodra de afwijking de drempel overschrijdt, markeert het dashboard het exacte moment."),
        ]:
            st.markdown(f"**{num} · {titel}**")
            st.caption(tekst)
            st.markdown("")


# ──────────────────────────────────────────────
#  HULPFUNCTIES  (low-level, niet gecached)
# ──────────────────────────────────────────────

def _is_gzip(filepath: str) -> bool:
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except Exception:
        return False


def _open_fit(filepath: str):
    if _is_gzip(filepath):
        tmp = tempfile.NamedTemporaryFile(suffix='.fit', delete=False)
        with gzip.open(filepath, 'rb') as f_in:
            shutil.copyfileobj(f_in, tmp)
        tmp.close()
        return FitFile(tmp.name), tmp.name
    return FitFile(filepath), None


@st.cache_data(show_spinner=False)
def _detect_sport_from_path(filepath: str, mtime: float | None = None) -> str | None:
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


@st.cache_data(show_spinner=False)
def parse_uploaded_file(filename: str, file_hash: str, file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, 'wb') as f:
        f.write(file_bytes)

    sport = _detect_sport_from_path(tmp_path)
    df    = _load_records_from_path(tmp_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if df.empty:
        return df, (sport or 'onbekend')

    df = _clean_fit_data(df)

    if len(df) < 30:
        return pd.DataFrame(), (sport or 'onbekend')

    return df, (sport or 'onbekend')


def _clean_fit_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp').sort_index()

    if 'heart_rate' not in df.columns or df['heart_rate'].isna().all():
        return pd.DataFrame()

    kernvars = [c for c in ['enhanced_speed', 'heart_rate', 'cadence'] if c in df.columns]
    if kernvars:
        df = df.dropna(subset=kernvars)

    if df.empty:
        return df

    if 'enhanced_speed' in df.columns:
        df = df[df['enhanced_speed'] > 0.5]
    if 'heart_rate' in df.columns:
        df = df[(df['heart_rate'] > 40) & (df['heart_rate'] < 220)]

    if df.empty:
        return df

    num_cols = df.select_dtypes(include=['number']).columns
    if isinstance(df.index, pd.DatetimeIndex):
        df[num_cols] = df[num_cols].interpolate(method='time', limit=5)
        df['elapsed_seconds'] = (df.index - df.index[0]).total_seconds().astype(int)
    else:
        df[num_cols] = df[num_cols].interpolate(limit=5)
        df['elapsed_seconds'] = range(len(df))

    df = df.reset_index()
    return df


@st.cache_data(show_spinner=False)
def build_feature_matrix(
    file_hashes: tuple[str, ...],
    rolling_window: int,
    _clean_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def run_cv(
    file_hashes: tuple[str, ...],
    rolling_window: int,
    max_rijen: int,
    snelle_modus: bool,
    _df_features: pd.DataFrame,
) -> dict:
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

    target_col = 'enhanced_speed'

    feature_cols = [c for c in [
        'heart_rate', 'cadence', 'heart_rate_rolling', 'cadence_rolling',
        'hr_drift', 'cadence_trend', 'hr_speed_ratio',
        'cumulative_distance', 'cumulative_power',
        'elapsed_seconds', 'progress',
    ] if c in _df_features.columns]

    if target_col not in _df_features.columns:
        raise RuntimeError(
            f"Kolom '{target_col}' niet gevonden in de data. "
            "Controleer of de FIT-bestanden snelheidsdata bevatten."
        )
    if not feature_cols:
        raise RuntimeError(
            "Geen bruikbare feature-kolommen gevonden. "
            "Controleer of de FIT-bestanden hart- of cadansdata bevatten."
        )

    alle_run_ids = sorted(_df_features['run_id'].unique())
    cv_scores      = {n: {'MAE': [], 'RMSE': [], 'R2': []} for n in modellen_def}
    fold_resultaten = {}

    for test_run_id in alle_run_ids:
        df_test  = _df_features[_df_features['run_id'] == test_run_id].copy()
        df_train = _df_features[_df_features['run_id'] != test_run_id].copy()

        if len(df_train) < 10 or len(df_test) < 2:
            continue

        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col]
        X_test  = df_test[feature_cols].fillna(0)
        y_test  = df_test[target_col]

        mask = y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 5 or y_train.std() < 1e-6:
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

    cv_scores = {k: v for k, v in cv_scores.items() if v['MAE']}
    if not cv_scores or not fold_resultaten:
        cv_scores = {list(modellen_def.keys())[0]: {'MAE': [0], 'RMSE': [0], 'R2': [0]}}
        beste_fallback = list(modellen_def.keys())[0]
        m = modellen_def[beste_fallback]()
        X_all = _df_features[feature_cols].fillna(0)
        y_all = _df_features[target_col].fillna(0)
        m.fit(X_all, y_all)
        for rid, grp in _df_features.groupby('run_id'):
            X_t = grp[feature_cols].fillna(0)
            y_pred = m.predict(X_t).tolist()
            fold_resultaten[rid] = {
                'naam':       grp['source_file'].iloc[0] if 'source_file' in grp.columns else f'run_{rid}',
                'df_test':    grp,
                'y_pred':     y_pred,
                'model_naam': beste_fallback + ' (train=test)',
            }

    beste_naam = min(cv_scores, key=lambda m: np.mean(cv_scores[m]['MAE']))

    finaal_model = modellen_def[beste_naam]()
    _mask_all = _df_features[target_col].notna()
    X_all = _df_features.loc[_mask_all, feature_cols].fillna(0)
    y_all = _df_features.loc[_mask_all, target_col]
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


def detect_pacing_events(
    fold_resultaten: dict,
    drempel_pct: int,
    aanhoudend_sec: int = 30,
    start_pct: float = 0.15,
) -> dict:
    """
    Detecteert twee events per activiteit:
      - muur : werkelijkheid blijft aanhoudend ONDER de voorspelling
      - flow : werkelijkheid blijft aanhoudend BOVEN de voorspelling

    Parameters:
        drempel_pct    - minimale afwijking in % om te tellen
        aanhoudend_sec - aantal opeenvolgende seconden boven/onder drempel
        start_pct      - eerste X% van activiteit negeren (opstartgedrag)
    """
    resultaten = {}

    for run_id, fold in fold_resultaten.items():
        df_tst = fold['df_test']
        y_pred = np.array(fold['y_pred'])

        cols = ['elapsed_seconds', 'enhanced_speed']
        for opt in ['heart_rate', 'cadence', 'source_file']:
            if opt in df_tst.columns:
                cols.append(opt)
        df_res = df_tst[cols].copy().reset_index(drop=True)

        df_res['speed_predicted'] = y_pred
        df_res['afwijking']       = df_res['enhanced_speed'] - df_res['speed_predicted']
        df_res['afwijking_pct']   = (df_res['afwijking'] / df_res['speed_predicted']) * 100

        # Smooth de afwijking om ruis te dempen
        afw_smooth = df_res['afwijking_pct'].rolling(window=10, min_periods=1, center=True).mean()

        # Eerste start_pct% van activiteit negeren
        max_sec  = df_res['elapsed_seconds'].max()
        start_sec = max_sec * start_pct
        geldig    = df_res['elapsed_seconds'] >= start_sec

        def _eerste_aanhoudende(richting: str):
            if richting == 'onder':
                vlag = (afw_smooth < -drempel_pct).astype(int)
            else:
                vlag = (afw_smooth > drempel_pct).astype(int)

            aanhoudend = vlag.rolling(
                window=aanhoudend_sec, min_periods=aanhoudend_sec
            ).sum() == aanhoudend_sec

            kandidaten = df_res[aanhoudend & geldig]
            if kandidaten.empty:
                return None
            # Terug naar het begin van het aanhoudende blok
            eerste_idx = max(kandidaten.index[0] - aanhoudend_sec + 1, 0)
            return float(df_res.loc[eerste_idx, 'elapsed_seconds'])

        muur_tijdstap = _eerste_aanhoudende('onder')
        flow_tijdstap = _eerste_aanhoudende('boven')

        resultaten[run_id] = {
            'naam':          fold['naam'],
            'df_result':     df_res,
            'muur_tijdstap': muur_tijdstap,
            'flow_tijdstap': flow_tijdstap,
            'model_naam':    fold['model_naam'],
        }
    return resultaten


# ──────────────────────────────────────────────
#  REPO-DATA LADEN
# ──────────────────────────────────────────────

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


@st.cache_data(show_spinner=False)
def scan_repo_files() -> list[dict]:
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
                full = os.path.join(d, fname)
                if full not in seen and os.path.isfile(full):
                    seen.add(full)
                    # detect sport once here (include mtime so cache is invalidated if file changes)
                    try:
                        sport = _detect_sport_from_path(full, os.path.getmtime(full))
                    except Exception:
                        sport = None
                    found.append({
                        'path': full,
                        'name': fname,
                        'athlete': athlete,
                        'sport': (sport or 'onbekend'),
                    })
    return found


_SPORT_BLACKLIST = {'generic', 'all', 'e_sports'}

@st.cache_data(show_spinner=False)
def scan_sporten(file_paths: tuple[str, ...]) -> list[str]:
    sporten = set()
    for path in file_paths:
        try:
            sport = _detect_sport_from_path(path)
            if not sport:
                continue
            if sport.strip().lstrip('-').isdigit():
                continue
            if sport.lower() in _SPORT_BLACKLIST:
                continue
            sporten.add(sport)
        except Exception:
            continue
    return sorted(sporten)


class RepoFile:
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
        atleten = sorted({f['athlete'] for f in repo_files})
        gekozen_atleten = st.multiselect(
            "Atleten",
            atleten,
            default=atleten,
        )

        # Use the precomputed `sport` field from `repo_files` (fast)
        beschikbare_sporten = sorted({
            f.get('sport') for f in repo_files
            if f.get('sport') and f.get('sport').lower() not in _SPORT_BLACKLIST
        })

        if beschikbare_sporten:
            sport_opties = ["Alle sporten"] + beschikbare_sporten
            gekozen_sport = st.selectbox(
                "Sport",
                sport_opties,
                index=0,
            )
            sportfilter = None if gekozen_sport == "Alle sporten" else gekozen_sport
        else:
            sportfilter = None
            st.caption("Sporten worden gedetecteerd bij inladen.")

        uploaded_files = [
            RepoFile(f['path'])
            for f in repo_files
            if f['athlete'] in gekozen_atleten and (sportfilter is None or f.get('sport') == sportfilter)
        ]

        st.caption(f"{len(uploaded_files)} FIT-bestanden geselecteerd uit repo")
    else:
        uploaded_files = st.file_uploader(
            "Upload .fit of .fit.gz bestanden",
            type=['fit', 'gz'],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            _tmp_paden = []
            _tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                _tp = os.path.join(_tmp_dir, uf.name)
                with open(_tp, 'wb') as _f:
                    _f.write(uf.read())
                uf.seek(0)
                _tmp_paden.append(_tp)

            beschikbare_sporten = scan_sporten(tuple(_tmp_paden))
            shutil.rmtree(_tmp_dir, ignore_errors=True)

            if beschikbare_sporten:
                sport_opties = ["Alle sporten"] + beschikbare_sporten
                gekozen_sport = st.selectbox("Sport", sport_opties, index=0)
                sportfilter = None if gekozen_sport == "Alle sporten" else gekozen_sport
            else:
                sportfilter = None
        else:
            sportfilter = None

    st.markdown('<div class="section-label">Model-instellingen</div>', unsafe_allow_html=True)

    drempel_pct = st.slider(
        "Muur-drempel (%)",
        min_value=1, max_value=30, value=10, step=1,
        help="Hoeveel procent de werkelijke snelheid onder de voorspelling moet zakken om als muur te tellen.",
    )
    rolling_window = st.slider(
        "Rolling window (sec)",
        min_value=10, max_value=300, value=60, step=10,
        help="Groter venster = soepelere features, iets minder precies.",
    )

    st.markdown('<div class="speed-box-label">Snelheid vs nauwkeurigheid</div>', unsafe_allow_html=True)

    max_activiteiten = st.slider(
        "Max. activiteiten (model)",
        min_value=2, max_value=50, value=10, step=1,
        help="Minder activiteiten = sneller, maar minder traindata voor het model.",
    )
    max_rijen = st.slider(
        "Max. rijen per activiteit",
        min_value=100, max_value=3000, value=500, step=100,
        help="Lagere waarde = sneller, maar fijnere pacing-patronen gaan verloren.",
    )
    snelle_modus = st.checkbox(
        "Alleen Random Forest",
        value=True,
        help="Traint alleen Random Forest (snelst). Uitvinken vergelijkt alle drie modellen.",
    )

    st.markdown("---")
    run_btn = st.button("▶  Analyse uitvoeren")
    if st.button("Cache wissen"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

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

for uf in uploaded_files:
    uf.seek(0)

_MAX_TOON = 10
_zichtbaar = list(zip(uploaded_files, file_hashes, file_bytes_list))[:_MAX_TOON]
_verborgen  = list(zip(uploaded_files, file_hashes, file_bytes_list))[_MAX_TOON:]

cols_files = st.columns(min(len(_zichtbaar), 4))
for i, (uf, fhash, fbytes) in enumerate(_zichtbaar):
    with cols_files[i % len(cols_files)]:
        size_kb = len(fbytes) / 1024
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

if _verborgen:
    st.caption(f"+ {len(_verborgen)} bestand(en) niet weergegeven")

# ──────────────────────────────────────────────
#  ANALYSE
# ──────────────────────────────────────────────
if run_btn:
    import random as _random
    if len(uploaded_files) > max_activiteiten:
        uploaded_files = _random.sample(uploaded_files, max_activiteiten)
    n_files  = len(uploaded_files)
    n_models = 1 if snelle_modus else 3
    total_stappen = n_files + 1 + (n_files * n_models) + 1
    stap_nu = [0]

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
        clean_dfs    = []
        sport_labels = []
        atleet_labels = []
        cached_count = 0

        _naam_naar_atleet = {f['name']: f['athlete'] for f in repo_files}

        for i, (uf, fhash, fbytes) in enumerate(
                zip(uploaded_files, file_hashes, file_bytes_list), start=1):

            import time as _time
            t0 = _time.perf_counter()
            df_clean, sport = parse_uploaded_file(uf.name, fhash, fbytes)
            dt = _time.perf_counter() - t0
            was_cached = dt < 0.15

            if df_clean.empty:
                tick(f"Bestand {i}/{n_files}: {uf.name} – overgeslagen (geen bruikbare data)")
                continue
            df_clean['source_file'] = uf.name
            df_clean['sport'] = sport or 'onbekend'
            clean_dfs.append(df_clean)
            sport_labels.append(sport)
            atleet_labels.append(_naam_naar_atleet.get(uf.name, ''))
            cached_count += int(was_cached)

            label = (f"Bestand {i}/{n_files}: {uf.name}"
                     + (" (cache)" if was_cached else " – parsen..."))
            tick(label)

        detected_sport  = sport_labels[0] if sport_labels else 'onbekend'
        sport_per_run   = {i+1: s for i, s in enumerate(sport_labels)}
        atleet_per_run  = {i+1: a for i, a in enumerate(atleet_labels)}

        tick("Feature engineering…")
        df_features = build_feature_matrix(file_hashes, rolling_window, clean_dfs)

        if 'source_file' not in df_features.columns:
            src_map = {i+1: uf.name for i, uf in enumerate(uploaded_files)}
            df_features['source_file'] = df_features['run_id'].map(src_map)

        for fold_i in range(n_files):
            for _ in range(n_models):
                tick(f"CV fold {fold_i+1}/{n_files} – modellen trainen…")

        cv_resultaat = run_cv(file_hashes, rolling_window, max_rijen, snelle_modus, df_features)

        tick("Muur-detectie…")
        act_res = detect_pacing_events(cv_resultaat['fold_resultaten'], drempel_pct)

        pbar.progress(100, text="Analyse klaar!")
        plog.empty()

        st.session_state['results'] = {
            **cv_resultaat,
            'activiteit_resultaten': act_res,
            'detected_sport':        detected_sport,
            'sport_per_run':         sport_per_run,
            'atleet_per_run':        atleet_per_run,
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

elif 'results' in st.session_state:
    prev = st.session_state['results']
    if prev.get('drempel_pct') != drempel_pct:
        act_res = detect_pacing_events(prev['fold_resultaten'], drempel_pct)
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
act_res      = results['activiteit_resultaten']
cv_scores    = results['cv_scores']
beste_naam   = results['beste_naam']
sport        = results['detected_sport']
sport_per_run  = results.get('sport_per_run', {})
atleet_per_run = results.get('atleet_per_run', {})
cached_cnt = results.get('cached_count', 0)
n_files    = results.get('n_files', len(act_res))

if cached_cnt > 0:
    st.markdown(
        f'<span class="cache-hit">{cached_cnt}/{n_files} bestand(en) uit cache geladen – '
        f'geen herverwerking nodig</span>',
        unsafe_allow_html=True,
    )

avg_mae = np.mean(cv_scores[beste_naam]['MAE'])
avg_r2  = np.mean(cv_scores[beste_naam]['R2'])
n_muur  = sum(1 for r in act_res.values() if r['muur_tijdstap'] is not None)

st.markdown('<div class="section-label">Samenvatting</div>', unsafe_allow_html=True)
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

st.markdown("", unsafe_allow_html=True)

_alle_namen = {rid: res['naam'] for rid, res in act_res.items()}
_max_weer   = min(len(_alle_namen), 10)

def _activiteit_label(rid):
    naam   = _alle_namen[rid]
    sport  = sport_per_run.get(rid, '')
    atleet = atleet_per_run.get(rid, '')
    delen  = []
    if atleet:
        delen.append(atleet)
    if sport and sport != 'onbekend':
        delen.append(sport.upper())
    suffix = '  ·  ' + '  ·  '.join(delen) if delen else ''
    return f"{naam}{suffix}"

_label_naar_rid = {_activiteit_label(rid): rid for rid in _alle_namen}

gekozen_labels = st.selectbox(
    "Activiteiten weergeven",
    options=list(_label_naar_rid.keys()),
    index=0,
    help="Selecteer welke activiteit je in de grafieken wilt zien.",
)
gekozen_labels = [gekozen_labels] if gekozen_labels else []

_weer_ids    = [_label_naar_rid[l] for l in gekozen_labels]
act_res_weer = {rid: act_res[rid] for rid in _weer_ids}

if not act_res_weer:
    st.warning("Selecteer minimaal één activiteit om resultaten te zien.")
    st.stop()

tab_grafieken, tab_cv, tab_importance = st.tabs([
    "Pacing grafieken",
    "Cross-validatie scores",
    "Feature importance",
])

with tab_cv:
    st.markdown('<div class="section-label">Cross-validatie scores</div>', unsafe_allow_html=True)
    cv_rows = [{
        'Model':      mnaam,
        'MAE (m/s)':  f"{np.mean(s['MAE']):.4f}",
        'RMSE (m/s)': f"{np.mean(s['RMSE']):.4f}",
        'R²':         f"{np.mean(s['R2']):.4f}",
        'Beste':      'ja' if mnaam == beste_naam else '',
    } for mnaam, s in cv_scores.items()]
    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Leave-one-out overzicht</div>', unsafe_allow_html=True)
    st.caption(
        "Per fold wordt één activiteit uitgelaten als testset. "
        "Het model traint op alle overige activiteiten en voorspelt de uitgelaten activiteit."
    )

    fold_resultaten = results.get('fold_resultaten', {})
    alle_run_ids    = sorted(fold_resultaten.keys())
    toon_ids        = alle_run_ids[:10]
    alle_namen_cv   = {rid: fold_resultaten[rid]['naam'] for rid in alle_run_ids}

    for fold_nr, test_id in enumerate(toon_ids, start=1):
        test_naam  = alle_namen_cv[test_id]
        test_sport = sport_per_run.get(test_id, '')
        sport_tag  = f"  ·  {test_sport.upper()}" if test_sport and test_sport != 'onbekend' else ''
        train_namen = []
        for rid in alle_run_ids:
            if rid != test_id:
                naam = alle_namen_cv[rid]
                sport = sport_per_run.get(rid, '')
                sport_tag = f"  ·  {sport.upper()}" if sport and sport != 'onbekend' else ''
                train_namen.append(f"{naam}{sport_tag}")
        train_rijen = "".join(
            f'<tr><td style="padding:0.2rem 0.75rem;color:#6b7a99;font-size:0.78rem;">{n}</td></tr>'
            for n in train_namen
        )
        st.markdown(f"""
        <div style="background:#10131c;border:1px solid #1e2535;border-left:3px solid #00c8ff;
                    border-radius:4px;padding:0.75rem 1rem;margin-bottom:0.6rem;">
            <div style="display:flex;gap:1.5rem;align-items:flex-start;">
                <div style="min-width:160px;">
                    <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                                color:#6b7a99;margin-bottom:0.2rem;">Fold {fold_nr}</div>
                    <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                                color:#00c8ff;margin-bottom:0.4rem;">Testset</div>
                    <div style="font-size:0.82rem;color:#e8eaf0;word-break:break-all;">{test_naam}{sport_tag}</div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                                color:#4a5568;margin-bottom:0.4rem;">Trainset ({len(train_namen)} activiteiten)</div>
                    <table style="width:100%;border-collapse:collapse;">{train_rijen}</table>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(alle_run_ids) > 10:
        st.caption(f"+ {len(alle_run_ids) - 10} folds niet weergegeven")

KLEUREN = ['#00c8ff', '#ff6b35', '#b084ff', '#2ecc71']
SMOOTH_WINDOW = 20

def _smooth(s: pd.Series, w: int = SMOOTH_WINDOW) -> pd.Series:
    return s.rolling(window=w, min_periods=1, center=True).mean()

def _truncate(df: pd.DataFrame, pct: float = 0.97) -> pd.DataFrame:
    cut = int(len(df) * pct)
    return df.iloc[:cut]

_LAYOUT = dict(
    plot_bgcolor='#10131c', paper_bgcolor='#10131c',
    font=dict(color='#8899aa'),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', showgrid=True),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', showgrid=True),
)

with tab_grafieken:
    st.markdown('<div class="section-label">Pacing grafieken per activiteit</div>', unsafe_allow_html=True)
    for run_id, res in act_res_weer.items():
        kleur  = KLEUREN[(run_id - 1) % len(KLEUREN)]
        naam   = res['naam']
        df_res = _truncate(res['df_result'])
        muur   = res['muur_tijdstap']
        flow   = res['flow_tijdstap']
        t_min  = df_res['elapsed_seconds'] / 60

        act_sport = sport_per_run.get(run_id, '')
        sport_tag  = f"  ·  {act_sport.upper()}" if act_sport and act_sport not in ('onbekend',) else ''
        with st.expander(f"{naam}{sport_tag}", expanded=True):
            # ── Badges ──────────────────────────────────────────────────
            badge_cols = st.columns(2)
            with badge_cols[0]:
                if muur:
                    muur_min = int(muur // 60)
                    st.markdown(
                        f'<span class="muur-badge">🔴 Muur – minuut {muur_min} ({int(muur)}s)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="geen-muur-badge">✓ Geen muur</span>',
                        unsafe_allow_html=True,
                    )
            with badge_cols[1]:
                flow = res['flow_tijdstap']
                if flow:
                    flow_min = int(flow // 60)
                    st.markdown(
                        f'<span class="geen-muur-badge">🟢 Flow – minuut {flow_min} ({int(flow)}s)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="muur-badge">✗ Geen flow</span>',
                        unsafe_allow_html=True,
                    )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([t_min, t_min[::-1]]),
                y=pd.concat([
                    _smooth(df_res['speed_predicted']),
                    _smooth(df_res['enhanced_speed'])[::-1]
                ]),
                fill='toself', fillcolor='rgba(0,200,255,0.07)',
                line=dict(width=0), hoverinfo='skip',
                showlegend=False, name='verschil',
            ))
            fig.add_trace(go.Scatter(
                x=t_min, y=_smooth(df_res['speed_predicted']),
                name='Voorspeld',
                line=dict(color='#ffffff', width=2.5, dash='dot'),
            ))
            fig.add_trace(go.Scatter(
                x=t_min, y=_smooth(df_res['enhanced_speed']),
                name='Werkelijk',
                line=dict(color=kleur, width=2),
            ))
            if muur:
                fig.add_vline(
                    x=muur / 60, line_color='#e74c3c', line_width=2,
                    annotation_text='Muur', annotation_font_color='#e74c3c',
                    annotation_position='top right',
                )
            if flow:
                fig.add_vline(
                    x=flow / 60, line_color='#2ecc71', line_width=2,
                    annotation_text='Flow', annotation_font_color='#2ecc71',
                    annotation_position='top left',
                )
            fig.update_layout(
                **_LAYOUT,
                title=dict(text='Snelheid (m/s)', font=dict(size=13, color='#c8d0e0')),
                xaxis_title='Tijd (min)', yaxis_title='m/s',
                legend=dict(orientation='h', y=1.12, font=dict(size=11),
                            bgcolor='rgba(0,0,0,0)'),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"speed_{run_id}")

            c1, c2 = st.columns(2)
            with c1:
                afw = _smooth(df_res['afwijking_pct'], w=30)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=t_min, y=afw.clip(upper=0),
                    fill='tozeroy', fillcolor='rgba(231,76,60,0.15)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))
                fig2.add_trace(go.Scatter(
                    x=t_min, y=afw,
                    line=dict(color='#8899aa', width=1.5),
                    fill='tozeroy', fillcolor='rgba(136,153,170,0.08)',
                    showlegend=False,
                ))
                afw_max = float(afw.max())
                afw_min = float(afw.min())
                fig2.add_hline(y=0, line_color='#4a5568', line_width=1)
                fig2.add_hline(
                    y=afw_max, line_dash='dash', line_color='#2ecc71', line_width=1.5,
                    annotation_text=f'Max +{afw_max:.1f}%',
                    annotation_font_color='#2ecc71', annotation_position='top right',
                )
                fig2.add_hline(
                    y=afw_min, line_dash='dash', line_color='#e74c3c', line_width=1.5,
                    annotation_text=f'Min {afw_min:.1f}%',
                    annotation_font_color='#e74c3c', annotation_position='bottom right',
                )
                fig2.add_hline(
                    y=-drempel_pct, line_dash='dot', line_color='#e74c3c', line_width=1,
                    annotation_text=f'Drempel -{drempel_pct}%',
                    annotation_font_color='#e74c3c', annotation_position='bottom left',
                )
                fig2.add_hline(
                    y=drempel_pct, line_dash='dot', line_color='#2ecc71', line_width=1,
                    annotation_text=f'Drempel +{drempel_pct}%',
                    annotation_font_color='#2ecc71', annotation_position='top left',
                )
                if muur:
                    fig2.add_vline(x=muur / 60, line_color='#e74c3c', line_width=1.5)
                if flow:
                    fig2.add_vline(x=flow / 60, line_color='#2ecc71', line_width=1.5)
                fig2.update_layout(
                    **_LAYOUT,
                    title=dict(text='Afwijking (%)', font=dict(size=13, color='#c8d0e0')),
                    xaxis_title='Tijd (min)', yaxis_title='%',
                    height=260,
                )
                st.plotly_chart(fig2, use_container_width=True, key=f"afw_{run_id}")

            with c2:
                if 'heart_rate' in df_res.columns:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=t_min, y=_smooth(df_res['heart_rate'], w=15),
                        line=dict(color='#e74c3c', width=1.8),
                        fill='tozeroy', fillcolor='rgba(231,76,60,0.06)',
                        showlegend=False,
                    ))
                    if muur:
                        fig3.add_vline(x=muur / 60, line_color='#e74c3c', line_width=1.5)
                    if flow:
                        fig3.add_vline(x=flow / 60, line_color='#2ecc71', line_width=1.5)
                    fig3.update_layout(
                        **_LAYOUT,
                        title=dict(text='Hartslag (bpm)', font=dict(size=13, color='#c8d0e0')),
                        xaxis_title='Tijd (min)', yaxis_title='bpm',
                        height=260,
                    )
                    st.plotly_chart(fig3, use_container_width=True, key=f"hr_{run_id}")

with tab_importance:
    if results.get('importances') is not None:
        st.markdown('<div class="section-label">Feature importance</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#10131c;border:1px solid #1e2535;border-left:3px solid #00c8ff;
                    border-radius:4px;padding:0.75rem 1.2rem;margin-bottom:1.2rem;font-size:0.82rem;color:#8899aa;">
            <strong style="color:#c8d8e8;">Wat zie je hier?</strong> Het model kent aan elke variabele een gewicht toe.
            Hoe groter het percentage, hoe meer die variabele het voorspelde tempo bepaalt.
            <strong style="color:#00c8ff;">hr_speed_ratio</strong> (verhouding hartslag/snelheid) is veruit de sterkste voorspeller.
        </div>
        """, unsafe_allow_html=True)

        imp = results['importances']
        imp_pct = imp * 100  # ← omzetten naar percentages

        fig_imp = go.Figure(go.Bar(
            x=imp_pct.values,
            y=imp_pct.index,
            orientation='h',
            marker_color='#00c8ff',
            marker=dict(
                color='#00c8ff',
                line=dict(color='rgba(0,200,255,0.3)', width=1),
            ),
            text=[f"{v:.1f}%" for v in imp_pct.values],  # ← labels op de bars
            textposition='outside',
            textfont=dict(color='#8899aa', size=11),
        ))
        fig_imp.update_layout(
            title=dict(text=f'{beste_naam} – getraind op alle data',
                       font=dict(size=13, color='#c8d0e0')),
            plot_bgcolor='#10131c', paper_bgcolor='#10131c',
            font=dict(color='#8899aa'),
            height=420,
            margin=dict(l=160, r=80, t=50, b=40),  # r vergroot voor labels buiten bar
            xaxis=dict(
                gridcolor='#1e2535', linecolor='#1e2535',
                title='Belang (%)',
                tickformat='.0f',          # ← geen decimalen op as
                ticksuffix='%',            # ← %-teken achter tick-waarden
            ),
            yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Geen feature importance beschikbaar voor het gekozen model.")

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#2a3248;font-size:0.75rem;letter-spacing:0.08em;">'
    'STRAVA PACING MODEL &nbsp;·&nbsp; LEAVE-ONE-OUT CV &nbsp;·&nbsp; MUUR-DETECTIE'
    '</p>',
    unsafe_allow_html=True,
)
