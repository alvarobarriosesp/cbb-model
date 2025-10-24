# -*- coding: utf-8 -*-
"""
Entrenamiento CBB: XGBoost (principal) + Regresión Lineal (secundario)
- Δ = R28 - R7, R28_hat = R7 + Δ_hat
- Filtros/derivadas/rollings, split temporal, recalibración lineal (a,b)
- Offsets por centro (14d, shrink τ=60), métricas globales y por centro
- Reentrena con TODO el train para producción (full-fit)
- Exporta artefactos y métricas a S3 bajo run_id y 'latest'
- Lee SIEMPRE el último archivo en data/input/current/

Config por env:
  BUCKET (default: modelo-ml-cbb)
  AWS_REGION (default: us-east-1)
  INPUT_S3_PREFIX (default: data/input/current/)
"""

import os, io, re, json, unicodedata, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import boto3

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)

# ========================== CONFIG ==========================
BUCKET = os.environ.get("BUCKET", "modelo-ml-cbb")
REGION = os.environ.get("AWS_REGION", "us-east-1")
INPUT_S3_PREFIX = os.environ.get("INPUT_S3_PREFIX", "data/input/current/")

# Splits / ventanas
TEST_FRAC = 0.16
VAL_FRAC_IN_TRAIN = 0.10
RANDOM_STATE = 42
ROLL_WINDOWS = [3, 7, 14, 28]
COUNT_ROLL_N = 15

# Recalibración y offsets
ENABLE_LINEAR_RECALIBRATION = True
RECAL_WINDOW_DAYS = 60
ENABLE_CENTER_OFFSETS = True
OFFSET_WINDOW_DAYS = 14
OFFSET_SHRINK_TAU = 60
MIN_SAMPLES_OFFSET_UNIT = 15

# Modelos
XGB_CFG = dict(
    n_estimators=1574,
    learning_rate=0.013389961802473335,
    max_depth=5,
    min_child_weight=4.0,
    subsample=0.6858338691278116,
    colsample_bytree=0.7042643218623181,
    reg_lambda=4.26701536634083,
    reg_alpha=0.23393948586534075,
    gamma=0.17490822506719805,
    objective="reg:absoluteerror",
    random_state=RANDOM_STATE,
    n_jobs=0,
    tree_method="hist",
)
LINR_CFG = dict()

# ====================== UTILIDADES S3/IO =====================
def ensure_local_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def s3_client():
    return boto3.client("s3", region_name=REGION)

def s3_upload(local_path: str, s3_uri: str):
    assert s3_uri.startswith("s3://")
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    s3_client().upload_file(local_path, bucket, key)

def s3_upload_bytes(data: bytes, s3_uri: str):
    assert s3_uri.startswith("s3://")
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    s3_client().put_object(Bucket=bucket, Key=key, Body=data)

def s3_resolve_latest_under_prefix(bucket: str, prefix: str) -> str:
    """Devuelve s3://bucket/key del objeto más reciente bajo prefix."""
    cli = s3_client()
    paginator = cli.get_paginator("list_objects_v2")
    latest = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/") or obj["Size"] == 0:
                continue
            if (latest is None) or (obj["LastModified"] > latest["LastModified"]):
                latest = obj
    if not latest:
        raise FileNotFoundError(f"No se encontraron archivos bajo s3://{bucket}/{prefix}")
    return f"s3://{bucket}/{latest['Key']}"

def read_table_from_s3(s3_uri: str) -> pd.DataFrame:
    assert s3_uri.startswith("s3://")
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    ext = os.path.splitext(key)[1].lower()

    if ext == ".csv":
        df = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(body), sep=";", encoding=enc, engine="python")
                break
            except Exception:
                df = None
        if df is None:
            df = pd.read_csv(io.BytesIO(body), sep=";", engine="python")
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        return df
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(io.BytesIO(body))
    else:
        raise ValueError(f"Extensión no soportada: {ext}")

# ================== PREPROCESO/FEATURES ======================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def sanitize_name(s: str) -> str:
    s = strip_accents(s).replace("°", "").replace(".", " ")
    return re.sub(r"\s+", " ", s).strip()

ALIASES = {
    "fecha":"fecha","laboratorio":"laboratorio","muestra":"muestra",
    "n guia":"n_guia","nro guia":"n_guia","nro_guia":"n_guia",
    "centro":"centro","centro 1":"centro_1","centro2":"centro_1","centro_2":"centro_1",
    "obra":"obra","codigo hormigon":"codigo_hormigon","hormigon":"hormigon","tipo":"tipo",
    "fc cilindrico":"fc_cilindrico","nc":"nc","tm":"tm","cono":"cono","edad":"edad",
    "tipocolocacion":"tipo_colocacion","tipo colocacion":"tipo_colocacion",
    "especiales":"especiales","cono real":"cono_real",
    "probeta":"probeta","cod cto":"cod_cto","tipo cto":"tipo_cto",
    "cto t":"cto_t","cto t incluyendo omisiones del analisis":"cto_t_con_omisiones",
    "r1":"r1","r3":"r3","r7":"r7","r28 1":"r28_1","r28 2":"r28_2",
    "fi cilindrica":"fi_cilindrica",
    "t hormigon":"t_hormigon","t ambiente":"t_ambiente",
    "dosis de agua":"dosis_agua","cantidad aditivo":"cantidad_aditivo","cantidad_aditivo":"cantidad_aditivo"
}

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = sanitize_name(c).lower()
        canon = ALIASES.get(c2)
        if canon is None:
            c2c = c2.replace(".", " ").replace("_", " ")
            canon = "cto_t_con_omisiones" if "cto t incluyendo omisiones" in c2c else re.sub(r"\s+", "_", c2c)
        new_cols.append(canon)
    out = df.copy(); out.columns = new_cols; return out

def safe_to_datetime(x):
    if pd.isna(x): return pd.NaT
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try: return datetime.strptime(str(x), fmt)
        except: pass
    return pd.to_datetime(x, errors="coerce", dayfirst=True)

def to_float_series(ser: pd.Series) -> pd.Series:
    if ser.dtype == object:
        ser = ser.astype(str).str.replace(",", ".", regex=False)
        ser = ser.replace({"#NAME?": np.nan})
    try:
        out = pd.to_numeric(ser, errors="coerce")
    except Exception:
        out = pd.to_numeric(ser.astype("string"), errors="coerce")
    return out.astype("float64")

def ensure_centro_column(df: pd.DataFrame) -> pd.DataFrame:
    if "centro" not in df.columns and "centro_1" in df.columns:
        df["centro"] = df["centro_1"]
    if "centro" in df.columns:
        if df["centro"].isna().all() and "centro_1" in df.columns:
            df["centro"] = df["centro_1"]
    return df

def add_time_rolls(g: pd.DataFrame, windows, lastN) -> pd.DataFrame:
    g = g.sort_values("fecha").copy()
    g["r28_r7_ratio"] = g["r28"]/g["r7"]
    for w in windows:
        win = f"{w}d"; minp = max(3, int(w*0.5))
        g[f"roll_delta_mean_{win}"]  = g["delta"].rolling(w, min_periods=minp).mean().shift(1)
        g[f"roll_delta_std_{win}"]   = g["delta"].rolling(w, min_periods=minp).std().shift(1)
        g[f"roll_r28_r7_mean_{win}"] = g["r28_r7_ratio"].rolling(w, min_periods=minp).mean().shift(1)
        g[f"roll_r28_r7_std_{win}"]  = g["r28_r7_ratio"].rolling(w, min_periods=minp).std().shift(1)
    minpN = max(3, lastN//3)
    g["roll_delta_mean_lastN"] = g["delta"].rolling(window=lastN, min_periods=minpN).mean().shift(1)
    g["roll_r28_r7_mean_lastN"] = g["r28_r7_ratio"].rolling(window=lastN, min_periods=minpN).mean().shift(1)
    return g

def compute_rollings(df, windows, lastN):
    group_keys = [c for c in ["cod_cto","centro"] if c in df.columns]
    if group_keys:
        try:
            df2 = df.groupby(group_keys, group_keys=False).apply(
                lambda x: add_time_rolls(x, windows, lastN), include_groups=False)
        except TypeError:
            df2 = df.groupby(group_keys, group_keys=False).apply(
                lambda x: add_time_rolls(x, windows, lastN))
    else:
        df2 = add_time_rolls(df, windows, lastN)
    return df2

def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100.0)

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)); mask = denom != 0
    if mask.sum() == 0: return np.nan
    return float(200.0 * np.mean(np.abs(y_pred[mask]-y_true[mask]) / denom[mask]))

def mase(y_true, y_pred, y_naive) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float); y_naive = np.asarray(y_naive, dtype=float)
    denom = np.mean(np.abs(y_true - y_naive))
    if denom == 0: return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

def build_sample_weights(df_train: pd.DataFrame) -> np.ndarray:
    w_time = np.ones(len(df_train), dtype=float)
    if "fecha" in df_train.columns and df_train["fecha"].notna().any():
        tmin, tmax = df_train["fecha"].min(), df_train["fecha"].max()
        span = max((tmax - tmin).days, 1)
        age = (df_train["fecha"] - tmin).dt.days / span
        w_time = 0.7 + 0.6*age
    w_center = np.ones(len(df_train), dtype=float)
    if "centro" in df_train.columns:
        vc = df_train["centro"].value_counts(dropna=True)
        mean_n = vc.mean() if len(vc) else 1.0
        w_center = df_train["centro"].map(
            lambda c: min(1.6, max(0.7, np.sqrt(mean_n/max(vc.get(c,1),1))))
            if pd.notna(c) else 1.0
        ).values
    return w_time * w_center

def make_ohe(categories=None):
    try:
        if categories is not None:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True, categories=categories)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.01)
    except TypeError:
        if categories is not None:
            return OneHotEncoder(handle_unknown="ignore", sparse=True, categories=categories)
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def make_preproc(num_cols, cat_cols, scaler=False, categories=None):
    num_steps = [("imputer", SimpleImputer(strategy="median", missing_values=np.nan))]
    if scaler: num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
        ("ohe", make_ohe(categories))
    ])
    return ColumnTransformer(
        transformers=[("num", Pipeline(num_steps), num_cols),
                      ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

def recent_slice(df_src, window_days):
    if "fecha" in df_src.columns and df_src["fecha"].notna().any():
        max_d = df_src["fecha"].max(); min_d = max_d - timedelta(days=window_days)
        sl = df_src[df_src["fecha"] >= min_d].copy()
        return sl if not sl.empty else df_src.copy()
    return df_src.copy()

def linear_recalibration(y_true, y_pred, min_n=30):
    a_g, b_g = 0.0, 1.0
    if len(y_true) >= min_n:
        lr = LinearRegression()
        lr.fit(y_pred.reshape(-1,1), y_true.reshape(-1,1))
        a_g = float(lr.intercept_.ravel()[0]); b_g = float(lr.coef_.ravel()[0])
        if not np.isfinite(b_g) or (b_g <= 0.8) or (b_g >= 1.2):
            resid = y_true - y_pred
            a_g = float(np.nanmedian(resid)); b_g = 1.0
    return a_g, b_g

def apply_recal(yhat, a, b): return a + b*yhat

def compute_center_offsets(df_val_recent, yhat_val_recent,
                           shrink_tau=OFFSET_SHRINK_TAU,
                           min_samples_unit=MIN_SAMPLES_OFFSET_UNIT):
    resid = df_val_recent["r28"].values - yhat_val_recent
    dfv = df_val_recent.assign(residual=resid)
    offset_global = float(np.nanmedian(dfv["residual"].values)) if len(dfv)>0 else 0.0

    offsets_by_center = {}
    if "centro" in dfv.columns:
        for c, g in dfv.groupby("centro", dropna=True):
            if len(g) >= min_samples_unit:
                offsets_by_center[c] = float(np.nanmedian(g["residual"].values))

    if offsets_by_center:
        counts = dfv["centro"].value_counts()
        for c in list(offsets_by_center.keys()):
            n = int(counts.get(c, 0))
            w = n/(n+shrink_tau)
            offsets_by_center[c] = w*offsets_by_center[c] + (1-w)*offset_global

    return offsets_by_center, offset_global

def apply_center_offsets(df_te, yhat_te, offsets_by_center, offset_global):
    if "centro" not in df_te.columns:
        return yhat_te
    off_vec = df_te["centro"].map(lambda c: offsets_by_center.get(c, np.nan))
    off_vec = off_vec.fillna(offset_global).astype(float).values
    return yhat_te + off_vec

# ========================= MAIN =============================
def main():
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_local = ensure_local_dir(f"/home/sagemaker-user/cbb/outputs/{run_id}")
    latest_local = ensure_local_dir(f"/home/sagemaker-user/cbb/outputs/latest")

    # --- Resolver SIEMPRE el último input bajo el prefijo ---
    input_uri = s3_resolve_latest_under_prefix(BUCKET, INPUT_S3_PREFIX)
    print("Leyendo dataset desde S3…", input_uri)
    df = read_table_from_s3(input_uri)
    print("Shape original:", df.shape)

    # Canonicalización
    df = canonicalize_columns(df)
    print("Columnas canónicas:", list(df.columns))

    # Fechas
    if "fecha" in df.columns: df["fecha"] = df["fecha"].apply(safe_to_datetime)

    # Numéricos base
    numeric_candidates = [
        "fc_cilindrico","nc","tm","cono","edad","cono_real","cto_t","cto_t_con_omisiones",
        "r1","r3","r7","r28_1","r28_2","t_hormigon","t_ambiente","dosis_agua","cantidad_aditivo"
    ]
    for c in numeric_candidates:
        if c in df.columns: df[c] = to_float_series(df[c])

    # Categóricas
    for c in ["laboratorio","tipo","tipo_colocacion","especiales","probeta",
              "cod_cto","tipo_cto","hormigon","codigo_hormigon","centro","centro_1","muestra"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
            df.loc[df[c].isin([""," ",None]), c] = pd.NA

    # Filtros POV
    def apply_filter(df_in: pd.DataFrame, mask: pd.Series, label: str, logs: list):
        before = len(df_in); df_out = df_in[mask].copy(); after = len(df_out)
        logs.append((label, before, after, before-after)); return df_out

    logs = []
    if "especiales" in df.columns:
        esp = df["especiales"].astype("string")
        mask = esp.isna() | esp.str.fullmatch(r"(?i)HIDP")
        df = apply_filter(df, mask, "Especiales ∈ {vacío, HIDP}", logs)
    if "edad" in df.columns:
        df = apply_filter(df, df["edad"].isna() | (df["edad"]==28), "Edad == 28 (o NaN)", logs)
    if "cono" in df.columns:
        df = apply_filter(df, df["cono"].isna() | ((df["cono"]>=6)&(df["cono"]<=18)), "Cono ∈ [6,18] (o NaN)", logs)
    if "tm" in df.columns:
        df = apply_filter(df, df["tm"].isna() | (df["tm"]!=10), "TM != 10 (o NaN)", logs)
    if "nc" in df.columns:
        df = apply_filter(df, df["nc"].isna() | (df["nc"]!=0), "NC != 0 (o NaN)", logs)
    if "tipo" in df.columns:
        df = apply_filter(df, df["tipo"].astype("string").str.upper()=="G", "Tipo == 'G'", logs)
    if "r7" in df.columns:
        df.loc[df["r7"]==0, "r7"] = np.nan
        df = apply_filter(df, ~df["r7"].isna(), "R7 != 0 y no NaN", logs)
    before = len(df); df = df.drop_duplicates().copy()
    logs.append(("Drop duplicados exactos", before, len(df), before-len(df)))

    print("\nDiagnóstico de filtros (label | antes -> después | eliminadas):")
    for lbl,b,a,d in logs:
        print(f" - {lbl:28s}   | {b:6d} -> {a:6d} | -{d}")

    df = ensure_centro_column(df)

    # Objetivo R28
    if "fi_cilindrica" in df.columns and df["fi_cilindrica"].notna().any():
        df["r28"] = to_float_series(df["fi_cilindrica"])
    else:
        cols_r28 = [c for c in ["r28_1","r28_2"] if c in df.columns]
        if not cols_r28:
            raise ValueError("No hay r28_1/r28_2/fi_cilindrica para R28.")
        df["r28"] = to_float_series(df[cols_r28].mean(axis=1, skipna=True))

    # R28 <= 0
    n_before = len(df)
    df.loc[df["r28"] <= 0, "r28"] = np.nan
    df = df[~df["r28"].isna()].copy()
    print(f"[Filtro R28] Removidas {n_before - len(df)} filas con R28 <= 0.")

    df["delta"] = df["r28"] - df["r7"]

    # Derivadas
    if "cto_t" in df.columns:
        df["log_cto"] = np.log(df["cto_t"].replace(0,np.nan))
        df["cto_t_squared"] = df["cto_t"]**2
        df["r7_divided_cto"] = df["r7"]/df["cto_t"]
    else:
        df["log_cto"] = df["cto_t_squared"] = df["r7_divided_cto"] = np.nan
    if ("dosis_agua" in df.columns) and ("cto_t" in df.columns):
        valid_wc = df["dosis_agua"].notna() & df["cto_t"].notna() & (df["cto_t"]>0)
        df["w_c"] = np.where(valid_wc, df["dosis_agua"]/df["cto_t"], np.nan)
    else:
        df["w_c"] = np.nan

    # Split temporal
    df = df.sort_values("fecha") if "fecha" in df.columns else df.sort_index()
    n = len(df); test_n = max(1, int(n*TEST_FRAC))
    train = df.iloc[:n-test_n].copy()
    test  = df.iloc[n-test_n:].copy()
    print(f"Filas total: {n} | Train: {len(train)} | Test: {len(test)}")

    # Rollings
    train = compute_rollings(train, ROLL_WINDOWS, COUNT_ROLL_N)
    test  = compute_rollings(test,  ROLL_WINDOWS, COUNT_ROLL_N)

    # Features
    def basic_feature_lists(df_src):
        num_feats_base = [c for c in [
            "r7","cto_t","tm","cono","cono_real","t_hormigon","t_ambiente","dosis_agua","cantidad_aditivo",
            "w_c","r7_divided_cto","log_cto","cto_t_squared","fc_cilindrico"
        ] if c in df_src.columns]
        cat_feats = [c for c in ["laboratorio","tipo_colocacion","especiales","probeta",
                                 "cod_cto","tipo_cto","hormigon","centro"] if c in df_src.columns]
        return num_feats_base, cat_feats

    def build_final_feature_lists(df_src):
        num_base, cat = basic_feature_lists(df_src)
        num_roll = []
        for w in ROLL_WINDOWS:
            num_roll += [
                f"roll_delta_mean_{w}d", f"roll_delta_std_{w}d",
                f"roll_r28_r7_mean_{w}d", f"roll_r28_r7_std_{w}d"
            ]
        num_roll += ["roll_delta_mean_lastN","roll_r28_r7_mean_lastN"]
        num_feats = [c for c in num_base + num_roll if c in df_src.columns]
        return num_feats, cat

    num_feats, cat_feats = build_final_feature_lists(train)

    def force_numpy_nan(df0: pd.DataFrame, num_cols, cat_cols):
        for c in num_cols:
            if c not in df0.columns: df0[c] = np.nan
            df0[c] = pd.to_numeric(df0[c], errors="coerce").astype("float64")
        for c in cat_cols:
            if c not in df0.columns: df0[c] = np.nan
            s = df0[c].astype("object"); s = s.where(~pd.isna(s), np.nan); df0[c] = s
        return df0

    train = force_numpy_nan(ensure_centro_column(train), num_feats, cat_feats)
    test  = force_numpy_nan(ensure_centro_column(test),  num_feats, cat_feats)

    present_nums = [c for c in num_feats if c in train.columns]
    present_cats = [c for c in cat_feats if c in train.columns]

    # Preprocesadores
    prep_trees = make_preproc(present_nums, present_cats, scaler=False)
    prep_linear= make_preproc(present_nums, present_cats, scaler=True)

    # Partición interna
    val_n = max(1, int(len(train)*VAL_FRAC_IN_TRAIN))
    train_in = train.iloc[:len(train)-val_n].copy()
    val_in   = train.iloc[len(train)-val_n:].copy()

    Xtr_in = train_in[present_nums + present_cats].copy()
    ytr_in = train_in["delta"].copy()
    Xva_in = val_in[present_nums + present_cats].copy()
    r7_va  = val_in["r7"].values
    y28_va = val_in["r28"].values

    Xtr = train[present_nums + present_cats].copy()
    ytr = train["delta"].copy()
    Xte = test[present_nums + present_cats].copy()
    r7_te = test["r7"].values
    y28_te = test["r28"].values

    MODELS = {
        "xgb": Pipeline([("prep", prep_trees), ("reg", XGBRegressor(**XGB_CFG))]),
        "linr": Pipeline([("prep", prep_linear), ("reg", LinearRegression(**LINR_CFG))]),
    }
    STATUS = {k:"ok" for k in MODELS}

    def eval_model(name, pipe):
        print(f"\n=== Entrenando {name} ===")
        try:
            # ajuste interno para (a,b)
            sw_in = build_sample_weights(train_in)
            try:
                pipe.fit(Xtr_in, ytr_in, reg__sample_weight=sw_in)
            except Exception:
                pipe.fit(Xtr_in, ytr_in)

            d_val = pipe.predict(Xva_in)
            yhat_val = r7_va + d_val

            # recalibración reciente (a,b)
            if ENABLE_LINEAR_RECALIBRATION:
                val_recent = recent_slice(val_in, RECAL_WINDOW_DAYS)
                idx = val_recent.index
                yhat_val_series = pd.Series(yhat_val, index=val_in.index)
                a, b = linear_recalibration(val_recent["r28"].values,
                                            yhat_val_series.loc[idx].values)
            else:
                a, b = 0.0, 1.0

            # offsets por centro
            if ENABLE_CENTER_OFFSETS:
                val_off_recent = recent_slice(val_in, OFFSET_WINDOW_DAYS)
                idx2 = val_off_recent.index
                yhat_val_recal = apply_recal(yhat_val_series.loc[idx2].values, a, b)
                offsets_by_center, offset_global = compute_center_offsets(
                    val_off_recent, yhat_val_recal,
                    shrink_tau=OFFSET_SHRINK_TAU, min_samples_unit=MIN_SAMPLES_OFFSET_UNIT
                )
            else:
                offsets_by_center, offset_global = {}, 0.0

            # reentrenar con todo el train
            sw_full = build_sample_weights(train)
            try:
                pipe.fit(Xtr, ytr, reg__sample_weight=sw_full)
            except Exception:
                pipe.fit(Xtr, ytr)

            # predicciones base
            d_tr = pipe.predict(Xtr)
            d_te = pipe.predict(Xte)
            yhat_tr = apply_recal(train["r7"].values + d_tr, a, b)
            yhat_te = apply_recal(r7_te + d_te, a, b)

            # aplicar offsets por centro
            if ENABLE_CENTER_OFFSETS:
                yhat_te = apply_center_offsets(test, yhat_te, offsets_by_center, offset_global)

            # Métricas globales
            MAE  = mean_absolute_error(y28_te, yhat_te)
            RMSE = float(np.sqrt(mean_squared_error(y28_te, yhat_te)))
            R2   = r2_score(y28_te, yhat_te)
            MAPE = mape_safe(y28_te, yhat_te)
            sMAPE= smape(y28_te, yhat_te)
            MASE = mase(y28_te, yhat_te, y_naive=r7_te)

            metrics_g = dict(model=name, rows_train=int(len(train)), rows_test=int(len(test)),
                             MAE=float(MAE), RMSE=float(RMSE), R2=float(R2), MAPE=float(MAPE),
                             sMAPE=float(sMAPE), MASE=float(MASE),
                             recal_a=float(a), recal_b=float(b),
                             offset_global=float(offset_global), status="ok")

            # Métricas por centro
            by_center = []
            if "centro" in test.columns:
                tmp = pd.DataFrame({"centro": test["centro"].values,
                                    "r28": y28_te, "yhat": yhat_te})
                tmp = tmp[~tmp["centro"].isna()].copy()
                for c, g in tmp.groupby("centro", dropna=True):
                    if len(g)==0: continue
                    naive_local = np.full(len(g), np.nanmedian(g["r28"]))
                    by_center.append({
                        "model": name, "centro": c, "rows": int(len(g)),
                        "MAE": float(mean_absolute_error(g["r28"], g["yhat"])),
                        "RMSE": float(np.sqrt(mean_squared_error(g["r28"], g["yhat"]))),
                        "R2": float(r2_score(g["r28"], g["yhat"])) if len(g)>1 else np.nan,
                        "MAPE": float(mape_safe(g["r28"].values, g["yhat"].values)),
                        "sMAPE": float(smape(g["r28"].values, g["yhat"].values)),
                        "MASE": float(mase(g["r28"].values, g["yhat"].values, y_naive=naive_local))
                    })

            return metrics_g, by_center, yhat_te, (offsets_by_center, offset_global), (a,b), pipe

        except Exception as e:
            print(f"[ERROR] {name} falló: {e}")
            STATUS[name] = "failed"
            return dict(model=name, rows_train=int(len(train)), rows_test=int(len(test)),
                        MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE=np.nan, sMAPE=np.nan, MASE=np.nan,
                        recal_a=np.nan, recal_b=np.nan, offset_global=np.nan, status="failed"), [], None, ({},0.0), (0.0,1.0), None

    all_global, all_centers = [], []
    preds_dict, offsets_dict, recal_dict, pipes = {}, {}, {}, {}

    for name, pipe in MODELS.items():
        mg, bc, yhat_te, (off_c, off_g), (a,b), fitted = eval_model(name, pipe)
        all_global.append(mg)
        all_centers.extend(bc)
        if yhat_te is not None:
            preds_dict[name] = yhat_te
        offsets_dict[name] = {"offset_global": off_g, "offsets_by_center": off_c}
        recal_dict[name] = {"a": a, "b": b}
        pipes[name] = fitted

    # Importancias (XGB, sobre Δ)
    try:
        if "xgb" in pipes and "xgb" in preds_dict:
            xgb_pipe = pipes["xgb"]
            Xte_perm = test[present_nums + present_cats].copy()
            perm = permutation_importance(xgb_pipe, Xte_perm, test["delta"].values,
                                          n_repeats=5, random_state=RANDOM_STATE,
                                          scoring="neg_mean_absolute_error")
            prep = xgb_pipe.named_steps["prep"]
            num_names = [c for c in present_nums if c in test.columns]
            try:
                ohe_f = prep.named_transformers_["cat"].named_steps["ohe"]
                cat_names = list(ohe_f.get_feature_names_out(present_cats))
            except Exception:
                cat_names = [f"{c}__enc" for c in present_cats]
            feat_names = num_names + cat_names
            k = min(len(feat_names), perm.importances_mean.shape[0])
            imp_df = pd.DataFrame({
                "feature": feat_names[:k],
                "importance_abs_mae": np.abs(perm.importances_mean[:k])
            }).sort_values("importance_abs_mae", ascending=False).head(20)
        else:
            imp_df = pd.DataFrame()
    except Exception as e:
        print("[WARN] No fue posible calcular importancias por permutación:", e)
        imp_df = pd.DataFrame()

    # ======= EXPORTES LOCALES =======
    metrics_g = pd.DataFrame(all_global)
    metrics_g.sort_values(["status","MAE"], ascending=[True, True]).to_excel(os.path.join(out_local, "metrics_global.xlsx"), index=False)

    df_centros = pd.DataFrame(all_centers)
    if not df_centros.empty:
        df_centros = df_centros.sort_values(["centro","MAE"])
    df_centros.to_excel(os.path.join(out_local, "metrics_por_centro.xlsx"), index=False)

    if not df_centros.empty:
        df_centros_ord = df_centros.sort_values(["centro","MAE","sMAPE","RMSE"], ascending=[True, True, True, True])
        ganador = df_centros_ord.drop_duplicates(subset=["centro"], keep="first").reset_index(drop=True)
        ganador.to_excel(os.path.join(out_local, "ganador_por_centro.xlsx"), index=False)
        resumen_ganadores = ganador["model"].value_counts().rename_axis("model").reset_index(name="wins")
        resumen_ganadores.to_excel(os.path.join(out_local, "resumen_ganadores.xlsx"), index=False)
    else:
        ganador = pd.DataFrame(columns=["centro","model","MAE","sMAPE","RMSE"])
        resumen_ganadores = pd.DataFrame(columns=["model","wins"])

    # Predicciones test comparativa
    base_cols = ["fecha","muestra","centro","cod_cto","r7","r28"]
    cols_presentes = [c for c in base_cols if c in test.columns]
    pred_test = test[cols_presentes].copy()
    for c in base_cols:
        if c not in pred_test.columns: pred_test[c] = np.nan
    for name, yhat in preds_dict.items():
        pred_test[f"r28_hat_{name}"] = yhat
    orden = base_cols + [c for c in pred_test.columns if c not in base_cols]
    pred_test = pred_test[orden]
    pred_test.to_excel(os.path.join(out_local, "predicciones_test.xlsx"), index=False)

    # Predicciones test con todas las features
    feature_cols = list(dict.fromkeys(present_nums + present_cats))
    full_base = [c for c in base_cols if c in test.columns]
    pred_test_full = test[full_base].copy()
    for c in feature_cols:
        if c not in pred_test_full.columns and c in test.columns:
            pred_test_full[c] = test[c].values
    for name, yhat in preds_dict.items():
        pred_test_full[f"r28_hat_{name}"] = yhat
    orden_full = full_base + [c for c in feature_cols if c not in full_base] + \
                 [c for c in pred_test_full.columns if c.startswith("r28_hat_")]
    pred_test_full = pred_test_full[orden_full]
    pred_test_full.to_excel(os.path.join(out_local, "predicciones_test_full.xlsx"), index=False)

    # Offsets export
    with pd.ExcelWriter(os.path.join(out_local, "offsets_por_modelo.xlsx"), engine="xlsxwriter") as w:
        for name, od in offsets_dict.items():
            df_off_c = pd.DataFrame(sorted((od["offsets_by_center"] or {}).items()), columns=["centro","offset_median_val"])
            pd.DataFrame([{"offset_global": float(od["offset_global"]),
                           "ventana_dias": int(OFFSET_WINDOW_DAYS),
                           "tau_shrink": int(OFFSET_SHRINK_TAU),
                           "a": float(recal_dict[name]["a"]),
                           "b": float(recal_dict[name]["b"])
                           }]).to_excel(w, index=False, sheet_name=f"{name}_global")
            if not df_off_c.empty:
                df_off_c.to_excel(w, index=False, sheet_name=f"{name}_por_centro")

    if not imp_df.empty:
        imp_df.to_excel(os.path.join(out_local, "importancias_xgb_top20.xlsx"), index=False)

    # Artefactos: guardar modelos + metadata
    import joblib
    for name, pipe in pipes.items():
        path = os.path.join(out_local, f"model_{name}.joblib")
        joblib.dump({
            "pipeline": pipe,
            "num_feats": present_nums,
            "cat_feats": present_cats,
            "roll_cfg": {"ROLL_WINDOWS": ROLL_WINDOWS, "COUNT_ROLL_N": COUNT_ROLL_N},
            "recal": recal_dict[name],
            "offsets": offsets_dict[name],
            "config": {"TEST_FRAC": TEST_FRAC, "VAL_FRAC_IN_TRAIN": VAL_FRAC_IN_TRAIN}
        }, path)

    # Copia "latest"
    for fname in os.listdir(out_local):
        src = os.path.join(out_local, fname)
        dst = os.path.join(latest_local, fname)
        if os.path.isfile(src):
            import shutil; shutil.copy2(src, dst)

    # ======= SUBIDA A S3 =======
    base_s3 = f"s3://{BUCKET}/outputs/{run_id}/"
    latest_s3 = f"s3://{BUCKET}/outputs/latest/"
    for fname in os.listdir(out_local):
        s3_upload(os.path.join(out_local, fname), base_s3 + fname)
        s3_upload(os.path.join(out_local, fname), latest_s3 + fname)

    # Guardar best params XGB + roll windows
    s3_upload_bytes(json.dumps(XGB_CFG, indent=2).encode("utf-8"), base_s3 + "best_xgb_params.json")
    s3_upload_bytes(json.dumps({"ROLL_WINDOWS": ROLL_WINDOWS, "COUNT_ROLL_N": COUNT_ROLL_N,
                                "OFFSET_WINDOW_DAYS": OFFSET_WINDOW_DAYS, "OFFSET_SHRINK_TAU": OFFSET_SHRINK_TAU},
                               indent=2).encode("utf-8"), base_s3 + "roll_windows.json")
    s3_upload_bytes(json.dumps({"run_id": run_id}, indent=2).encode("utf-8"), base_s3 + "run.json")
    s3_upload_bytes(json.dumps({"run_id": run_id}, indent=2).encode("utf-8"), latest_s3 + "run.json")

    print("\n=== Estado de modelos ===")
    for k,v in {k:"ok" for k in MODELS}.items(): print(f"- {k}: {v}")
    print("\nArtefactos subidos a:", base_s3, "y", latest_s3)
    print("run_id:", run_id)

if __name__ == "__main__":
    main()
