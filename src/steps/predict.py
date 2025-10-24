# -*- coding: utf-8 -*-
"""
Predicción diaria CBB (SageMaker Processing):
- Descarga artefactos del último run desde S3 (outputs/latest/ o el run_id más reciente).
- Reconstruye features/rollings igual que en train.
- Predice con XGB y Linear, aplica recalibración (a,b) + offsets por centro.
- Exporta Excel con timestamp y lo sube a S3 (predictions/{run_id}/ y predictions/latest/).
- Lee SIEMPRE el último archivo en data/input/current/

Config por env:
  BUCKET (default: modelo-ml-cbb)
  AWS_REGION (default: us-east-1)
  INPUT_S3_PREFIX (default: data/input/current/)
"""

import os, io, re, json, unicodedata, warnings
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import boto3

warnings.filterwarnings("ignore", category=UserWarning)

# ====================== CONFIG ======================
BUCKET = os.environ.get("BUCKET", "modelo-ml-cbb")
REGION = os.environ.get("AWS_REGION", "us-east-1")
INPUT_S3_PREFIX = os.environ.get("INPUT_S3_PREFIX", "data/input/current/")

# ====================== UTIL FS/S3 ======================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def s3():
    return boto3.client("s3", region_name=REGION)

def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def s3_list_dirs(bucket: str, prefix: str):
    paginator = s3().get_paginator("list_objects_v2")
    result = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            result.append(cp["Prefix"])
    return result

def s3_resolve_latest_under_prefix(bucket: str, prefix: str) -> str:
    """Devuelve s3://bucket/key del objeto más reciente bajo prefix."""
    paginator = s3().get_paginator("list_objects_v2")
    latest = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/") or obj["Size"] == 0:
                continue
            if (latest is None) or (obj["LastModified"] > latest["LastModified"]):
                latest = obj
    if not latest:
        raise FileNotFoundError(f"No hay archivos bajo s3://{bucket}/{prefix}")
    return f"s3://{bucket}/{latest['Key']}"

def parse_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://")
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key

def s3_download_to_file(bucket: str, key: str, local_path: str):
    ensure_dir(os.path.dirname(local_path))
    s3().download_file(bucket, key, local_path)

def s3_upload_file(local_path: str, s3_uri: str):
    b, k = parse_s3_uri(s3_uri)
    s3().upload_file(local_path, b, k)

def read_table_from_s3(s3_uri: str) -> pd.DataFrame:
    b, k = parse_s3_uri(s3_uri)
    obj = s3().get_object(Bucket=b, Key=k)
    body = obj["Body"].read()
    ext = os.path.splitext(k)[1].lower()

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(body), sep=";", encoding="utf-8-sig", engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(body), sep=";", encoding="latin-1", engine="python")
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(io.BytesIO(body))
    else:
        raise ValueError(f"Extensión no soportada para predicción: {ext}")

# ================== NORMALIZACIÓN/FEATURES ==================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def sanitize_name(s: str) -> str:
    s = strip_accents(s).replace("°", "").replace(".", " ")
    return re.sub(r"\s+", " ", s).strip()

ALIASES = {
    "fecha":"fecha","laboratorio":"laboratorio","muestra":"muestra",
    "n guia":"n_guia","nro guia":"n_guia","nro_guia":"n_guia","n° guia":"n_guia",
    "centro":"centro","centro 1":"centro_1","centro2":"centro_1","centro_2":"centro_1",
    "obra":"obra","codigo hormigon":"codigo_hormigon","código hormigón":"codigo_hormigon",
    "hormigon":"hormigon","tipo":"tipo","fc cilindrico":"fc_cilindrico","nc":"nc","tm":"tm",
    "cono":"cono","edad":"edad","tipocolocacion":"tipo_colocacion","tipo colocacion":"tipo_colocacion",
    "tipo colocación":"tipo_colocacion","especiales":"especiales","cono real":"cono_real",
    "probeta":"probeta","cod cto":"cod_cto","tipo cto":"tipo_cto",
    "cto t":"cto_t","cto t incluyendo omisiones del analisis":"cto_t_con_omisiones",
    "cto t. incluyendo omisiones del analisis":"cto_t_con_omisiones",
    "cto t. incluyendo omisiones del análisis":"cto_t_con_omisiones",
    "r1":"r1","r3":"r3","r7":"r7","r28 1":"r28_1","r28 2":"r28_2",
    "fi cilindrica":"fi_cilindrica","fi cilíndrica":"fi_cilindrica",
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

def apply_center_offsets(df_te, yhat_te, offsets_by_center, offset_global):
    if "centro" not in df_te.columns:
        return yhat_te
    off_vec = df_te["centro"].map(lambda c: offsets_by_center.get(c, np.nan))
    off_vec = off_vec.fillna(offset_global).astype(float).values
    return yhat_te + off_vec

# ====================== LOCALIZAR ARTEFACTOS ======================
def resolve_latest_artifacts_s3(bucket: str):
    latest_prefix = "outputs/latest/"
    candidates = [
        ("xgb", ["model_xgb.joblib", "modelo_xgb.joblib"]),
        ("linr", ["model_linr.joblib", "modelo_linr.joblib"]),
    ]
    have_latest = False
    for _, names in candidates:
        for nm in names:
            if s3_exists(bucket, latest_prefix + nm):
                have_latest = True
                break
        if have_latest:
            break
    if have_latest:
        return latest_prefix, candidates

    # Si no hay latest, busca el último run_id
    paginator = s3().get_paginator("list_objects_v2")
    run_dirs = []
    for page in paginator.paginate(Bucket=bucket, Prefix="outputs/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            m = re.match(r"^outputs/(\d{8}_\d{6})/$", cp["Prefix"])
            if m: run_dirs.append(m.group(1))
    if not run_dirs:
        raise FileNotFoundError(f"No se encontraron runs en s3://{bucket}/outputs/")
    run_dirs.sort()
    last_run = run_dirs[-1]
    return f"outputs/{last_run}/", candidates

# ====================== MAIN ======================
def main():
    run_id_exec = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_local = ensure_dir(f"/home/sagemaker-user/cbb/pred_outputs/{run_id_exec}")
    latest_local = ensure_dir(f"/home/sagemaker-user/cbb/pred_outputs/latest")

    # Resolver artefactos
    base_prefix, name_candidates = resolve_latest_artifacts_s3(BUCKET)
    local_art_dir = ensure_dir("/opt/ml/processing/models")
    paths = {}

    for model_key, filenames in name_candidates:
        found = None
        for fn in filenames:
            key = base_prefix + fn
            if s3_exists(BUCKET, key):
                local_path = os.path.join(local_art_dir, fn)
                s3_download_to_file(BUCKET, key, local_path)
                found = local_path
                break
        if not found:
            raise FileNotFoundError(f"No se encontró ninguno de {filenames} en s3://{BUCKET}/{base_prefix}")
        paths[model_key] = found

    # Metadatos (roll windows)
    roll_cfg = None
    for aux in ["roll_windows.json", "rolls.json"]:
        if s3_exists(BUCKET, base_prefix + aux):
            tmp = os.path.join(local_art_dir, aux)
            s3_download_to_file(BUCKET, base_prefix + aux, tmp)
            with open(tmp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "ROLL_WINDOWS" in data and "COUNT_ROLL_N" in data:
                    roll_cfg = {"ROLL_WINDOWS": data["ROLL_WINDOWS"], "COUNT_ROLL_N": data["COUNT_ROLL_N"]}
            break

    import joblib
    arte = {name: joblib.load(fpath) for name, fpath in paths.items()}
    if roll_cfg is None:
        roll_cfg = arte["xgb"]["roll_cfg"]
    recal = {k: arte[k]["recal"] for k in arte}
    offsets = {k: arte[k]["offsets"] for k in arte}
    num_feats = arte["xgb"]["num_feats"]
    cat_feats = arte["xgb"]["cat_feats"]

    # Dataset (último archivo bajo el prefijo)
    input_uri = s3_resolve_latest_under_prefix(BUCKET, INPUT_S3_PREFIX)
    print("Leyendo dataset de predicción desde S3…", input_uri)
    df_all = read_table_from_s3(input_uri)
    df_all = canonicalize_columns(df_all)
    if "fecha" in df_all.columns:
        df_all["fecha"] = df_all["fecha"].apply(safe_to_datetime)

    for c in ["fc_cilindrico","nc","tm","cono","edad","cono_real","cto_t","r1","r3","r7",
              "r28_1","r28_2","t_hormigon","t_ambiente","dosis_agua","cantidad_aditivo"]:
        if c in df_all.columns:
            df_all[c] = to_float_series(df_all[c])

    for c in ["laboratorio","tipo","tipo_colocacion","especiales","probeta",
              "cod_cto","tipo_cto","hormigon","codigo_hormigon","centro","centro_1","muestra"]:
        if c in df_all.columns:
            df_all[c] = df_all[c].astype("string").str.strip()
            df_all.loc[df_all[c].isin([""," ",None]), c] = pd.NA

    df_all = ensure_centro_column(df_all)

    if "fi_cilindrica" in df_all.columns and df_all["fi_cilindrica"].notna().any():
        df_all["r28"] = to_float_series(df_all["fi_cilindrica"])
    else:
        cols_r28 = [c for c in ["r28_1","r28_2"] if c in df_all.columns]
        df_all["r28"] = to_float_series(df_all[cols_r28].mean(axis=1, skipna=True)) if cols_r28 else np.nan

    # Solo predecimos si r28 está ausente o <=0 y r7 es válido (>0)
    mask_pred = (df_all["r28"].isna() | (df_all["r28"] <= 0)) & (df_all["r7"].notna() & (df_all["r7"] > 0))
    df_pred = df_all[mask_pred].copy()
    if df_pred.empty:
        print("No hay filas para predecir (todas tienen R28 o r7 inválido). Fin.")
        return

    df_pred = df_pred.sort_values("fecha") if "fecha" in df_pred.columns else df_pred.sort_index()
    if "cto_t" in df_pred.columns:
        df_pred["log_cto"] = np.log(df_pred["cto_t"].replace(0,np.nan))
        df_pred["cto_t_squared"] = df_pred["cto_t"]**2
        df_pred["r7_divided_cto"] = df_pred["r7"]/df_pred["cto_t"]
    else:
        df_pred["log_cto"] = df_pred["cto_t_squared"] = df_pred["r7_divided_cto"] = np.nan
    if ("dosis_agua" in df_pred.columns) and ("cto_t" in df_pred.columns):
        valid_wc = df_pred["dosis_agua"].notna() & df_pred["cto_t"].notna() & (df_pred["cto_t"]>0)
        df_pred["w_c"] = np.where(valid_wc, df_pred["dosis_agua"]/df_pred["cto_t"], np.nan)
    else:
        df_pred["w_c"] = np.nan

    df_pred["delta"] = np.nan
    df_pred = compute_rollings(df_pred, roll_cfg["ROLL_WINDOWS"], roll_cfg["COUNT_ROLL_N"])

    def force_numpy_nan(df0: pd.DataFrame, num_cols, cat_cols):
        for c in num_cols:
            if c not in df0.columns: df0[c] = np.nan
            df0[c] = pd.to_numeric(df0[c], errors="coerce").astype("float64")
        for c in cat_cols:
            if c not in df0.columns: df0[c] = np.nan
            s = df0[c].astype("object"); s = s.where(~pd.isna(s), np.nan); df0[c] = s
        return df0

    df_pred = force_numpy_nan(ensure_centro_column(df_pred), num_feats, cat_feats)

    if "r7" not in df_pred.columns or df_pred["r7"].isna().all():
        raise RuntimeError("No se encontró 'r7' válido en el dataset de predicción.")

    X_pred = df_pred[num_feats + cat_feats].copy()
    r7_vec = df_pred["r7"].values

    outputs = {}
    for name in ["xgb", "linr"]:
        pipe = arte[name]["pipeline"]
        d_hat = pipe.predict(X_pred)
        base = r7_vec + d_hat
        a = float(recal[name]["a"]); b = float(recal[name]["b"])
        y_recal = a + b * base
        off_c = offsets[name]["offsets_by_center"]; off_g = float(offsets[name]["offset_global"])
        y_final = apply_center_offsets(df_pred, y_recal, off_c, off_g)
        outputs[name] = y_final

    # ----- Export Seguro -----
    base_cols = ["fecha","muestra","centro","cod_cto","r7"]
    res = df_pred.copy()
    for c in base_cols:
        if c not in res.columns:
            res[c] = np.nan
    res = res[base_cols].copy()
    for name, vec in outputs.items():
        res[f"r28_hat_{name}"] = vec

    out_filename = f"predicciones_diarias_{run_id_exec}.xlsx"
    out_path = os.path.join(out_local, out_filename)
    res.to_excel(out_path, index=False)

    base_s3 = f"s3://{BUCKET}/predictions/{run_id_exec}/{out_filename}"
    latest_s3 = f"s3://{BUCKET}/predictions/latest/predicciones_diarias.xlsx"
    s3_upload_file(out_path, base_s3)
    s3_upload_file(out_path, latest_s3)

    import shutil
    shutil.copy2(out_path, os.path.join(latest_local, "predicciones_diarias.xlsx"))

    print("Artefactos usados desde: s3://%s/%s" % (BUCKET, base_prefix))
    print("Predicciones subidas a:", base_s3, "y", latest_s3)
    print("run_id_exec:", run_id_exec)
    print("Filas predichas:", len(res))

if __name__ == "__main__":
    main()
