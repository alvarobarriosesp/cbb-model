import os, yaml, boto3
from datetime import datetime, timezone
from utils.io_s3 import read_last_table
import pandas as pd
CFG_PATH=os.environ.get("CBB_CFG","/opt/ml/code/config/runtime.yaml")
cfg=yaml.safe_load(open(CFG_PATH))
BUCKET=cfg["bucket"]; INPUT_PREFIX=cfg["input_prefix"]; PRED_PREFIX=cfg["predictions_prefix"]
df,key = read_last_table(f"s3://{BUCKET}/{INPUT_PREFIX}")
print(f"[predict] dataset: s3://{BUCKET}/{key} shape={df.shape}")
# TODO: pega aquí tu lógica real de PREDICCIÓN (actual predict_daily.py) con routing por centro
pred = df.head(10).copy()
pred["r28_hat_xgb"]=25.0; pred["r28_hat_linr"]=24.0
run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
local=f"/opt/ml/processing/pred/{run_id}"; os.makedirs(local,exist_ok=True)
out=os.path.join(local,"predicciones_diarias.csv"); pred.to_csv(out,index=False)
s3=boto3.client("s3",region_name=cfg.get("aws_region","us-east-1"))
s3.upload_file(out,BUCKET,f"{PRED_PREFIX}{run_id}/predicciones_diarias.csv")
s3.upload_file(out,BUCKET,f"{PRED_PREFIX}latest/predicciones_diarias.csv")
print(f"[predict] predicciones en s3://{BUCKET}/{PRED_PREFIX}{run_id}/predicciones_diarias.csv")
