import os, json, yaml, boto3
from datetime import datetime, timezone
from utils.io_s3 import read_last_table
CFG_PATH=os.environ.get("CBB_CFG","/opt/ml/code/config/runtime.yaml")
cfg=yaml.safe_load(open(CFG_PATH))
BUCKET=cfg["bucket"]; INPUT_PREFIX=cfg["input_prefix"]; OUTPUTS_PREFIX=cfg["outputs_prefix"]
df,key = read_last_table(f"s3://{BUCKET}/{INPUT_PREFIX}")
print(f"[train] dataset: s3://{BUCKET}/{key} shape={df.shape}")
# TODO: pega aquí tu lógica real de ENTRENAMIENTO (actual train.py)
run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out_dir=f"/opt/ml/processing/output/{run_id}"; os.makedirs(out_dir,exist_ok=True)
open(os.path.join(out_dir,"model_xgb.joblib"),"wb").write(b"stub-xgb")
open(os.path.join(out_dir,"model_linr.joblib"),"wb").write(b"stub-linr")
with open(os.path.join(out_dir,"metrics_global.json"),"w") as f: json.dump({"MAE_xgb":9.9,"MAE_linr":10.5},f,indent=2)
s3=boto3.client("s3",region_name=cfg.get("aws_region","us-east-1"))
def up_folder(folder,bucket,prefix):
    for r,_,fs in os.walk(folder):
        for fn in fs:
            p=os.path.join(r,fn); key=f"{prefix}{os.path.relpath(p,folder)}"; s3.upload_file(p,bucket,key)
up_folder(out_dir,BUCKET,f"{OUTPUTS_PREFIX}{run_id}/"); up_folder(out_dir,BUCKET,f"{OUTPUTS_PREFIX}latest/")
print(f"[train] outputs en s3://{BUCKET}/{OUTPUTS_PREFIX}{run_id}/")
