import os, io, boto3, pandas as pd
def s3_client(region=None):
    return boto3.client("s3", region_name=region or os.environ.get("AWS_REGION","us-east-1"))
def parse_s3_uri(s3_uri:str):
    assert s3_uri.startswith("s3://")
    _,_,rest = s3_uri.partition("s3://"); bucket,_,key = rest.partition("/"); return bucket,key
def list_objects(bucket:str, prefix:str):
    s3=s3_client(); pag=s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents",[]): yield obj
def latest_key(bucket:str, prefix:str):
    latest=None
    for obj in list_objects(bucket,prefix):
        k=obj["Key"]
        if k.endswith("/") or (not k.lower().endswith((".csv",".parquet",".pq"))): continue
        if (latest is None) or (obj["LastModified"]>latest["LastModified"]): latest=obj
    return latest["Key"] if latest else None
def read_last_table(s3_uri_prefix:str):
    b,p=parse_s3_uri(s3_uri_prefix); k=latest_key(b,p)
    if not k: raise FileNotFoundError(f"No se encontró archivo en s3://{b}/{p}")
    s3=s3_client(); body=s3.get_object(Bucket=b,Key=k)["Body"].read(); ext=os.path.splitext(k)[1].lower()
    if ext==".csv":
        try: df=pd.read_csv(io.BytesIO(body),sep=";",encoding="utf-8-sig",engine="python")
        except Exception: df=pd.read_csv(io.BytesIO(body),sep=";",encoding="latin-1",engine="python")
        df.columns=[str(c).replace("\ufeff","").strip() for c in df.columns]; return df,k
    elif ext in (".parquet",".pq"): import pandas as pd; return pd.read_parquet(io.BytesIO(body)),k
    else: raise ValueError(f"Extensión no soportada: {ext}")
