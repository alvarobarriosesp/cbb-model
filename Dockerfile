FROM public.ecr.aws/docker/library/python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/ml/code/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /opt/ml/code/requirements.txt
COPY src /opt/ml/code/src
COPY config /opt/ml/code/config
COPY scripts/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh
ENV PYTHONPATH=/opt/ml/code/src:$PYTHONPATH
WORKDIR /opt/ml/code
ENV TASK=train
CMD ["/opt/ml/code/entrypoint.sh"]
