# CBB ML – Contenedor SageMaker (esqueleto)

Usa esta carpeta como base para construir una imagen de ECR que corre tus **Training/Processing Jobs** sin internet.

## Cómo usar
1) Sube esta carpeta a CodeCommit/GitHub.
2) Crea un proyecto **CodeBuild** con `buildspec.yml` → publica imagen a ECR.
3) Lanza **SageMaker Training Job** con la imagen (env `TASK=train`).
4) Lanza **SageMaker Processing Job** con la imagen (env `TASK=predict`).

Edita `config/runtime.yaml` con tu bucket/prefijos. Pega tu lógica real dentro de:
- `src/steps/train.py`  (pega tu train.py actual)
- `src/steps/predict.py` (pega tu predict_daily.py actual)

El util `src/utils/io_s3.py` lee **el último archivo** en `s3://<bucket>/<input_prefix>/`.
