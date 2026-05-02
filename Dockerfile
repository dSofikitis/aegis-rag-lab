FROM node:20-alpine AS ui-build

WORKDIR /ui

COPY ui/package.json /ui/package.json
COPY ui/tsconfig.json ui/tsconfig.node.json /ui/
COPY ui/vite.config.ts ui/index.html /ui/
COPY ui/src /ui/src

RUN npm install
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY --from=ui-build /ui/dist /app/ui/dist

# Install CPU-only torch first so the larger CUDA torch from PyPI is not pulled
# in by sentence-transformers. Saves ~1.5 GB on the image.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch && \
    pip install --no-cache-dir -e .

# Pre-download the cross-encoder so the first query doesn't pay for it.
RUN python -c "from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["uvicorn", "aegis_rag_lab.main:app", "--host", "0.0.0.0", "--port", "8000"]
