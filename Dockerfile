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

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY --from=ui-build /ui/dist /app/ui/dist

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "aegis_rag_lab.main:app", "--host", "0.0.0.0", "--port", "8000"]
