FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/home/user/app/.cache/models
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user data ./data
COPY --chown=user web ./web
COPY --chown=user results ./results

# Train both local models at build time (embeds ~4,100 messages once), so startup is instant.
RUN python -m spamrectifier.benchmark --build-only && python -c "import numpy as np; assert str(np.load('artifacts/embedding_model.npz', allow_pickle=True)['embedder'][0]) != 'hashing'"

EXPOSE 7860
CMD ["uvicorn", "spamrectifier.web:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
