FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=user:user app.py .

# Pre-download models during build (faster cold start)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('MBZUAI/LaMini-Flan-T5-248M'); \
    AutoModelForSeq2SeqLM.from_pretrained('MBZUAI/LaMini-Flan-T5-248M')"
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

# Create writable directories
RUN mkdir -p uploaded_pdfs chroma_db

EXPOSE 7860

CMD ["python", "app.py"]
