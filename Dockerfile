FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project code
COPY config.py /app/config.py
COPY train/train_local.py /app/train/train_local.py
COPY data/ /app/data/

WORKDIR /app

# Default entrypoint: run training
ENTRYPOINT ["python", "train/train_local.py"]
