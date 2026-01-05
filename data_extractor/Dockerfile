# Use the official Python slim image from the Docker Hub
FROM python:3.12-slim-bookworm AS builder

# Install any needed packages to build pysus
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y build-essential libffi-dev

COPY requirements.txt requirements.txt
# Install any needed packages specified in requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \ 
    python3 -m pip install --upgrade pip==24.3.1 && \
    pip install -r requirements.txt --target /packages

FROM python:3.12-slim-bookworm AS runner
# Set the working directory in the container
WORKDIR /app
COPY --from=builder /packages /usr/local/lib/python3.12/site-packages

# Copy script into the container
COPY __main__.py /app/data_extractor/__main__.py
COPY sinan_extractor.py /app/data_extractor/sinan_extractor.py


ENTRYPOINT ["python3", "-m", "data_extractor"]