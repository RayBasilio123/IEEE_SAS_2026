FROM python:3.10-slim-bookworm AS model_fetcher
ARG EXPERIMENT_PATH

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install huggingface-hub[cli]==0.27.1

COPY $EXPERIMENT_PATH/setup.s[h] /setup.sh

RUN mkdir -p /models
RUN --mount=type=cache,target=/root/.cache/huggingface \
    if [ -f /setup.sh ]; then \
        chmod +x /setup.sh && \
        /setup.sh; \
    fi

FROM python:3.10-slim-bookworm AS builder
ARG EXPERIMENT_PATH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
    git
COPY $EXPERIMENT_PATH/requirements.txt /workspaces/$EXPERIMENT_PATH/

RUN --mount=type=cache,target=/root/.cache/pip \ 
    python3 -m pip install --upgrade pip==24.3.1 && \
    python3 -m pip install -r /workspaces/$EXPERIMENT_PATH/requirements.txt --target /packages

FROM python:3.10-slim-bookworm AS runner
WORKDIR /workspaces

ENV GIT_PYTHON_REFRESH=quiet
ARG EXPERIMENT_PATH
ENV EXPERIMENT_PATH_ENV=$EXPERIMENT_PATH

COPY --from=model_fetcher /models /models

COPY --from=builder /packages /usr/local/lib/python3.10/site-packages

ENV MLFLOW_TRACKING_URI=/artifacts/mlruns
COPY utils_exp/ /workspaces/utils_exp
COPY $EXPERIMENT_PATH /workspaces/$EXPERIMENT_PATH
COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["sh", "/entrypoint.sh"]