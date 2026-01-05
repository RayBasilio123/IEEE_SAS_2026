FROM python:3.10-slim-bookworm

ARG DEFAULT_USER=vscode

# Install dependencies required to install Poetry and clean up
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y git \
    python3-virtualenv \
    python3-pip \
    sudo

RUN useradd -m -s /bin/bash $DEFAULT_USER

RUN echo $DEFAULT_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DEFAULT_USER \
    && chmod 0440 /etc/sudoers.d/$DEFAULT_USER

RUN --mount=type=cache,target=/root/.cache/pip \ 
    python3 -m pip install --upgrade pip==24.3.1 && \
    python3 -m pip install ipykernel==6.29.5

WORKDIR /workspaces
