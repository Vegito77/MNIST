# Use a specific platform and Python slim image
FROM --platform=linux/amd64 python:3.11.5-slim-bookworm

# Set environment variables for Python, pip, Poetry, and AWS
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.8.3 \
    AWS_DEFAULT_REGION='eu-west-2'

# Install system dependencies and Poetry via pipx
RUN apt-get update && apt-get install -y curl unzip less \
    && python3 -m pip install --user pipx \
    && python3 -m pipx ensurepath \
    && pipx install poetry

# Add Poetry to PATH
ENV PATH=/root/.local/bin:$PATH

# Install AWS CLI v2 for AWS integration
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install

# Set working directory
WORKDIR /app

# Copy Poetry files first to leverage Docker layer caching
COPY poetry.lock pyproject.toml /app/

# Install Python dependencies with Poetry
RUN poetry install --no-interaction --no-ansi

# Copy rest of your project code
COPY . /app/

# Default command to run your MNIST loader script
CMD ["python", "MNIST_Loader.py"]
