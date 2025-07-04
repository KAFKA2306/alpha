FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    zsh \
    sudo \
    postgresql-client \
    redis-tools \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML/AI tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    mlflow \
    tensorboard \
    wandb

# Install Node.js and npm (for frontend development)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Change ownership of workspace to vscode user
RUN chown -R vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

EXPOSE 8000 8888 3000

CMD ["sleep", "infinity"]