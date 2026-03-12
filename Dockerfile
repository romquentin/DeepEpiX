FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /DeepEpiX

# Copy requirements first for better layer caching
COPY requirements/ /DeepEpiX/requirements/

# Install system dependencies including GUI support
RUN apt-get update && apt-get install -y \
    python3-tk

# Create virtual environments within the project directory
RUN python3 -m venv /DeepEpiX/.dashenv
RUN python3 -m venv /DeepEpiX/.tfenv
RUN python3 -m venv /DeepEpiX/.torchenv

# Upgrade pip in both environments
RUN /DeepEpiX/.dashenv/bin/pip install --upgrade pip
RUN /DeepEpiX/.tfenv/bin/pip install --upgrade pip
RUN /DeepEpiX/.torchenv/bin/pip install --upgrade pip

# Install dash environment dependencies
RUN /DeepEpiX/.dashenv/bin/pip install -r requirements/requirements-dashenv.txt

# Install TensorFlow and Pytorch environment dependencies based on OS/architecture
RUN OS=$(uname) && ARCH=$(uname -m) && echo "OS: $OS, ARCH: $ARCH" && \
    if [ "$OS" = "Darwin" ]; then \
    echo "Detected macOS ($ARCH). Installing Metal-compatible TensorFlow..."; \
    /DeepEpiX/.tfenv/bin/pip install -r requirements/requirements-tfenv-macos.txt; \
    /DeepEpiX/.torchenv/bin/pip install -r requirements/requirements-torchenv-macos.txt; \
    elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then \
    echo "Detected Linux ($ARCH). Installing CUDA-compatible TensorFlow..."; \
    /DeepEpiX/.tfenv/bin/pip install -r requirements/requirements-tfenv-cuda.txt; \
    /DeepEpiX/.torchenv/bin/pip install -r requirements/requirements-torchenv-cuda.txt; \
    else \
    echo "Unknown architecture: $ARCH. Cannot install CPU-only TensorFlow..."; \
    fi

# Copy the rest of the application
COPY ./ /DeepEpiX/

# Set environment variables
ENV VIRTUAL_ENV=/DeepEpiX/.dashenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/DeepEpiX/src

# Set working directory to src
WORKDIR /DeepEpiX/src

EXPOSE 8050

# Use exec form for better signal handling
CMD exec /DeepEpiX/.dashenv/bin/gunicorn -w $(python3 -c "import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)") -b 0.0.0.0:8050 --timeout 600 run:server
