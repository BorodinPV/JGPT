# JGPT Training Environment
# Build: docker build -t jgpt .
# Run: docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints jgpt

FROM nvidia/cuda:12.6-devel-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    openjdk-25-jdk \
    maven \
    cmake \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source (see .dockerignore — data/, checkpoints/, .env, .git are excluded)
COPY pom.xml ./
COPY src ./src
COPY scripts ./scripts
COPY env ./env

# Build CUDA libraries, Java, then drop privileges in the same layer
RUN cd src/main/cpp && \
    mkdir -p build && cd build && \
    cmake .. && \
    cmake --build . && \
    cd /app && \
    mvn compile -DskipTests && \
    useradd --create-home --uid 1000 jgpt && chown -R jgpt:jgpt /app
USER jgpt

# Set environment variables
ENV JGPT_DECODER_GPU_PIPELINE=1
ENV JGPT_FULL_GPU_TRAIN=1
ENV JGPT_FLASH_ATTENTION=1

# Default command
CMD ["./scripts/linux/jgpt-smart.sh"]
