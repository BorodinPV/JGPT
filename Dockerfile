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

# Copy source code
COPY . .

# Build CUDA libraries
RUN cd src/main/cpp && \
    mkdir -p build && cd build && \
    cmake .. && \
    cmake --build .

# Build Java code
RUN mvn compile -DskipTests

# Set environment variables
ENV JGPT_DECODER_GPU_PIPELINE=1
ENV JGPT_FULL_GPU_TRAIN=1
ENV JGPT_FLASH_ATTENTION=1

# Default command
CMD ["./scripts/jgpt-smart.sh"]
