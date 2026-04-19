# JGPT Makefile - Convenience wrapper for common tasks

.PHONY: all build build-cuda build-java test clean run smart

# Default target
all: build

# Build everything
build: build-cuda build-java

# Build CUDA libraries
build-cuda:
	@echo "Building CUDA libraries..."
	cd src/main/cpp && \
	mkdir -p build && cd build && \
	cmake .. && \
	cmake --build .

# Build with custom tile size (e.g., make build-cuda-tile TILE=128)
build-cuda-tile:
	@echo "Building CUDA libraries with tile size $(TILE)..."
	cd src/main/cpp && \
	mkdir -p build && cd build && \
	JGPT_FA_TILE_SIZE=$(TILE) cmake .. && \
	cmake --build .

# Build Java code
build-java:
	@echo "Building Java code..."
	mvn compile

# Run tests
test:
	mvn test -Djgpt.allow.no.gpu=true

# Clean build artifacts
clean:
	mvn clean
	rm -rf src/main/cpp/build
	rm -rf build

# Run training
run: build
	./scripts/jgpt-smart.sh

# Run with specific preset
smart:
	./scripts/jgpt-smart.sh

# Development mode (faster compilation, no optimizations)
dev:
	cd src/main/cpp && \
	mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	cmake --build .

# Release mode (optimized)
release:
	cd src/main/cpp && \
	mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release .. && \
	cmake --build .

# Help
help:
	@echo "JGPT Makefile targets:"
	@echo "  make build          - Build CUDA and Java code"
	@echo "  make build-cuda     - Build only CUDA libraries"
	@echo "  make build-cuda-tile TILE=128 - Build with custom tile size"
	@echo "  make build-java     - Build only Java code"
	@echo "  make test           - Run tests"
	@echo "  make clean          - Clean all build artifacts"
	@echo "  make run            - Build and run training"
	@echo "  make smart          - Run jgpt-smart.sh"
	@echo "  make dev            - Debug build"
	@echo "  make release        - Optimized release build"
