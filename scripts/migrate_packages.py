#!/usr/bin/env python3
"""One-shot package migration for JGPT module. Run: python3 scripts/migrate_packages.py"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_MOVE = {
    "Tensor": "core",
    "Fp16Tensor": "core",
    "QuantizedTensor": "core",
    "TensorOps": "ops",
    "TensorOpsBackward": "ops",
    "Fp16Ops": "ops",
    "TransformerBackward": "ops",
    "TensorCudaLibrary": "cuda",
    "GpuPendingGradients": "cuda",
    "GpuAttentionBackwardWorkspace": "cuda",
    "GpuForwardBlockWorkspace": "cuda",
    "GpuBlockWorkspace": "cuda",
    "GPUTest": "cuda",
    "GPTModel": "model",
    "BlockActivationCache": "model",
    "KvCache": "model",
    "LLMTrainer": "training",
    "TrainingConfig": "training",
    "LLMConfig": "training",
    "AdamOptimizer": "training",
    "LearningRateScheduler": "training",
    "LearningRateSchedule": "training",
    "TrainingProfiler": "training",
    "TrainingTimings": "training",
    "BookTrainingState": "training",
    "BookCheckpointPaths": "training",
    "LrRangeFinder": "training",
    "SimpleTrainer": "training",
    "TextDataset": "data",
    "BPETokenizer": "data",
    "DataLoader": "data",
    "TrainLLM": "app",
    "MultiBookTrain": "app",
    "ProfileQuickRun": "app",
    "LlmTextGeneration": "app",
}

ROOT_CLASSES = frozenset({"TensorOpsGPU", "GpuFloatBuffer"})

TEST_MOVE = {
    "TransformerBackwardTest": "ops",
    "TensorOpsTest": "ops",
    "TensorTest": "core",
    "TensorOpsBackwardTest": "ops",
    "CausalMaskTest": "ops",
    "AttentionBlockBackwardGpuTest": "cuda",
    "FfnNormResidualGpuTest": "cuda",
    "TransformerBlockTest": "model",
    "AttentionTest": "model",
    "GPTModelTest": "model",
    "KvCacheTest": "model",
    "SwiGluForwardTest": "model",
    "FFNTest": "model",
    "MultiHeadAttentionTest": "model",
    "RoPEAttentionTest": "model",
    "LayerNormTest": "model",
    "AdamOptimizerTest": "training",
    "LearningRateSchedulerTest": "training",
    "FullTrainingTest": "training",
    "TrainingLoopTest": "training",
    "BPETokenizerTest": "data",
    "EmbeddingGpuTest": "cuda",
    "AttentionForwardGpuTest": "cuda",
    "AttentionBackwardGpuTest": "cuda",
    "CrossEntropyGpuTest": "cuda",
    "GpuFloatBufferTest": "cuda",
    "QuantizedTensorTest": "core",
    "Fp16TensorTest": "core",
    "AdamGpuParityTest": "cuda",
}


def pkg(sub):
    if sub is None:
        return "com.veles.llm.jgpt"
    return f"com.veles.llm.jgpt.{sub}"


def rewrite_imports(content: str) -> str:
    """Flatten import com.veles.llm.jgpt.Foo -> subpackage (Foo starts with uppercase)."""

    def repl(m):
        cls = m.group(1)
        if cls in ROOT_CLASSES:
            return f"import com.veles.llm.jgpt.{cls};"
        if cls in MAIN_MOVE:
            return f"import com.veles.llm.jgpt.{MAIN_MOVE[cls]}.{cls};"
        return m.group(0)

    return re.sub(
        r"^import com\.veles\.llm\.tensor\.([A-Z][a-zA-Z0-9_]*);",
        repl,
        content,
        flags=re.MULTILINE,
    )


def set_package(content: str, new_pkg: str) -> str:
    return re.sub(
        r"^package com\.veles\.llm\.tensor(\.[a-z.]+)*;",
        f"package {new_pkg};",
        content,
        count=1,
        flags=re.MULTILINE,
    )


def collect_java(base):
    out = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".java"):
                out.append(os.path.join(root, f))
    return out


def main():
    base_main = os.path.join(ROOT, "src/main/java/com/veles/llm/jgpt")
    base_test = os.path.join(ROOT, "src/test/java/com/veles/llm/jgpt")

    moves = []
    for cls, sub in MAIN_MOVE.items():
        old = os.path.join(base_main, cls + ".java")
        if not os.path.isfile(old):
            print("SKIP missing", old)
            continue
        new_dir = os.path.join(base_main, sub)
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, cls + ".java")
        moves.append((old, new_path, pkg(sub)))

    for cls, sub in TEST_MOVE.items():
        old = os.path.join(base_test, cls + ".java")
        if not os.path.isfile(old):
            continue
        new_dir = os.path.join(base_test, sub)
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, cls + ".java")
        moves.append((old, new_path, pkg(sub)))

    for old, new_path, new_pkg in moves:
        with open(old, "r", encoding="utf-8") as f:
            text = f.read()
        text = set_package(text, new_pkg)
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(text)
        if os.path.abspath(old) != os.path.abspath(new_path):
            os.remove(old)
        print("moved", os.path.basename(old), "->", new_path)

    # Rewrite imports in all sources (after moves)
    all_java = collect_java(os.path.join(ROOT, "src/main/java"))
    all_java += collect_java(os.path.join(ROOT, "src/test/java"))
    for path in all_java:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = rewrite_imports(content)
        if new_content != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("imports", path)

    print("Done.")


if __name__ == "__main__":
    main()
