#!/usr/bin/env python3
"""Add missing imports for types moved out of com.veles.llm.jgpt."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TYPE_IMPORT = {
    "Tensor": "com.veles.llm.jgpt.core.Tensor",
    "Fp16Tensor": "com.veles.llm.jgpt.core.Fp16Tensor",
    "QuantizedTensor": "com.veles.llm.jgpt.core.QuantizedTensor",
    "TensorOps": "com.veles.llm.jgpt.ops.TensorOps",
    "TensorOpsBackward": "com.veles.llm.jgpt.ops.TensorOpsBackward",
    "Fp16Ops": "com.veles.llm.jgpt.ops.Fp16Ops",
    "TransformerBackward": "com.veles.llm.jgpt.ops.TransformerBackward",
    "TensorCudaLibrary": "com.veles.llm.jgpt.cuda.TensorCudaLibrary",
    "GpuPendingGradients": "com.veles.llm.jgpt.cuda.GpuPendingGradients",
    "GpuAttentionBackwardWorkspace": "com.veles.llm.jgpt.cuda.GpuAttentionBackwardWorkspace",
    "GpuForwardBlockWorkspace": "com.veles.llm.jgpt.cuda.GpuForwardBlockWorkspace",
    "GpuBlockWorkspace": "com.veles.llm.jgpt.cuda.GpuBlockWorkspace",
    "GPUTest": "com.veles.llm.jgpt.cuda.GPUTest",
    "GPTModel": "com.veles.llm.jgpt.model.GPTModel",
    "BlockActivationCache": "com.veles.llm.jgpt.model.BlockActivationCache",
    "KvCache": "com.veles.llm.jgpt.model.KvCache",
    "LLMTrainer": "com.veles.llm.jgpt.training.LLMTrainer",
    "TrainingConfig": "com.veles.llm.jgpt.training.TrainingConfig",
    "LLMConfig": "com.veles.llm.jgpt.training.LLMConfig",
    "AdamOptimizer": "com.veles.llm.jgpt.training.AdamOptimizer",
    "LearningRateScheduler": "com.veles.llm.jgpt.training.LearningRateScheduler",
    "LearningRateSchedule": "com.veles.llm.jgpt.training.LearningRateSchedule",
    "TrainingProfiler": "com.veles.llm.jgpt.training.TrainingProfiler",
    "TrainingTimings": "com.veles.llm.jgpt.training.TrainingTimings",
    "BookTrainingState": "com.veles.llm.jgpt.training.BookTrainingState",
    "BookCheckpointPaths": "com.veles.llm.jgpt.training.BookCheckpointPaths",
    "LrRangeFinder": "com.veles.llm.jgpt.training.LrRangeFinder",
    "SimpleTrainer": "com.veles.llm.jgpt.training.SimpleTrainer",
    "TextDataset": "com.veles.llm.jgpt.data.TextDataset",
    "BPETokenizer": "com.veles.llm.jgpt.data.BPETokenizer",
    "DataLoader": "com.veles.llm.jgpt.data.DataLoader",
    "TrainLLM": "com.veles.llm.jgpt.app.TrainLLM",
    "MultiBookTrain": "com.veles.llm.jgpt.app.MultiBookTrain",
    "ProfileQuickRun": "com.veles.llm.jgpt.app.ProfileQuickRun",
    "LlmTextGeneration": "com.veles.llm.jgpt.app.LlmTextGeneration",
    "TensorOpsGPU": "com.veles.llm.jgpt.TensorOpsGPU",
    "GpuFloatBuffer": "com.veles.llm.jgpt.GpuFloatBuffer",
}


def pkg_of_file(content: str) -> str | None:
    m = re.search(r"^package ([a-z0-9.]+);", content, re.MULTILINE)
    return m.group(1) if m else None


def needs_import(pkg: str, fqcn: str) -> bool:
    return fqcn.rsplit(".", 1)[0] != pkg


def add_imports(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pkg = pkg_of_file(content)
    if not pkg or not pkg.startswith("com.veles.llm.jgpt"):
        return False

    to_add = []
    for simple, fqcn in TYPE_IMPORT.items():
        if not needs_import(pkg, fqcn):
            continue
        if not re.search(r"\b" + re.escape(simple) + r"\b", content):
            continue
        imp = f"import {fqcn};"
        if imp in content:
            continue
        to_add.append(imp)

    if not to_add:
        return False

    to_add = sorted(set(to_add))
    m = re.search(r"^(package [^;]+;\s*\n)", content, re.MULTILINE)
    if not m:
        return False
    insert_at = m.end()
    block = "\n".join(to_add) + "\n\n"
    new_content = content[:insert_at] + block + content[insert_at:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def main():
    for base in [
        os.path.join(ROOT, "src/main/java"),
        os.path.join(ROOT, "src/test/java"),
    ]:
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".java"):
                    p = os.path.join(root, f)
                    if add_imports(p):
                        print("fixed", p)


if __name__ == "__main__":
    main()
