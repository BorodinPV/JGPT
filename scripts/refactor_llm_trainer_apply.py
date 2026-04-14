#!/usr/bin/env python3
"""One-shot refactor: wire LLMTrainer to training/* helper classes."""
from __future__ import annotations

from pathlib import Path


def parse_method_end(lines: list[str], i: int) -> int:
    b = 0
    started = False
    while i < len(lines):
        for c in lines[i]:
            if c == "{":
                b += 1
                started = True
            elif c == "}":
                b -= 1
        i += 1
        if started and b == 0:
            return i
    raise RuntimeError("unclosed method")


def find_line(lines: list[str], pred) -> int:
    for i, ln in enumerate(lines):
        if pred(ln):
            return i
    return -1


def javadoc_start_above(lines: list[str], method_line: int) -> int:
    """If method_line is preceded by a Javadoc block, return index of /** line; else method_line."""
    t = method_line - 1
    while t >= 0 and lines[t].strip() == "":
        t -= 1
    if t < 0:
        return method_line
    st = lines[t].strip()
    if not (st.startswith("/**") or st.startswith("*")):
        return method_line
    while t > 0 and not lines[t].strip().startswith("/**"):
        t -= 1
    if t >= 0 and lines[t].strip().startswith("/**"):
        return t
    return method_line


def main() -> None:
    p = Path(__file__).resolve().parents[1] / "src/main/java/com/veles/llm/jgpt/training/LLMTrainer.java"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

    # --- Ranges on original indices (delete high indices first) ---
    ranges: list[tuple[int, int]] = []

    i_clip = find_line(
        lines,
        lambda l: l.startswith("    private boolean clipAndOptimizerStep(Tensor logits, float avgMicroLoss) {"),
    )
    if i_clip < 0:
        raise SystemExit("clipAndOptimizerStep not found")
    start_block1 = javadoc_start_above(lines, i_clip)
    i_ckpt_v4 = find_line(
        lines,
        lambda l: "CHECKPOINT_FORMAT_V4" in l and "private static final String" in l,
    )
    if i_ckpt_v4 < 0:
        raise SystemExit("CHECKPOINT_FORMAT_V4 not found")
    ranges.append((start_block1, i_ckpt_v4 + 1))

    i_full_star = find_line(
        lines,
        lambda l: "Full-GPU clip/optimizer step" in l and l.strip().startswith("*"),
    )
    if i_full_star < 0:
        raise SystemExit("Full-GPU javadoc line not found")
    s_full = i_full_star
    while s_full > 0:
        prev = lines[s_full - 1].strip()
        if prev.startswith("/**"):
            s_full -= 1
            break
        if prev.startswith("*") or prev == "":
            s_full -= 1
            continue
        break
    i_zero = find_line(
        lines,
        lambda l: l.startswith("    private static void zeroGpuGrads(Map<Tensor, GpuTensor> paramMap) {"),
    )
    if i_zero < 0:
        raise SystemExit("zeroGpuGrads not found")
    e_full = parse_method_end(lines, i_zero)
    ranges.append((s_full, e_full))

    i_fmt = find_line(lines, lambda l: "private static String formatEvalBestLossForLog(float v)" in l)
    if i_fmt < 0:
        raise SystemExit("formatEvalBestLossForLog not found")
    ranges.append((i_fmt, parse_method_end(lines, i_fmt)))

    i_fp16 = find_line(lines, lambda l: "private static boolean fp16AuxSoftenScaleAfterInfer()" in l)
    if i_fp16 < 0:
        raise SystemExit("fp16AuxSoftenScaleAfterInfer not found")
    ranges.append((javadoc_start_above(lines, i_fp16), parse_method_end(lines, i_fp16)))

    i_sync = find_line(lines, lambda l: "private static void synchronizeGpuAfterOverflowSkip()" in l)
    if i_sync < 0:
        raise SystemExit("synchronizeGpuAfterOverflowSkip not found")
    ranges.append((javadoc_start_above(lines, i_sync), parse_method_end(lines, i_sync)))

    i_json = find_line(lines, lambda l: "private static String jsonEsc(String s)" in l)
    if i_json < 0:
        raise SystemExit("jsonEsc not found")
    ranges.append((i_json, parse_method_end(lines, i_json)))

    i_agent = find_line(lines, lambda l: "private static void agentLogB39372" in l)
    if i_agent < 0:
        raise SystemExit("agentLogB39372 not found")
    ranges.append((i_agent, parse_method_end(lines, i_agent)))

    for a, b in sorted(ranges, key=lambda t: t[0], reverse=True):
        del lines[a:b]

    # --- Checkpoint cluster -> delegations ---
    i_save = find_line(lines, lambda l: "public void saveCheckpoint(String name)" in l)
    if i_save < 0:
        raise SystemExit("saveCheckpoint not found after deletes")

    # loadCheckpoint / saveCheckpoint bodies contain `{`/`}` inside format strings — do not use brace parsing here.
    i_load_model = find_line(
        lines,
        lambda l: l.startswith("    public void loadModelWeights(String name)")
        and "ClassNotFoundException" in l,
    )
    if i_load_model < 0:
        raise SystemExit("loadModelWeights (unique end marker) not found")
    if i_load_model <= i_save:
        raise SystemExit("checkpoint block end before start")

    idx = i_load_model

    replacement = """    public void saveCheckpoint(String name) throws IOException {
        LlmTrainerCheckpointIo.saveCheckpoint(this, name);
    }

    public void saveModelWeights(String name) throws IOException {
        LlmTrainerCheckpointIo.saveModelWeights(this, name);
    }

    public void awaitPendingCheckpointWrites() {
        LlmTrainerCheckpointIo.awaitPendingCheckpointWrites(this);
    }

    public void loadCheckpoint(String path) throws IOException, ClassNotFoundException {
        LlmTrainerCheckpointIo.loadCheckpoint(this, path);
    }

"""
    lines[i_save:idx] = [replacement]

    text = "".join(lines)

    repls = [
        ("canDeviceSampledTrainForward()", "LlmTrainerCrossEntropy.canDeviceSampledTrainForward(this)"),
        (
            "effectiveSampledCandidateCount(vocabSize)",
            "LlmTrainerCrossEntropy.effectiveSampledCandidateCount(this, vocabSize)",
        ),
        (
            "prepareSampledCandidateIds(batch.target, rows, vocabSize, candCount)",
            "LlmTrainerCrossEntropy.prepareSampledCandidateIds(this, batch.target, rows, vocabSize, candCount)",
        ),
        (
            "applyCrossEntropyLossAndGradDeviceAsync(logits, batch.target, ceScale)",
            "LlmTrainerCrossEntropy.applyCrossEntropyLossAndGradDeviceAsync(this, logits, batch.target, ceScale)",
        ),
        (
            "applyTrainLossAndGrad(logits, batch.target, ceScale)",
            "LlmTrainerCrossEntropy.applyTrainLossAndGrad(this, logits, batch.target, ceScale)",
        ),
        ("scaleGradients(parameters, partialScale)", "LlmTrainerOptimizerStep.scaleGradients(parameters, partialScale)"),
        (
            "scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale)",
            "LlmTrainerOptimizerStep.scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale)",
        ),
        (
            "clipAndOptimizerStepFullGpu(logits, avgMicroLoss)",
            "LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu(this, logits, avgMicroLoss)",
        ),
        (
            "clipAndOptimizerStep(logits, avgMicroLoss)",
            "LlmTrainerOptimizerStep.clipAndOptimizerStep(this, logits, avgMicroLoss)",
        ),
        (
            "clipAndOptimizerStepFullGpu(logits, loss)",
            "LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu(this, logits, loss)",
        ),
        (
            "clipAndOptimizerStep(logits, loss)",
            "LlmTrainerOptimizerStep.clipAndOptimizerStep(this, logits, loss)",
        ),
        ("zeroGradients(logits)", "LlmTrainerOptimizerStep.zeroGradients(this, logits)"),
        (
            "clearGpuParamGradsAfterOverflowSkip()",
            "LlmTrainerOptimizerStep.clearGpuParamGradsAfterOverflowSkip(this)",
        ),
        (
            "zeroGpuGradsMarkingParamGradsClean(model.gpuTensorByTrainableParameter())",
            "LlmTrainerOptimizerStep.zeroGpuGradsMarkingParamGradsClean(this, model.gpuTensorByTrainableParameter())",
        ),
        ("synchronizeGpuAfterOverflowSkip()", "LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip()"),
        ("float evalLoss = evaluate();", "float evalLoss = LlmTrainerEvalAndSample.evaluate(this);"),
        ("maybeAutoSample(epoch + 1)", "LlmTrainerEvalAndSample.maybeAutoSample(this, epoch + 1)"),
        ("formatEvalBestLossForLog(bestLoss)", "LlmTrainerTrainingFormat.formatEvalBestLossForLog(bestLoss)"),
    ]
    for a, b in repls:
        if a not in text:
            raise SystemExit(f"pattern not found for replace: {a!r}")
        text = text.replace(a, b)

    extra = [
        (
            "this.exitAfterOptimizerSteps = readPositiveEnvInt(",
            "this.exitAfterOptimizerSteps = LlmTrainerEnvUtils.readPositiveEnvInt(",
        ),
        (
            "this.cudaTrimEveryOptimizerSteps = readCudaTrimEveryOptimizerStepsFromEnv();",
            "this.cudaTrimEveryOptimizerSteps = LlmTrainerEnvUtils.readCudaTrimEveryOptimizerStepsFromEnv();",
        ),
        (
            "fp16AuxSoftenScaleAfterInfer() ? \"1\" : \"0\"",
            "LlmTrainerGpuUtils.fp16AuxSoftenScaleAfterInfer() ? \"1\" : \"0\"",
        ),
        (
            "config.batchSize * config.maxSeqLen * effectiveSampledCandidateCount(config.vocabSize)",
            "config.batchSize * config.maxSeqLen * LlmTrainerCrossEntropy.effectiveSampledCandidateCount(this, config.vocabSize)",
        ),
    ]
    for a, b in extra:
        if a not in text:
            raise SystemExit(f"extra pattern not found: {a!r}")
        text = text.replace(a, b)

    text = text.replace("agentLogB39372(", "LlmTrainerDebugLog.b39372(")
    text = text.replace("jsonEsc(", "LlmTrainerDebugLog.jsonEsc(")
    text = text.replace(
        "if (fp16AuxSoftenScaleAfterInfer()) {",
        "if (LlmTrainerGpuUtils.fp16AuxSoftenScaleAfterInfer()) {",
    )

    field_decls = [
        "private final GPTModel model;",
        "private final TrainingConfig config;",
        "private final DataLoader dataLoader;",
        "private final DataLoader evalDataLoader;",
        "private final AdamOptimizer optimizer;",
        "private int globalStep;",
        "private float bestLoss;",
        "private int shutdownProgressBaselineStep = 0;",
        "private int loadedResumeEpochIndex = 0;",
        "private int pendingCheckpointEpochIndex = 0;",
        "private int pendingCheckpointDataLoaderIndex = 0;",
        "private int loadedResumeDataLoaderIndex = 0;",
        "private boolean resumeReplayCheckpointShuffles = false;",
        "private final List<Tensor> parameters;",
        "private final int totalTrainingSteps;",
        "private final int warmupSteps;",
        "private final DynamicLossScaler dynamicLossScaler;",
        "private final int exitAfterOptimizerSteps;",
        "private final int cudaTrimEveryOptimizerSteps;",
        "private final boolean fp16Matmul;",
        "private final boolean ceAsyncDevice;",
        "private final boolean fp16DynamicResetEachEpoch;",
        "private int overflowLogPlannedStepKey = -1;",
        "private int overflowSkipRepeatCount;",
        "private GpuIntBuffer ceTargetsDevice;",
        "private int ceTargetsCapRows;",
        "private int[] ceHostTargetScratch;",
        "private GpuIntBuffer sampledCandidateIdsDevice;",
        "private int sampledCandidateIdsCapElems;",
        "private GpuFloatBuffer sampledCandidateLogitsDevice;",
        "private GpuFloatBuffer sampledCandidateGradDevice;",
        "private int sampledCandidateFloatCapElems;",
        "private int[] sampledCandidateIdsHostScratch;",
        "private int[] sampledSharedNegativeScratch;",
        "private int sampledTrainCandidatesPerRow;",
        "private float[] evalCeGradScratch;",
        "private boolean gpuTrainableParamGradsKnownClean;",
        "private GpuFloatBuffer[] nonFiniteParamBufsScratch = new GpuFloatBuffer[0];",
        "private int[] nonFiniteParamLensScratch = new int[0];",
        "private long[] sumSqPtrsScratch = new long[0];",
        "private int[] sumSqLensScratch = new int[0];",
        "private final List<Tensor> logitsOnlyScratch = new ArrayList<>(1);",
        "private final boolean checkpointAsyncIo;",
        "private final ExecutorService checkpointIoExecutor;",
        "private volatile CompletableFuture<Void> checkpointIoTail = CompletableFuture.completedFuture(null);",
        "private float lastGlobalGradNorm;",
        "private final TrainingEventCallback trainingEventCallback;",
        "private final TrainingStatsWriter trainingStatsWriter;",
        "private volatile boolean supervisedStopRequested;",
        "private volatile boolean exitedDueToSupervisorRequest;",
    ]
    for decl in field_decls:
        pkg = decl.replace("private ", "", 1)
        if decl not in text:
            raise SystemExit(f"field decl not found: {decl}")
        text = text.replace(decl, pkg, 1)

    text = text.replace(
        "private void synchronizeTrainingPipelineAfterGpuAuxiliaryInfer(String reason) {",
        "void synchronizeTrainingPipelineAfterGpuAuxiliaryInfer(String reason) {",
        1,
    )
    text = text.replace("private float lossScaleForForward() {", "float lossScaleForForward() {", 1)
    text = text.replace(
        "private void syncShutdownProgressBaselineFromGlobalStep() {",
        "void syncShutdownProgressBaselineFromGlobalStep() {",
        1,
    )

    text = text.replace("{@link #ceFusedGradScaleOverTotal}", "{@link LlmTrainerCrossEntropy#ceFusedGradScaleOverTotal}")

    text = text.replace("import com.veles.llm.jgpt.app.LlmTextGeneration;\n", "")

    p.write_text(text, encoding="utf-8")
    print("OK:", p)


if __name__ == "__main__":
    main()
