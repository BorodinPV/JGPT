#include "jgpt_cudnn_sdpa.h"

#ifndef JGPT_HAS_CUDNN

int jgpt_cudnn_sdpa_available(void) { return 0; }
int jgpt_cudnn_sdpa_fwd(
        const float*,
        const float*,
        const float*,
        float*,
        float*,
        int,
        int,
        int,
        int,
        float) {
    return 0;
}
int jgpt_cudnn_sdpa_bwd(
        const float*,
        const float*,
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
        float*,
        int,
        int,
        int,
        int,
        float) {
    return 0;
}
int jgpt_cudnn_sdpa_fwd_half(
        const void*,
        const void*,
        const void*,
        void*,
        float*,
        int,
        int,
        int,
        int,
        float) {
    return 0;
}
int jgpt_cudnn_sdpa_bwd_half(
        const void*,
        const void*,
        const void*,
        const void*,
        const void*,
        const float*,
        void*,
        void*,
        void*,
        int,
        int,
        int,
        int,
        float) {
    return 0;
}
void jgpt_cudnn_sdpa_prewarm(int, int, int, int, float) {}
void jgpt_cudnn_sdpa_cleanup(void) {}

#else

#include "jgpt_cudnn_convert.h"
#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_tls_blob.cuh"

#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t kUidQ = 1;
constexpr int64_t kUidK = 2;
constexpr int64_t kUidV = 3;
constexpr int64_t kUidO = 4;
constexpr int64_t kUidStats = 5;
constexpr int64_t kUidDO = 101;
constexpr int64_t kUidDQ = 102;
constexpr int64_t kUidDK = 103;
constexpr int64_t kUidDV = 104;

struct GraphSlot {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_bytes = 0;
};

struct Key {
    int bwd;
    int batch;
    int nHeads;
    int seq;
    int dHead;
    uint32_t scale_bits;
    bool operator==(const Key& o) const {
        return bwd == o.bwd && batch == o.batch && nHeads == o.nHeads && seq == o.seq && dHead == o.dHead
                && scale_bits == o.scale_bits;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = (size_t) k.bwd;
        h = h * 131u + (size_t) k.batch;
        h = h * 131u + (size_t) k.nHeads;
        h = h * 131u + (size_t) k.seq;
        h = h * 131u + (size_t) k.dHead;
        h = h * 131u + (size_t) k.scale_bits;
        return h;
    }
};

thread_local cudnnHandle_t tl_handle = nullptr;
thread_local std::unordered_map<Key, GraphSlot, KeyHash> tl_graphs;
thread_local jgpt_cuda_tls::TlsDeviceBlob tl_ws;
thread_local jgpt_cuda_tls::TlsDeviceBlob tl_half;
thread_local jgpt_cuda_tls::TlsDeviceBlob tl_dummy_stats;
int g_disabled = 0;
int g_logged = 0;
std::mutex g_log_mu;

uint32_t scale_bits(float s) {
    uint32_t u = 0;
    std::memcpy(&u, &s, sizeof(u));
    return u;
}

int env_disabled() {
    const char* e = std::getenv("JGPT_CUDNN_SDPA");
    if (e == nullptr || e[0] == '\0') {
        return 0;
    }
    return (e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F') ? 1 : 0;
}

bool stream_capturing() {
    jgpt_cuda_ensure_stream();
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaError_t e = cudaStreamGetCaptureInfo(kTensorCudaStream, &cap, nullptr);
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return cap == cudaStreamCaptureStatusActive;
}

void clear_cuda_error_if_capturing() {
    if (stream_capturing()) {
        (void) cudaGetLastError();
    }
}

bool blob_ensure(jgpt_cuda_tls::TlsDeviceBlob& blob, size_t bytes) {
    if (bytes == 0U) {
        return true;
    }
    if (blob.ptr != nullptr && blob.bytes >= bytes) {
        return true;
    }
    if (stream_capturing()) {
        return false;
    }
    return blob.grow_to_fit(bytes);
}

cudnnHandle_t handle() {
    if (tl_handle != nullptr) {
        return tl_handle;
    }
    if (stream_capturing()) {
        return nullptr;
    }
    if (cudnnCreate(&tl_handle) != CUDNN_STATUS_SUCCESS) {
        tl_handle = nullptr;
        return nullptr;
    }
    return tl_handle;
}

void bind_stream() {
    cudnnHandle_t h = handle();
    if (h == nullptr) {
        return;
    }
    jgpt_cuda_ensure_stream();
    (void) cudnnSetStream(h, kTensorCudaStream);
}

bool log_err(const char* where, fe::error_t st) {
    fprintf(stderr, "jgpt cuDNN SDPA %s: %s\n", where, st.get_message().c_str());
    return false;
}

/* BHSD: [B,H,S,Dh] contiguous ≡ Java [B*H, S, Dh] (head after batch). */
std::shared_ptr<fe::graph::Graph> make_fwd_graph(int batch, int nHeads, int seq, int d, float scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b = batch;
    const int64_t h = nHeads;
    const int64_t s = seq;
    const int64_t dh = d;
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(kUidQ)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(kUidK)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(kUidV)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto opts = fe::graph::SDPA_attributes()
                        .set_name("jgpt_sdpa")
                        .set_generate_stats(true)
                        .set_attn_scale(scale)
                        .set_causal_mask(true);
    auto [O, Stats] = graph->sdpa(Q, K, V, opts);
    O->set_output(true).set_uid(kUidO).set_dim({b, h, s, dh}).set_stride({h * s * dh, s * dh, dh, 1});
    Stats->set_output(true)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(kUidStats)
            .set_dim({b, h, s, 1})
            .set_stride({h * s, s, 1, 1});
    return graph;
}

std::shared_ptr<fe::graph::Graph> make_bwd_graph(int batch, int nHeads, int seq, int d, float scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b = batch;
    const int64_t h = nHeads;
    const int64_t s = seq;
    const int64_t dh = d;
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(kUidQ)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(kUidK)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(kUidV)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("O")
                                   .set_uid(kUidO)
                                   .set_dim({b, h, s, dh})
                                   .set_stride({h * s * dh, s * dh, dh, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("dO")
                                    .set_uid(kUidDO)
                                    .set_dim({b, h, s, dh})
                                    .set_stride({h * s * dh, s * dh, dh, 1}));
    auto Stats = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("Stats")
                                       .set_uid(kUidStats)
                                       .set_dim({b, h, s, 1})
                                       .set_stride({h * s, s, 1, 1})
                                       .set_data_type(fe::DataType_t::FLOAT));
    auto opts = fe::graph::SDPA_backward_attributes()
                        .set_name("jgpt_sdpa_bwd")
                        .set_attn_scale(scale)
                        .set_causal_mask(true);
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, opts);
    dQ->set_output(true).set_uid(kUidDQ).set_dim({b, h, s, dh}).set_stride({h * s * dh, s * dh, dh, 1});
    dK->set_output(true).set_uid(kUidDK).set_dim({b, h, s, dh}).set_stride({h * s * dh, s * dh, dh, 1});
    dV->set_output(true).set_uid(kUidDV).set_dim({b, h, s, dh}).set_stride({h * s * dh, s * dh, dh, 1});
    return graph;
}

GraphSlot* get_or_build(int bwd, int batch, int nHeads, int seq, int d, float scale) {
    if (g_disabled || env_disabled() || batch <= 0 || nHeads <= 0 || seq <= 0 || d <= 0 || (d % 8) != 0) {
        return nullptr;
    }
    cudnnHandle_t h = handle();
    if (h == nullptr) {
        return nullptr;
    }
    bind_stream();
    Key key{bwd, batch, nHeads, seq, d, scale_bits(scale)};
    auto it = tl_graphs.find(key);
    if (it != tl_graphs.end()) {
        return &it->second;
    }
    if (stream_capturing()) {
        /* Первая сборка графа — cudaMalloc; только после prewarm. */
        return nullptr;
    }
    GraphSlot slot;
    slot.graph = bwd ? make_bwd_graph(batch, nHeads, seq, d, scale) : make_fwd_graph(batch, nHeads, seq, d, scale);
    auto st = slot.graph->validate();
    if (!st.is_good()) {
        log_err("validate", st);
        g_disabled = 1;
        return nullptr;
    }
    st = slot.graph->build(h, {fe::HeurMode_t::A});
    if (!st.is_good()) {
        log_err("build", st);
        g_disabled = 1;
        return nullptr;
    }
    st = slot.graph->get_workspace_size(slot.workspace_bytes);
    if (!st.is_good()) {
        log_err("workspace", st);
        g_disabled = 1;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_log_mu);
        if (!g_logged) {
            g_logged = 1;
            fprintf(stderr,
                    "[jgpt] cuDNN SDPA FlashAttention (cudnn %zu, B=%d H=%d S=%d Dh=%d, ws=%lld B, packed half I/O)\n",
                    (size_t) cudnnGetVersion(),
                    batch,
                    nHeads,
                    seq,
                    d,
                    (long long) slot.workspace_bytes);
        }
    }
    auto ins = tl_graphs.emplace(key, std::move(slot));
    return &ins.first->second;
}

__half* half_base(size_t n_half, int slots) {
    const size_t bytes = n_half * sizeof(__half) * (size_t) slots;
    if (!blob_ensure(tl_half, bytes)) {
        return nullptr;
    }
    return static_cast<__half*>(tl_half.ptr);
}

void* workspace_ptr(int64_t bytes) {
    if (bytes <= 0) {
        return nullptr;
    }
    if (!blob_ensure(tl_ws, (size_t) bytes)) {
        return nullptr;
    }
    return tl_ws.ptr;
}

int execute_fwd_packed(GraphSlot* slot, __half* qh, __half* kh, __half* vh, __half* oh, float* stats, int disable_on_fail) {
    bind_stream();
    void* ws = workspace_ptr(slot->workspace_bytes);
    if (slot->workspace_bytes > 0 && ws == nullptr) {
        return 0;
    }
    std::unordered_map<int64_t, void*> pack = {
            {kUidQ, qh}, {kUidK, kh}, {kUidV, vh}, {kUidO, oh}, {kUidStats, stats}};
    auto st = slot->graph->execute(handle(), pack, ws);
    if (!st.is_good()) {
        log_err("execute fwd", st);
        if (disable_on_fail) {
            g_disabled = 1;
        }
        clear_cuda_error_if_capturing();
        return 0;
    }
    return 1;
}

int execute_bwd_packed(
        GraphSlot* slot,
        __half* qh,
        __half* kh,
        __half* vh,
        __half* oh,
        __half* doh,
        float* stats,
        __half* dqh,
        __half* dkh,
        __half* dvh,
        int disable_on_fail) {
    bind_stream();
    void* ws = workspace_ptr(slot->workspace_bytes);
    if (slot->workspace_bytes > 0 && ws == nullptr) {
        return 0;
    }
    std::unordered_map<int64_t, void*> pack = {
            {kUidQ, qh},
            {kUidK, kh},
            {kUidV, vh},
            {kUidO, oh},
            {kUidDO, doh},
            {kUidStats, stats},
            {kUidDQ, dqh},
            {kUidDK, dkh},
            {kUidDV, dvh}};
    auto st = slot->graph->execute(handle(), pack, ws);
    if (!st.is_good()) {
        log_err("execute bwd", st);
        if (disable_on_fail) {
            g_disabled = 1;
        }
        clear_cuda_error_if_capturing();
        return 0;
    }
    return 1;
}

}  // namespace

int jgpt_cudnn_sdpa_available(void) {
    if (g_disabled || env_disabled()) {
        return 0;
    }
    return handle() != nullptr ? 1 : 0;
}

int jgpt_cudnn_sdpa_fwd_half(
        const void* q,
        const void* k,
        const void* v,
        void* o,
        float* stats,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale) {
    GraphSlot* slot = get_or_build(0, batch, nHeads, seq, dHead, scale);
    if (slot == nullptr || q == nullptr || k == nullptr || v == nullptr || o == nullptr || stats == nullptr) {
        return 0;
    }
    return execute_fwd_packed(
            slot,
            static_cast<__half*>(const_cast<void*>(q)),
            static_cast<__half*>(const_cast<void*>(k)),
            static_cast<__half*>(const_cast<void*>(v)),
            static_cast<__half*>(o),
            stats,
            1);
}

int jgpt_cudnn_sdpa_bwd_half(
        const void* q,
        const void* k,
        const void* v,
        const void* o,
        const void* dO,
        const float* stats,
        void* dQ,
        void* dK,
        void* dV,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale) {
    GraphSlot* slot = get_or_build(1, batch, nHeads, seq, dHead, scale);
    if (slot == nullptr || q == nullptr || k == nullptr || v == nullptr || o == nullptr || dO == nullptr
            || stats == nullptr || dQ == nullptr || dK == nullptr || dV == nullptr) {
        return 0;
    }
    return execute_bwd_packed(
            slot,
            static_cast<__half*>(const_cast<void*>(q)),
            static_cast<__half*>(const_cast<void*>(k)),
            static_cast<__half*>(const_cast<void*>(v)),
            static_cast<__half*>(const_cast<void*>(o)),
            static_cast<__half*>(const_cast<void*>(dO)),
            const_cast<float*>(stats),
            static_cast<__half*>(dQ),
            static_cast<__half*>(dK),
            static_cast<__half*>(dV),
            1);
}

int jgpt_cudnn_sdpa_fwd(
        const float* q,
        const float* k,
        const float* v,
        float* o,
        float* stats,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale) {
    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr || stats == nullptr) {
        return 0;
    }
    const size_t n = (size_t) batch * (size_t) nHeads * (size_t) seq * (size_t) dHead;
    __half* base = half_base(n, 4);
    if (base == nullptr) {
        return 0;
    }
    __half* qh = base;
    __half* kh = qh + n;
    __half* vh = kh + n;
    __half* oh = vh + n;
    jgpt_extra_f32_to_f16(q, qh, n);
    jgpt_extra_f32_to_f16(k, kh, n);
    jgpt_extra_f32_to_f16(v, vh, n);
    if (!jgpt_cudnn_sdpa_fwd_half(qh, kh, vh, oh, stats, batch, nHeads, seq, dHead, scale)) {
        return 0;
    }
    jgpt_extra_f16_to_f32(oh, o, n);
    return 1;
}

int jgpt_cudnn_sdpa_bwd(
        const float* q,
        const float* k,
        const float* v,
        const float* o,
        const float* dO,
        const float* stats,
        float* dQ,
        float* dK,
        float* dV,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale) {
    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr || dO == nullptr || stats == nullptr
            || dQ == nullptr || dK == nullptr || dV == nullptr) {
        return 0;
    }
    const size_t n = (size_t) batch * (size_t) nHeads * (size_t) seq * (size_t) dHead;
    __half* base = half_base(n, 8);
    if (base == nullptr) {
        return 0;
    }
    __half* qh = base;
    __half* kh = qh + n;
    __half* vh = kh + n;
    __half* oh = vh + n;
    __half* doh = oh + n;
    __half* dqh = doh + n;
    __half* dkh = dqh + n;
    __half* dvh = dkh + n;
    jgpt_extra_f32_to_f16(q, qh, n);
    jgpt_extra_f32_to_f16(k, kh, n);
    jgpt_extra_f32_to_f16(v, vh, n);
    jgpt_extra_f32_to_f16(o, oh, n);
    jgpt_extra_f32_to_f16(dO, doh, n);
    if (!jgpt_cudnn_sdpa_bwd_half(qh, kh, vh, oh, doh, stats, dqh, dkh, dvh, batch, nHeads, seq, dHead, scale)) {
        return 0;
    }
    jgpt_extra_f16_to_f32(dqh, dQ, n);
    jgpt_extra_f16_to_f32(dkh, dK, n);
    jgpt_extra_f16_to_f32(dvh, dV, n);
    return 1;
}

void jgpt_cudnn_sdpa_prewarm(int batch, int nHeads, int seq, int dHead, float scale) {
    if (stream_capturing()) {
        return;
    }
    GraphSlot* fwd = get_or_build(0, batch, nHeads, seq, dHead, scale);
    GraphSlot* bwd = get_or_build(1, batch, nHeads, seq, dHead, scale);
    if (fwd == nullptr) {
        return;
    }
    const size_t n = (size_t) batch * (size_t) nHeads * (size_t) seq * (size_t) dHead;
    __half* base = half_base(n, 8);
    if (base == nullptr) {
        return;
    }
    const size_t stats_bytes = (size_t) batch * (size_t) nHeads * (size_t) seq * sizeof(float);
    if (!blob_ensure(tl_dummy_stats, stats_bytes)) {
        return;
    }
    int64_t ws_need = fwd->workspace_bytes;
    if (bwd != nullptr && bwd->workspace_bytes > ws_need) {
        ws_need = bwd->workspace_bytes;
    }
    if (ws_need > 0 && workspace_ptr(ws_need) == nullptr) {
        return;
    }
    jgpt_cuda_ensure_stream();
    (void) cudaMemsetAsync(base, 0, n * 8U * sizeof(__half), kTensorCudaStream);
    (void) cudaMemsetAsync(tl_dummy_stats.ptr, 0, stats_bytes, kTensorCudaStream);
    __half* qh = base;
    __half* kh = qh + n;
    __half* vh = kh + n;
    __half* oh = vh + n;
    __half* doh = oh + n;
    __half* dqh = doh + n;
    __half* dkh = dqh + n;
    __half* dvh = dkh + n;
    float* stats = static_cast<float*>(tl_dummy_stats.ptr);
    if (!execute_fwd_packed(fwd, qh, kh, vh, oh, stats, 0)) {
        return;
    }
    if (bwd != nullptr) {
        (void) execute_bwd_packed(bwd, qh, kh, vh, oh, doh, stats, dqh, dkh, dvh, 0);
    }
    (void) jgpt_cuda_sync_stream_unless_capturing("jgpt_cudnn_sdpa_prewarm");
}

void jgpt_cudnn_sdpa_cleanup(void) {
    tl_graphs.clear();
    tl_ws.free_cached();
    tl_half.free_cached();
    tl_dummy_stats.free_cached();
    if (tl_handle != nullptr) {
        cudnnDestroy(tl_handle);
        tl_handle = nullptr;
    }
}

#endif
