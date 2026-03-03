/**
 * preproc_runner/main.cpp
 * -----------------------
 * C++ binary that loads the TorchScript FeaturePreproc module and runs it
 * on a synthetic batch, demonstrating C++ inference with the same .pt file
 * used in Python training.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
 *   cmake --build . --config Release
 *
 * Run:
 *   ./build/preproc_runner artifacts/preproc.pt [--batch 128] [--dense 13] [--sparse 26]
 *
 * In a production system, this binary would:
 *   1. Read raw feature tensors from a message queue (Kafka, SQS, etc.)
 *   2. Run the TorchScript preproc module
 *   3. Write preprocessed tensors to shared memory or a downstream queue
 *      for the trainer/server to consume.
 */

#include <torch/script.h>   // torch::jit::load
#include <torch/torch.h>    // tensor ops

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

/**
 * Generate one synthetic batch of DLRM-like features.
 * Mirrors data/synthetic.py but in C++ using LibTorch.
 */
struct SyntheticBatch {
    torch::Tensor              dense;           // [B, D]
    std::vector<torch::Tensor> sparse_indices;  // [total_ids_i] per feature
    std::vector<torch::Tensor> sparse_offsets;  // [B] per feature
};

SyntheticBatch make_batch(int batch, int num_dense, int num_sparse,
                           int avg_bag_len, const std::vector<int>& vocab_sizes) {
    SyntheticBatch b;

    // Dense: half exponential, half gaussian
    int half = num_dense / 2;
    auto exp_part  = torch::empty({batch, half}).exponential_(2.0f);  // Exp(0.5)
    auto norm_part = torch::randn({batch, num_dense - half}) * 3.0f;
    b.dense = torch::cat({exp_part, norm_part}, /*dim=*/1);

    // Sparse
    for (int f = 0; f < num_sparse; ++f) {
        int vocab = vocab_sizes[f];
        auto lengths = torch::randint(1, avg_bag_len * 2 + 1, {batch});  // [B]
        int total = lengths.sum().item<int>();

        // Power-law IDs: (rand^2 * vocab * 2) % vocab
        auto raw_ids = (torch::rand({total}).pow(2) * vocab * 2).to(torch::kLong);
        auto indices = raw_ids % vocab;

        // Compute EmbeddingBag-style offsets [B]
        auto lengths_long = lengths.to(torch::kLong);
        auto offsets = torch::zeros({batch}, torch::kLong);
        auto cumsum  = lengths_long.cumsum(0);
        if (batch > 1) {
            offsets.slice(0, 1) = cumsum.slice(0, 0, batch - 1);
        }

        b.sparse_indices.push_back(indices);
        b.sparse_offsets.push_back(offsets);
    }
    return b;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // ── args ──────────────────────────────────────────────────────────────────
    std::string module_path = "artifacts/preproc.pt";
    int batch       = 128;
    int num_dense   = 13;
    int num_sparse  = 26;
    int avg_bag_len = 5;
    int warmup      = 10;
    int iters       = 50;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--batch"  && i+1 < argc) batch       = std::stoi(argv[++i]);
        else if (a == "--dense"  && i+1 < argc) num_dense   = std::stoi(argv[++i]);
        else if (a == "--sparse" && i+1 < argc) num_sparse  = std::stoi(argv[++i]);
        else if (a == "--warmup" && i+1 < argc) warmup      = std::stoi(argv[++i]);
        else if (a == "--iters"  && i+1 < argc) iters       = std::stoi(argv[++i]);
        else if (a.rfind("--", 0) != 0)          module_path = a;  // positional
    }

    std::cout << "=== DLRM Preprocessing Runner (C++ / LibTorch) ===" << std::endl;
    std::cout << "module:  " << module_path << std::endl;
    std::cout << "batch:   " << batch       << "  dense: " << num_dense
              << "  sparse: " << num_sparse << std::endl;

    // ── load TorchScript module ───────────────────────────────────────────────
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(module_path);
        module.eval();
        std::cout << "Loaded preproc module OK" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading " << module_path << ": " << e.what() << std::endl;
        return 1;
    }

    // ── Criteo-like vocab sizes (capped at 200 000) ───────────────────────────
    std::vector<int> vocab_sizes = {
        1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3,
        93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652,
        2173, 4, 7046547, 18, 15, 286181, 105, 142572
    };
    for (auto& v : vocab_sizes) v = std::min(v, 200000);
    // Trim / pad to num_sparse
    vocab_sizes.resize(num_sparse, 100000);

    // ── pre-generate a batch (outside timing) ─────────────────────────────────
    auto batch_data = make_batch(batch, num_dense, num_sparse, avg_bag_len, vocab_sizes);

    // ── build IValue inputs for module.forward() ──────────────────────────────
    // FeaturePreproc.forward(dense, sparse_indices: List[Tensor], sparse_offsets: List[Tensor])
    //
    // PyTorch 2.x requires typed c10::List<at::Tensor> for List[Tensor] args.
    // Constructing IValue from std::vector<IValue> creates a generic
    // c10::List<IValue> which is rejected by a static assertion since PyTorch 2.0.
    auto to_ivalue_list = [](const std::vector<torch::Tensor>& v) -> torch::IValue {
        c10::List<at::Tensor> list;
        list.reserve(v.size());
        for (const auto& t : v) list.push_back(t);
        return torch::IValue(list);
    };

    std::vector<torch::IValue> inputs = {
        batch_data.dense,
        to_ivalue_list(batch_data.sparse_indices),
        to_ivalue_list(batch_data.sparse_offsets),
    };

    // ── warmup ────────────────────────────────────────────────────────────────
    std::cout << "\nWarmup (" << warmup << " iters) …" << std::endl;
    for (int i = 0; i < warmup; ++i) {
        module.forward(inputs);
    }

    // ── timed run ─────────────────────────────────────────────────────────────
    std::cout << "Timing (" << iters << " iters) …" << std::endl;
    double t_start = now_ms();
    for (int i = 0; i < iters; ++i) {
        module.forward(inputs);
    }
    double elapsed = now_ms() - t_start;
    double avg_ms  = elapsed / iters;

    // ── inspect output ────────────────────────────────────────────────────────
    auto output = module.forward(inputs);
    auto elems  = output.toTuple()->elements();
    auto dense_out   = elems[0].toTensor();
    auto idx_list    = elems[1].toListRef();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "avg latency:      " << avg_ms << " ms / batch" << std::endl;
    std::cout << "throughput:       " << (1000.0 / avg_ms) << " batches/s" << std::endl;
    std::cout << "dense_out shape:  [" << dense_out.size(0) << ", "
              << dense_out.size(1) << "]" << std::endl;
    std::cout << "dense_out mean:   " << dense_out.mean().item<float>() << std::endl;
    std::cout << "dense_out std:    " << dense_out.std().item<float>()  << std::endl;
    std::cout << "dense_out range:  ["
              << dense_out.min().item<float>() << ", "
              << dense_out.max().item<float>() << "]" << std::endl;
    std::cout << "sparse features:  " << idx_list.size() << " features processed" << std::endl;

    auto first_idx = idx_list[0].toTensor();
    std::cout << "sparse[0] IDs:    total=" << first_idx.size(0)
              << "  max_id=" << first_idx.max().item<int64_t>()
              << "  min_id=" << first_idx.min().item<int64_t>() << std::endl;

    std::cout << "\nPreprocessing runner finished successfully." << std::endl;
    return 0;
}
