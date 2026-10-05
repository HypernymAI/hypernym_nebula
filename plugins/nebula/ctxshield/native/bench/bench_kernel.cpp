// Stage-split profiler for the CPU kernel (no Python): tokenize only / + hash / full featurize+score.
//   build-*/shield_bench <docs.bin> [reps]
// docs.bin: u32 n_docs, u32 n_fields, n_fields × (u32 len + name), then per doc per field u32 len + UTF-8.
// Written by native/bench/bench_cpu.py --dump.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "shield.h"

using namespace ctxshield;
namespace ctxshield { extern double g_prof[8]; }

static uint32_t rd32(std::ifstream& f) {
    uint32_t v;
    f.read(reinterpret_cast<char*>(&v), 4);
    if (!f) throw std::runtime_error("short read");
    return v;
}
static std::string rds(std::ifstream& f) {
    uint32_t n = rd32(f);
    std::string s(n, '\0');
    f.read(s.data(), n);
    if (!f) throw std::runtime_error("short read");
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: shield_bench docs.bin [reps]\n"); return 2; }
    int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    std::ifstream f(argv[1], std::ios::binary);
    uint32_t n = rd32(f), nf = rd32(f);
    std::vector<std::string> names;
    for (uint32_t i = 0; i < nf; ++i) names.push_back(rds(f));
    std::vector<std::string> texts(static_cast<size_t>(n) * nf);
    size_t chars = 0;
    for (auto& t : texts) { t = rds(f); chars += t.size(); }
    std::vector<std::string_view> V(texts.begin(), texts.end());
    FieldSet fs(names);
    Scratch sc;
    std::vector<float> W(static_cast<size_t>(kDim) * 4, 0.01f), We(static_cast<size_t>(fs.n_extras()) * 4, 0.1f), b(4, 0.f);
    float lg[16];
    for (int dl = 0; dl < 2; ++dl) {
        double best_tok = 1e30, best_full = 1e30;
        size_t adds = 0, ntok = 0;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            adds = 0; ntok = 0;
            for (uint32_t i = 0; i < n; ++i) { adds += tokenize_doc(fs, V.data() + static_cast<size_t>(i) * nf, dl, sc); ntok += sc.toks.size(); }
            auto t1 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < n; ++i)
                featurize_score(fs, V.data() + static_cast<size_t>(i) * nf, dl, sc, W.data(), We.data(), b.data(), 4, lg);
            auto t2 = std::chrono::steady_clock::now();
            std::fill(g_prof, g_prof + 8, 0.0);
            for (uint32_t i = 0; i < n; ++i)
                featurize_score(fs, V.data() + static_cast<size_t>(i) * nf, dl, sc, W.data(), We.data(), b.data(), 4, lg);
            if (r == reps - 1)
                std::printf("  stages us/doc: tokenize %.2f | hash %.2f | insert %.2f | finalize %.2f | extras %.2f | gather-dot %.2f | (V2: nfkc %.2f delex+tok %.2f)\n",
                            g_prof[0] * 1e6 / n, g_prof[1] * 1e6 / n, g_prof[2] * 1e6 / n, g_prof[3] * 1e6 / n, g_prof[4] * 1e6 / n, g_prof[5] * 1e6 / n, g_prof[6] * 1e6 / n, g_prof[7] * 1e6 / n);
            best_tok = std::min(best_tok, std::chrono::duration<double>(t1 - t0).count());
            best_full = std::min(best_full, std::chrono::duration<double>(t2 - t1).count());
        }
        std::printf("%s: %u docs, %.0f bytes/doc, %.0f tokens/doc, %.0f adds/doc | tokenize %.2f us/doc | full %.2f us/doc (%.2f ns/byte, %.2f ns/add)\n",
                    dl ? "V2" : "V1", n, double(chars) / n, double(ntok) / n, double(adds) / n,
                    best_tok * 1e6 / n, best_full * 1e6 / n, best_full * 1e9 / chars, best_full * 1e9 / adds);
    }
    return 0;
}
