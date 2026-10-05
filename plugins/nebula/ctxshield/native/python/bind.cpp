// pybind11 module `ctxshield._native` (build doc §9.1). Batch calls take UTF-8 views of the Python
// strings under the GIL (PyUnicode_AsUTF8AndSize: zero-copy for ASCII str, cached UTF-8 otherwise),
// then release it and run on a fixed fan-out (contiguous chunks, no locks).
//
// Step 8: `Scorer` = a handle over the plain-C ABI (include/shield.h) for backend 'cpu' | 'metal' |
// 'cuda'; `score_batch_metal` = one-shot Metal scoring. A backend that is not built / not available
// raises — never a silent CPU fallback. `time_score` = in-kernel per-message latency (bench).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../src/backend.h"
#include "../src/parallel.h"
#include "shield.h"

namespace py = pybind11;
using namespace ctxshield;

namespace {

using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;

// docs: list of lists of str (each list aligned with the sorted field names). The views point into
// the str objects' UTF-8 buffers; `docs` keeps them alive for the duration of the call.
std::vector<std::string_view> views(const py::list& docs, size_t nfields) {
    std::vector<std::string_view> out;
    out.reserve(docs.size() * nfields);
    for (const py::handle& d : docs) {
        PyObject* seq = d.ptr();
        if (!PyList_Check(seq) && !PyTuple_Check(seq)) throw std::runtime_error("ctxshield: each doc must be a list of str");
        const Py_ssize_t k = PySequence_Fast_GET_SIZE(seq);
        if (static_cast<size_t>(k) != nfields) throw std::runtime_error("ctxshield: doc has wrong number of fields");
        PyObject** items = PySequence_Fast_ITEMS(seq);
        for (Py_ssize_t j = 0; j < k; ++j) {
            if (!PyUnicode_Check(items[j])) throw std::runtime_error("ctxshield: field text must be str");
            Py_ssize_t n = 0;
            const char* p = PyUnicode_AsUTF8AndSize(items[j], &n);
            if (!p) throw py::error_already_set();
            out.emplace_back(p, static_cast<size_t>(n));
        }
    }
    return out;
}

void check_weights(const FieldSet& fs, const FArr& W, const FArr& We, const FArr& b) {
    if (W.ndim() != 2 || W.shape(0) != static_cast<py::ssize_t>(kDim))
        throw std::runtime_error("ctxshield: W must be [2^18, hpad]");
    const py::ssize_t hpad = W.shape(1);
    if (hpad <= 0 || hpad > kMaxHeadsPad || (hpad & 3) != 0)
        throw std::runtime_error("ctxshield: hpad must be 4, 8, 12 or 16");
    if (We.ndim() != 2 || We.shape(0) != fs.n_extras() || We.shape(1) != hpad)
        throw std::runtime_error("ctxshield: We must be [n_extras, hpad]");
    if (b.ndim() != 1 || b.shape(0) != hpad) throw std::runtime_error("ctxshield: b must be [hpad]");
}

py::tuple featurize_batch(const std::vector<std::string>& field_names, const py::list& docs,
                          bool delex, int threads) {
    FieldSet fs(field_names);
    const size_t nf = field_names.size();
    std::vector<std::string_view> V = views(docs, nf);
    const size_t n = docs.size();
    const int ne = fs.n_extras();
    std::vector<Features> feats(n);
    {
        py::gil_scoped_release release;
        parallel_chunks(n, threads, [&](size_t a, size_t b) {
            Scratch& sc = thread_scratch();
            for (size_t i = a; i < b; ++i) featurize(fs, V.data() + i * nf, delex, sc, feats[i]);
        });
    }
    size_t nnz = 0;
    for (auto& f : feats) nnz += f.idx.size();
    py::array_t<int64_t> indptr(static_cast<py::ssize_t>(n + 1));
    py::array_t<int32_t> indices(static_cast<py::ssize_t>(nnz));
    py::array_t<float> data(static_cast<py::ssize_t>(nnz));
    py::array_t<float> extras({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(ne)});
    auto ip = indptr.mutable_data();
    auto ix = indices.mutable_data();
    auto dv = data.mutable_data();
    auto ex = extras.mutable_data();
    size_t pos = 0;
    ip[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < feats[i].idx.size(); ++k) {
            ix[pos] = static_cast<int32_t>(feats[i].idx[k]);
            dv[pos] = feats[i].val[k];
            ++pos;
        }
        ip[i + 1] = static_cast<int64_t>(pos);
        std::memcpy(ex + i * static_cast<size_t>(ne), feats[i].extras.data(), sizeof(float) * static_cast<size_t>(ne));
    }
    return py::make_tuple(indptr, indices, data, extras);
}

py::array_t<float> score_batch(const std::vector<std::string>& field_names, const py::list& docs,
                               bool delex, FArr W, FArr We, FArr b, int threads) {
    FieldSet fs(field_names);
    check_weights(fs, W, We, b);
    const int hpad = static_cast<int>(W.shape(1));
    const size_t nf = field_names.size();
    std::vector<std::string_view> V = views(docs, nf);
    const size_t n = docs.size();
    py::array_t<float> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(hpad)});
    float* o = out.mutable_data();
    const float* w = W.data();
    const float* we = We.data();
    const float* bb = b.data();
    {
        py::gil_scoped_release release;
        parallel_chunks(n, threads, [&](size_t a, size_t e) {
            Scratch& sc = thread_scratch();
            for (size_t i = a; i < e; ++i)
                featurize_score(fs, V.data() + i * nf, delex, sc, w, we, bb, hpad, o + i * static_cast<size_t>(hpad));
        });
    }
    return out;
}

// In-kernel latency per message (featurize + all-heads gather-dot, one thread, no Python):
// every doc is timed once per pass, `reps` passes, per-doc median returned in ns.
py::array_t<int64_t> time_score(const std::vector<std::string>& field_names, const py::list& docs,
                                bool delex, FArr W, FArr We, FArr b, int reps) {
    FieldSet fs(field_names);
    check_weights(fs, W, We, b);
    if (reps < 1) throw std::runtime_error("ctxshield: reps must be >= 1");
    const int hpad = static_cast<int>(W.shape(1));
    const size_t nf = field_names.size();
    std::vector<std::string_view> V = views(docs, nf);
    const size_t n = docs.size();
    std::vector<int64_t> t(n * static_cast<size_t>(reps));
    {
        py::gil_scoped_release release;
        Scratch& sc = thread_scratch();
        float logits[kMaxHeadsPad];
        for (size_t i = 0; i < std::min<size_t>(n, 200); ++i)   // warm
            featurize_score(fs, V.data() + i * nf, delex, sc, W.data(), We.data(), b.data(), hpad, logits);
        for (int r = 0; r < reps; ++r)
            for (size_t i = 0; i < n; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                featurize_score(fs, V.data() + i * nf, delex, sc, W.data(), We.data(), b.data(), hpad, logits);
                auto t1 = std::chrono::steady_clock::now();
                t[i * static_cast<size_t>(reps) + static_cast<size_t>(r)] =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            }
    }
    py::array_t<int64_t> out(static_cast<py::ssize_t>(n));
    auto o = out.mutable_data();
    for (size_t i = 0; i < n; ++i) {
        auto* p = t.data() + i * static_cast<size_t>(reps);
        std::nth_element(p, p + reps / 2, p + reps);
        o[i] = p[reps / 2];
    }
    return out;
}

int backend_code(const std::string& name) {
    if (name == "cpu") return SHIELD_BACKEND_CPU;
    if (name == "metal") return SHIELD_BACKEND_METAL;
    if (name == "cuda") return SHIELD_BACKEND_CUDA;
    throw std::runtime_error("ctxshield: unknown backend '" + name + "' (cpu | metal | cuda)");
}

// Scorer = C ABI handle (weights resident in the backend's memory).
class Scorer {
public:
    Scorer(const std::string& backend, const std::vector<std::string>& field_names, bool delex,
           FArr W, FArr We, FArr b)
        : backend_(backend), names_(field_names) {
        FieldSet fs(field_names);   // validates sorted + unique
        check_weights(fs, W, We, b);
        hpad_ = static_cast<int>(W.shape(1));
        std::vector<const char*> nm;
        for (auto& s : names_) nm.push_back(s.c_str());
        char err[1024] = {0};
        shield_scorer* h = nullptr;
        int rc;
        {
            py::gil_scoped_release release;
            rc = shield_scorer_create(backend_code(backend), nm.data(), static_cast<int>(nm.size()),
                                      delex ? 1 : 0, W.data(), We.data(), b.data(), hpad_, &h, err,
                                      sizeof(err));
        }
        if (rc != 0) throw std::runtime_error(err);
        h_ = h;
    }
    ~Scorer() { if (h_) shield_scorer_destroy(h_); }
    Scorer(const Scorer&) = delete;
    Scorer& operator=(const Scorer&) = delete;

    py::array_t<float> score(const py::list& docs, int threads) {
        const size_t nf = names_.size();
        std::vector<std::string_view> V = views(docs, nf);
        const size_t n = docs.size();
        std::vector<const char*> ptrs(V.size());
        std::vector<size_t> lens(V.size());
        for (size_t i = 0; i < V.size(); ++i) { ptrs[i] = V[i].data(); lens[i] = V[i].size(); }
        py::array_t<float> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(hpad_)});
        float* o = out.mutable_data();
        char err[1024] = {0};
        shield_stats st{};
        int rc;
        {
            py::gil_scoped_release release;
            rc = shield_scorer_score(h_, ptrs.data(), lens.data(), n, threads, o, &st, err, sizeof(err));
        }
        if (rc != 0) throw std::runtime_error(err);
        if (st.backend != backend_code(backend_))
            throw std::runtime_error("ctxshield: backend mismatch: requested " + backend_ + ", ran " +
                                     shield_backend_name(st.backend));
        last_ = py::dict();
        last_["backend"] = std::string(shield_backend_name(st.backend));
        last_["n_docs"] = st.n_docs;
        last_["n_overflow_docs"] = st.n_overflow_docs;
        last_["n_tokens"] = st.n_tokens;
        last_["n_arena_bytes"] = st.n_arena_bytes;
        last_["t_tokenize_s"] = st.t_tokenize_s;
        last_["t_device_s"] = st.t_device_s;
        last_["t_total_s"] = st.t_total_s;
        return out;
    }
    py::dict last_stats() const { return last_; }
    std::string backend() const { return backend_; }

private:
    std::string backend_;
    std::vector<std::string> names_;
    int hpad_ = 0;
    shield_scorer* h_ = nullptr;
    py::dict last_;
};

py::array_t<float> score_batch_metal(const std::vector<std::string>& field_names, const py::list& docs,
                                     bool delex, FArr W, FArr We, FArr b, int threads) {
    Scorer s("metal", field_names, delex, W, We, b);
    return s.score(docs, threads);
}

py::tuple backend_available(const std::string& name) {
    char why[512] = {0};
    int ok = shield_backend_available(backend_code(name), why, sizeof(why));
    return py::make_tuple(ok == 1, std::string(why));
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "ctxshield native core: featurize + all-heads gather-dot (CPU; Metal / CUDA via Scorer)";
    m.attr("DIM_BITS") = kDimBits;
    m.attr("ABI_VERSION") = shield_abi_version();
    m.attr("METAL_BUILT") = metal_built();
    m.attr("CUDA_BUILT") = cuda_built();
    m.def("utf8proc_version", []() { return std::string(utf8proc_version_string()); });
    m.def("unicode_version", []() { return std::string(unicode_version_string()); });
    m.def("default_threads", &default_threads);
    m.def("featurize_batch", &featurize_batch, py::arg("field_names"), py::arg("docs"),
          py::arg("delex"), py::arg("threads") = 0,
          "-> (indptr int64[n+1], indices int32[nnz], data float32[nnz], extras float32[n, e])");
    m.def("score_batch", &score_batch, py::arg("field_names"), py::arg("docs"), py::arg("delex"),
          py::arg("W"), py::arg("We"), py::arg("b"), py::arg("threads") = 0,
          "CPU -> logits float32[n, hpad]");
    m.def("time_score", &time_score, py::arg("field_names"), py::arg("docs"), py::arg("delex"),
          py::arg("W"), py::arg("We"), py::arg("b"), py::arg("reps") = 3,
          "per-doc in-kernel ns (median of reps), one thread");
    m.def("score_batch_metal", &score_batch_metal, py::arg("field_names"), py::arg("docs"),
          py::arg("delex"), py::arg("W"), py::arg("We"), py::arg("b"), py::arg("threads") = 0,
          "Metal -> logits float32[n, hpad]; raises if Metal is not built / not available");
    m.def("backend_available", &backend_available, py::arg("backend"), "-> (available, why_not)");
    py::class_<Scorer>(m, "Scorer")
        .def(py::init<const std::string&, const std::vector<std::string>&, bool, FArr, FArr, FArr>(),
             py::arg("backend"), py::arg("field_names"), py::arg("delex"), py::arg("W"), py::arg("We"),
             py::arg("b"))
        .def("score", &Scorer::score, py::arg("docs"), py::arg("threads") = 0)
        .def_property_readonly("backend", &Scorer::backend)
        .def_property_readonly("last_stats", &Scorer::last_stats);
}
