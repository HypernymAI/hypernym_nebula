// ctxshield-scorer — the native core as a standalone program (fully static on Linux: no libc, libstdc++
// or libgcc needed at run time, so ONE binary per CPU runs on every Linux — glibc or musl/Alpine).
//
// Python talks to it over stdin/stdout (ctxshield/_native_static.py). It is the same kernel as the
// pybind11 module (featurize.cpp / score.cpp / c_abi.cpp) — only the transport differs.
//
// Wire format (little-endian, one request → one response, strictly in order):
//   request  : "CXS1" | u32 op     | u64 len | payload[len]
//   response : "CXR1" | u32 status | u64 len | payload[len]      status 0 = ok, 1 = error (payload = message)
//   str      : u32 len | UTF-8 bytes
//   names    : u32 n | n × str                                  (sorted, unique — FieldSet validates)
//   docs     : u64 n_docs | u32 n_fields | u64 lens[n_docs × n_fields] | concatenated UTF-8
//
//   op 1 HELLO      ()                                   → u32 protocol | i32 dim_bits | i32 abi | i32 threads | str utf8proc | str unicode
//   op 2 AVAILABLE  i32 backend                          → u8 ok | str why
//   op 3 FEATURIZE  names | u8 delex | i32 threads | docs → u64 n | u64 nnz | u32 ne | i64 indptr[n+1] | i32 idx[nnz] | f32 val[nnz] | f32 extras[n×ne]
//   op 4 CREATE     i32 backend | names | u8 delex | i32 hpad | f32 W[2^18×hpad] | f32 We[ne×hpad] | f32 b[hpad] → u32 handle
//   op 5 SCORE      u32 handle | i32 threads | docs      → f32 logits[n×hpad] | i32 backend | u64 n_docs, n_overflow, n_tokens, n_arena | f64 t_tok, t_dev, t_total
//   op 6 DESTROY    u32 handle                           → ()
//
// A request that fails (bad weights, invalid UTF-8, unknown backend …) gets an error response and the
// process keeps serving. A broken STREAM (bad magic, short read) is fatal: message on stderr, exit 2.
// EOF between requests = the client is done: exit 0.
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "backend.h"
#include "parallel.h"
#include "shield.h"

using namespace ctxshield;

namespace {

constexpr uint32_t kProtocol = 1;
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "wire format is little-endian");

[[noreturn]] void fatal(const std::string& m) {
    std::fprintf(stderr, "ctxshield-scorer: %s\n", m.c_str());
    std::exit(2);
}

// Read exactly n bytes. Returns false only on EOF before the first byte (when eof_ok).
bool read_full(void* dst, size_t n, bool eof_ok) {
    auto* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(0, p + got, n - got);
        if (r < 0) {
            if (errno == EINTR) continue;
            fatal(std::string("read: ") + std::strerror(errno));
        }
        if (r == 0) {
            if (got == 0 && eof_ok) return false;
            fatal("short read (client closed mid-request)");
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

void write_full(const void* src, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    size_t put = 0;
    while (put < n) {
        ssize_t w = ::write(1, p + put, n - put);
        if (w < 0) {
            if (errno == EINTR) continue;
            fatal(std::string("write: ") + std::strerror(errno));
        }
        put += static_cast<size_t>(w);
    }
}

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    const uint8_t* take(size_t k) {
        if (k > n_ - i_) throw std::runtime_error("ctxshield-scorer: request payload too short");
        const uint8_t* r = p_ + i_;
        i_ += k;
        return r;
    }
    template <typename T> T pod() {
        T v;
        std::memcpy(&v, take(sizeof(T)), sizeof(T));
        return v;
    }
    std::string str() {
        const uint32_t k = pod<uint32_t>();
        const uint8_t* s = take(k);
        return std::string(reinterpret_cast<const char*>(s), k);
    }
    std::vector<std::string> names() {
        const uint32_t k = pod<uint32_t>();
        std::vector<std::string> v;
        v.reserve(k);
        for (uint32_t i = 0; i < k; ++i) v.push_back(str());
        return v;
    }
    // Float arrays may be unaligned inside the payload: copy out.
    std::vector<float> floats(size_t k) {
        std::vector<float> v(k);
        const uint8_t* s = take(k * sizeof(float));
        if (k) std::memcpy(v.data(), s, k * sizeof(float));
        return v;
    }
    // Views into the payload (it outlives the request).
    std::vector<std::string_view> docs(size_t want_fields, size_t& n_docs) {
        n_docs = pod<uint64_t>();
        const uint32_t nf = pod<uint32_t>();
        if (nf != want_fields) throw std::runtime_error("ctxshield: doc has wrong number of fields");
        const size_t total = n_docs * nf;
        if (n_docs != 0 && total / n_docs != nf) throw std::runtime_error("ctxshield-scorer: docs size overflow");
        std::vector<uint64_t> lens(total);
        const uint8_t* l = take(total * sizeof(uint64_t));
        if (total) std::memcpy(lens.data(), l, total * sizeof(uint64_t));
        std::vector<std::string_view> v(total);
        for (size_t i = 0; i < total; ++i) {
            const uint8_t* s = take(static_cast<size_t>(lens[i]));
            v[i] = std::string_view(reinterpret_cast<const char*>(s), static_cast<size_t>(lens[i]));
        }
        return v;
    }
    void end() const {
        if (i_ != n_) throw std::runtime_error("ctxshield-scorer: trailing bytes in request payload");
    }

private:
    const uint8_t* p_;
    size_t n_;
    size_t i_ = 0;
};

class Writer {
public:
    template <typename T> void pod(T v) { raw(&v, sizeof(T)); }
    void str(const std::string& s) {
        pod<uint32_t>(static_cast<uint32_t>(s.size()));
        raw(s.data(), s.size());
    }
    void raw(const void* p, size_t k) {
        const auto* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + k);
    }
    std::vector<uint8_t> buf;
};

void respond(uint32_t status, const std::vector<uint8_t>& payload) {
    uint8_t h[16];
    std::memcpy(h, "CXR1", 4);
    std::memcpy(h + 4, &status, 4);
    const uint64_t n = payload.size();
    std::memcpy(h + 8, &n, 8);
    write_full(h, sizeof(h));
    if (n) write_full(payload.data(), payload.size());
}

struct Entry {
    shield_scorer* h;
    size_t n_fields;
    int hpad;
};
std::map<uint32_t, Entry> g_scorers;
uint32_t g_next_handle = 1;

std::vector<const char*> c_names(const std::vector<std::string>& names) {
    std::vector<const char*> v;
    for (auto& s : names) v.push_back(s.c_str());
    return v;
}

void op_hello(Reader& r, Writer& w) {
    r.end();
    w.pod<uint32_t>(kProtocol);
    w.pod<int32_t>(kDimBits);
    w.pod<int32_t>(shield_abi_version());
    w.pod<int32_t>(default_threads());
    w.str(utf8proc_version_string());
    w.str(unicode_version_string());
}

void op_available(Reader& r, Writer& w) {
    const int32_t be = r.pod<int32_t>();
    r.end();
    char why[512] = {0};
    const int ok = shield_backend_available(be, why, sizeof(why));
    w.pod<uint8_t>(ok == 1 ? 1 : 0);
    w.str(why);
}

void op_featurize(Reader& r, Writer& w) {
    const std::vector<std::string> names = r.names();
    const bool delex = r.pod<uint8_t>() != 0;
    const int32_t threads = r.pod<int32_t>();
    FieldSet fs(names);
    const size_t nf = names.size();
    size_t n = 0;
    std::vector<std::string_view> V = r.docs(nf, n);
    r.end();
    const uint32_t ne = static_cast<uint32_t>(fs.n_extras());
    std::vector<Features> feats(n);
    parallel_chunks(n, threads, [&](size_t a, size_t b) {
        Scratch& sc = thread_scratch();
        for (size_t i = a; i < b; ++i) featurize(fs, V.data() + i * nf, delex, sc, feats[i]);
    });
    uint64_t nnz = 0;
    for (auto& f : feats) nnz += f.idx.size();
    w.pod<uint64_t>(n);
    w.pod<uint64_t>(nnz);
    w.pod<uint32_t>(ne);
    w.buf.reserve(w.buf.size() + (n + 1) * 8 + nnz * 8 + n * ne * 4);
    int64_t pos = 0;
    w.pod<int64_t>(0);
    for (auto& f : feats) {
        pos += static_cast<int64_t>(f.idx.size());
        w.pod<int64_t>(pos);
    }
    for (auto& f : feats)
        for (uint32_t ix : f.idx) w.pod<int32_t>(static_cast<int32_t>(ix));
    for (auto& f : feats) w.raw(f.val.data(), f.val.size() * sizeof(float));
    for (auto& f : feats) {
        if (f.extras.size() != ne) throw std::runtime_error("ctxshield-scorer: extras size mismatch");
        w.raw(f.extras.data(), f.extras.size() * sizeof(float));
    }
}

void op_create(Reader& r, Writer& w) {
    const int32_t be = r.pod<int32_t>();
    const std::vector<std::string> names = r.names();
    const bool delex = r.pod<uint8_t>() != 0;
    const int32_t hpad = r.pod<int32_t>();
    if (hpad <= 0 || hpad > kMaxHeadsPad || (hpad & 3) != 0)
        throw std::runtime_error("ctxshield: hpad must be 4, 8, 12 or 16");
    FieldSet fs(names);   // validates sorted + unique, gives n_extras
    const std::vector<float> W = r.floats(static_cast<size_t>(kDim) * static_cast<size_t>(hpad));
    const std::vector<float> We = r.floats(static_cast<size_t>(fs.n_extras()) * static_cast<size_t>(hpad));
    const std::vector<float> b = r.floats(static_cast<size_t>(hpad));
    r.end();
    std::vector<const char*> nm = c_names(names);
    char err[1024] = {0};
    shield_scorer* h = nullptr;
    if (shield_scorer_create(be, nm.data(), static_cast<int>(nm.size()), delex ? 1 : 0, W.data(), We.data(),
                             b.data(), hpad, &h, err, sizeof(err)) != 0)
        throw std::runtime_error(err);
    const uint32_t id = g_next_handle++;
    g_scorers[id] = Entry{h, names.size(), hpad};
    w.pod<uint32_t>(id);
}

const Entry& lookup(uint32_t id) {
    auto it = g_scorers.find(id);
    if (it == g_scorers.end()) throw std::runtime_error("ctxshield-scorer: unknown scorer handle");
    return it->second;
}

void op_score(Reader& r, Writer& w) {
    const Entry& s = lookup(r.pod<uint32_t>());
    const int32_t threads = r.pod<int32_t>();
    size_t n = 0;
    std::vector<std::string_view> V = r.docs(s.n_fields, n);
    r.end();
    std::vector<const char*> ptrs(V.size());
    std::vector<size_t> lens(V.size());
    for (size_t i = 0; i < V.size(); ++i) { ptrs[i] = V[i].data(); lens[i] = V[i].size(); }
    std::vector<float> logits(n * static_cast<size_t>(s.hpad));
    shield_stats st{};
    char err[1024] = {0};
    if (shield_scorer_score(s.h, ptrs.data(), lens.data(), n, threads, logits.data(), &st, err, sizeof(err)) != 0)
        throw std::runtime_error(err);
    w.raw(logits.data(), logits.size() * sizeof(float));
    w.pod<int32_t>(st.backend);
    w.pod<uint64_t>(st.n_docs);
    w.pod<uint64_t>(st.n_overflow_docs);
    w.pod<uint64_t>(st.n_tokens);
    w.pod<uint64_t>(st.n_arena_bytes);
    w.pod<double>(st.t_tokenize_s);
    w.pod<double>(st.t_device_s);
    w.pod<double>(st.t_total_s);
}

void op_destroy(Reader& r, Writer&) {
    const uint32_t id = r.pod<uint32_t>();
    r.end();
    shield_scorer_destroy(lookup(id).h);
    g_scorers.erase(id);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::printf("ctxshield-scorer protocol %u abi %d utf8proc %s unicode %s\n", kProtocol,
                    shield_abi_version(), utf8proc_version_string(), unicode_version_string());
        return 0;
    }
    if (argc != 1) fatal("usage: ctxshield-scorer [--version]   (speaks the CXS1 protocol on stdin/stdout)");
    std::vector<uint8_t> payload;
    for (;;) {
        uint8_t h[16];
        if (!read_full(h, sizeof(h), true)) return 0;
        if (std::memcmp(h, "CXS1", 4) != 0) fatal("bad request magic");
        uint32_t op;
        uint64_t len;
        std::memcpy(&op, h + 4, 4);
        std::memcpy(&len, h + 8, 8);
        if (len > (uint64_t(1) << 36)) fatal("request too large");
        payload.resize(static_cast<size_t>(len));
        if (len) read_full(payload.data(), payload.size(), false);
        Reader r(payload.data(), payload.size());
        Writer w;
        try {
            switch (op) {
                case 1: op_hello(r, w); break;
                case 2: op_available(r, w); break;
                case 3: op_featurize(r, w); break;
                case 4: op_create(r, w); break;
                case 5: op_score(r, w); break;
                case 6: op_destroy(r, w); break;
                default: throw std::runtime_error("ctxshield-scorer: unknown op " + std::to_string(op));
            }
        } catch (const std::exception& e) {
            const std::string m = e.what();
            respond(1, std::vector<uint8_t>(m.begin(), m.end()));
            continue;
        }
        respond(0, w.buf);
    }
}
