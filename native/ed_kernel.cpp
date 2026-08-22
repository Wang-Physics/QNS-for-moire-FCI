#include <algorithm>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Context {
    std::int64_t dimension;
    int n_pairs;
    int n_orbitals;
    const std::uint64_t* basis;
    const double* diagonal;
    const int* pair_first;
    const int* pair_second;
    const std::int64_t* offsets;
    const int* destinations;
    const std::complex<double>* interaction;
    std::unordered_map<std::uint64_t, std::int64_t> lookup;
    std::vector<int> pair_lookup;
};

inline int parity_below(std::uint64_t state, int orbital) {
    const std::uint64_t mask = orbital == 0 ? 0ULL : ((1ULL << orbital) - 1ULL);
    return __builtin_popcountll(state & mask) & 1;
}

inline bool annihilate(std::uint64_t& state, int orbital, int& sign) {
    const std::uint64_t bit = 1ULL << orbital;
    if ((state & bit) == 0ULL) return false;
    if (parity_below(state, orbital)) sign = -sign;
    state ^= bit;
    return true;
}

inline bool create(std::uint64_t& state, int orbital, int& sign) {
    const std::uint64_t bit = 1ULL << orbital;
    if ((state & bit) != 0ULL) return false;
    if (parity_below(state, orbital)) sign = -sign;
    state |= bit;
    return true;
}

inline bool apply_pair(
    std::uint64_t state,
    int annihilate_first,
    int annihilate_second,
    int create_first,
    int create_second,
    std::uint64_t& output,
    int& sign
) {
    sign = 1;
    // This order exactly matches src.multiband_ed._apply_pair.
    if (!annihilate(state, annihilate_first, sign)) return false;
    if (!annihilate(state, annihilate_second, sign)) return false;
    if (!create(state, create_second, sign)) return false;
    if (!create(state, create_first, sign)) return false;
    output = state;
    return true;
}

}  // namespace

extern "C" {

void* ed_create_context(
    std::int64_t dimension,
    int n_pairs,
    int n_orbitals,
    const std::uint64_t* basis,
    const double* diagonal,
    const int* pair_first,
    const int* pair_second,
    const std::int64_t* offsets,
    const int* destinations,
    const std::complex<double>* interaction
) {
    auto* context = new Context{
        dimension,
        n_pairs,
        n_orbitals,
        basis,
        diagonal,
        pair_first,
        pair_second,
        offsets,
        destinations,
        interaction,
        {},
        std::vector<int>(static_cast<std::size_t>(n_orbitals * n_orbitals), -1),
    };
    context->lookup.reserve(static_cast<std::size_t>(dimension * 1.35));
    for (std::int64_t index = 0; index < dimension; ++index) {
        context->lookup.emplace(basis[index], index);
    }
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int a = pair_first[pair];
        const int b = pair_second[pair];
        context->pair_lookup[a * n_orbitals + b] = pair;
        context->pair_lookup[b * n_orbitals + a] = pair;
    }
    return context;
}

void ed_destroy_context(void* raw_context) {
    delete static_cast<Context*>(raw_context);
}

void ed_matvec(
    void* raw_context,
    const std::complex<double>* input,
    std::complex<double>* output
) {
    const auto& context = *static_cast<Context*>(raw_context);

    // Row-oriented evaluation avoids atomic writes.  Hermiticity gives
    // H[row,col] = sign * V[q,p] when pair q is removed from row and pair p
    // is inserted to obtain col.
    #pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < context.dimension; ++row) {
        const std::uint64_t state = context.basis[row];
        std::complex<double> value = context.diagonal[row] * input[row];
        int occupied[64];
        int n_occupied = 0;
        std::uint64_t remaining = state;
        while (remaining != 0ULL) {
            occupied[n_occupied++] = __builtin_ctzll(remaining);
            remaining &= remaining - 1ULL;
        }
        for (int first_index = 0; first_index < n_occupied; ++first_index) {
          for (int second_index = first_index + 1; second_index < n_occupied; ++second_index) {
            const int c = occupied[first_index];
            const int d = occupied[second_index];
            const int q = context.pair_lookup[c * context.n_orbitals + d];
            for (std::int64_t location = context.offsets[q]; location < context.offsets[q + 1]; ++location) {
                const int p = context.destinations[location];
                std::uint64_t column_state = 0ULL;
                int sign = 1;
                if (!apply_pair(state, c, d, context.pair_first[p], context.pair_second[p], column_state, sign)) continue;
                const auto found = context.lookup.find(column_state);
                if (found == context.lookup.end()) continue;
                value += static_cast<double>(sign)
                    * context.interaction[static_cast<std::int64_t>(q) * context.n_pairs + p]
                    * input[found->second];
            }
          }
        }
        output[row] = value;
    }
}

}
