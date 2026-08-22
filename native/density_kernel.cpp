#include <complex>
#include <cstdint>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct DensityContext {
    std::int64_t source_dimension;
    int n_orbitals;
    const std::uint64_t* source_basis;
    std::unordered_map<std::uint64_t, std::int64_t> source_lookup;
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

}  // namespace

extern "C" {

void* density_create_context(
    std::int64_t source_dimension,
    int n_orbitals,
    const std::uint64_t* source_basis
) {
    auto* context = new DensityContext{
        source_dimension, n_orbitals, source_basis, {}
    };
    context->source_lookup.reserve(static_cast<std::size_t>(source_dimension * 1.35));
    for (std::int64_t index = 0; index < source_dimension; ++index) {
        context->source_lookup.emplace(source_basis[index], index);
    }
    return context;
}

void density_destroy_context(void* raw_context) {
    delete static_cast<DensityContext*>(raw_context);
}

void density_apply(
    void* raw_context,
    std::int64_t target_dimension,
    const std::uint64_t* target_basis,
    const std::int64_t* offsets,
    const int* initial_orbitals,
    const std::complex<double>* coefficients,
    const std::complex<double>* input,
    std::complex<double>* output
) {
    const auto& context = *static_cast<DensityContext*>(raw_context);
    #pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < target_dimension; ++row) {
        const std::uint64_t target = target_basis[row];
        std::complex<double> value = 0.0;
        std::uint64_t occupied = target;
        while (occupied != 0ULL) {
            const int final_orbital = __builtin_ctzll(occupied);
            occupied &= occupied - 1ULL;
            for (std::int64_t location = offsets[final_orbital];
                 location < offsets[final_orbital + 1]; ++location) {
                std::uint64_t source = target;
                int sign = 1;
                if (!annihilate(source, final_orbital, sign)) continue;
                if (!create(source, initial_orbitals[location], sign)) continue;
                const auto found = context.source_lookup.find(source);
                if (found == context.source_lookup.end()) continue;
                value += static_cast<double>(sign) * coefficients[location]
                    * input[found->second];
            }
        }
        output[row] = value;
    }
}

}
