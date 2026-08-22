#include <cstdint>

extern "C" void ed_generate_all_basis(
    int n_orbitals,
    int n_particles,
    int determinant,
    const int* orbital_momentum_x,
    const int* orbital_momentum_y,
    const int* residue_lookup,
    std::uint64_t* states,
    int* sectors
) {
    if (n_particles <= 0 || n_particles > n_orbitals || n_orbitals >= 64) return;
    std::uint64_t state = (1ULL << n_particles) - 1ULL;
    const std::uint64_t limit = 1ULL << n_orbitals;
    std::int64_t output = 0;
    while (state < limit) {
        int mx = 0;
        int my = 0;
        std::uint64_t remaining = state;
        while (remaining != 0ULL) {
            const int orbital = __builtin_ctzll(remaining);
            mx += orbital_momentum_x[orbital];
            my += orbital_momentum_y[orbital];
            remaining &= remaining - 1ULL;
        }
        mx %= determinant;
        my %= determinant;
        states[output] = state;
        sectors[output] = residue_lookup[mx * determinant + my];
        ++output;
        const std::uint64_t low = state & (~state + 1ULL);
        const std::uint64_t ripple = state + low;
        if (ripple >= limit || ripple == 0ULL) break;
        state = ripple | (((state ^ ripple) >> 2) / low);
    }
}
