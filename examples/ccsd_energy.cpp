// The closed-shell CCSD correlation energy, as a first look at Katal.
//
//   E = 2 sum_ijab <ij|ab> t[i,j,a,b] - sum_ijab <ij|ba> t[i,j,a,b]
//
// where i, j index occupied orbitals and a, b virtual ones. Both terms are
// contractions over every index, so the result is a single number.
//
// It runs twice, once in double precision and once in complex double, to show
// that the calls do not change with the element type. The amplitudes are
// random, so neither energy means anything.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "util.hpp"

namespace {

using example::ein;

void print(const char* label, double value) {
  std::printf("  %-9s %+.9f\n", label, value);
}

void print(const char* label, std::complex<double> value) {
  std::printf("  %-9s %+.9f %+.9fi\n", label, value.real(), value.imag());
}

/// Identical for every element type: only the tensors differ.
template <class T>
void correlation_energy(const char* label) {
  constexpr std::int64_t o = 10;  // occupied orbitals
  constexpr std::int64_t v = 40;  // virtual orbitals

  const example::RandomTensor<T> t2({o, o, v, v}, 1);    // doubles amplitudes
  const example::RandomTensor<T> oovv({o, o, v, v}, 2);  // <ij|ab> integrals
  const example::RandomTensor<T> energy({}, 3);  // rank zero: one number

  // Contracting over every index leaves nothing, so the result has rank zero.
  ein("ijab,ijab->", T{1}, t2, oovv, T{0}, energy);
  cudaDeviceSynchronize();
  const T direct = energy.scalar();

  // The exchange term swaps the two virtual labels of the integrals.
  ein("ijab,ijba->", T{1}, t2, oovv, T{0}, energy);
  cudaDeviceSynchronize();
  const T exchange = energy.scalar();

  std::printf("%s\n", label);
  print("direct", direct);
  print("exchange", exchange);
  print("E(corr)", T{2} * direct - exchange);
}

}  // namespace

int main() {
  correlation_energy<double>("real (f64)");
  correlation_energy<std::complex<double>>("complex (c64)");
  std::printf("\nrandom amplitudes, so neither energy is physical\n");
  return 0;
}
