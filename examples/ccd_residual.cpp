// The closed-shell CCD doubles residual, in the form a production code uses:
// chemist-notation integral blocks, spin-adapted, intermediates built once and
// reused. Follows PySCF's restricted CCSD with the singles set to zero.
//
// i, j, k, l index occupied orbitals and a, b, c, d virtual ones. Tensors are
// random, so this shows how the equations are written, not a working solver.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "util.hpp"

using example::ein;
using Tensor = example::RandomTensor<double>;

int main(int argc, char** argv) try {
  const katal::Runtime runtime;
  // Orbital counts, overridable so a different size needs no rebuild.
  const std::int64_t o = argc > 1 ? std::atoll(argv[1]) : 8;
  const std::int64_t v = argc > 2 ? std::atoll(argv[2]) : 48;

  unsigned seed = 0;
  const auto next = [&seed] { return ++seed; };

  const Tensor t2({o, o, v, v}, next());

  // Fock blocks, and integral blocks in chemist notation.
  const Tensor foo({o, o}, next());
  const Tensor fvv({v, v}, next());
  const Tensor ovov({o, v, o, v}, next());  // (ia|jb)
  const Tensor oooo({o, o, o, o}, next());  // (ki|lj)
  const Tensor oovv({o, o, v, v}, next());  // (ki|ac)
  const Tensor ovvo({o, v, v, o}, next());  // (kc|ai)
  const Tensor vvvv({v, v, v, v}, next());  // (ac|bd)

  const Tensor loo({o, o}, next());
  const Tensor lvv({v, v}, next());
  const Tensor woooo({o, o, o, o}, next());
  const Tensor wvvvv({v, v, v, v}, next());
  const Tensor wvoov({v, o, o, v}, next());
  const Tensor wvovo({v, o, v, o}, next());
  const Tensor t2new({o, o, v, v}, next());

  const auto evaluate = [&] {
    // ---- two-index intermediates ---------------------------------------
    // With no singles, Loo and Lvv reduce to the Foo and Fvv of Eqs. (37)-(38).
    ein("ki->ki", 1.0, foo, 0.0, loo);
    ein("kcld,ilcd->ki", 2.0, ovov, t2, 1.0, loo);
    ein("kdlc,ilcd->ki", -1.0, ovov, t2, 1.0, loo);

    ein("ac->ac", 1.0, fvv, 0.0, lvv);
    ein("kcld,klad->ac", -2.0, ovov, t2, 1.0, lvv);
    ein("kdlc,klad->ac", 1.0, ovov, t2, 1.0, lvv);

    // ---- four-index intermediates --------------------------------------
    // Each stored block enters through a one-operand einsum that permutes it
    // into the intermediate's index order.
    ein("kilj->klij", 1.0, oooo, 0.0, woooo);
    ein("kcld,ijcd->klij", 1.0, ovov, t2, 1.0, woooo);

    // Every singles term of Wvvvv vanishes, leaving a permutation of (ac|bd).
    ein("acbd->abcd", 1.0, vvvv, 0.0, wvvvv);

    ein("kcai->akic", 1.0, ovvo, 0.0, wvoov);
    ein("ldkc,ilda->akic", -0.5, ovov, t2, 1.0, wvoov);
    ein("lckd,ilad->akic", -0.5, ovov, t2, 1.0, wvoov);
    ein("ldkc,ilad->akic", 1.0, ovov, t2, 1.0, wvoov);

    ein("kiac->akci", 1.0, oovv, 0.0, wvovo);
    ein("lckd,ilda->akci", -0.5, ovov, t2, 1.0, wvovo);

    // ---- doubles residual ----------------------------------------------
    // The driver term, then the two ladders. tau is t2 when there are no
    // singles.
    ein("iajb->ijab", 1.0, ovov, 0.0, t2new);
    ein("klij,klab->ijab", 1.0, woooo, t2, 1.0, t2new);
    ein("abcd,ijcd->ijab", 1.0, wvvvv, t2, 1.0, t2new);

    // Each of the remaining terms is symmetric under exchanging both index
    // pairs. numpy forms the term, then adds its transpose; here the second
    // contraction writes straight into the same output with its result labels
    // swapped.
    ein("ac,ijcb->ijab", 1.0, lvv, t2, 1.0, t2new);
    ein("ac,ijcb->jiba", 1.0, lvv, t2, 1.0, t2new);

    ein("ki,kjab->ijab", -1.0, loo, t2, 1.0, t2new);
    ein("ki,kjab->jiba", -1.0, loo, t2, 1.0, t2new);

    ein("akic,kjcb->ijab", 2.0, wvoov, t2, 1.0, t2new);
    ein("akci,kjcb->ijab", -1.0, wvovo, t2, 1.0, t2new);
    ein("akic,kjcb->jiba", 2.0, wvoov, t2, 1.0, t2new);
    ein("akci,kjcb->jiba", -1.0, wvovo, t2, 1.0, t2new);

    ein("akic,kjbc->ijab", -1.0, wvoov, t2, 1.0, t2new);
    ein("akic,kjbc->jiba", -1.0, wvoov, t2, 1.0, t2new);

    ein("bkci,kjac->ijab", -1.0, wvovo, t2, 1.0, t2new);
    ein("bkci,kjac->jiba", -1.0, wvovo, t2, 1.0, t2new);
  };

  // The first pass pays for cuTENSOR's one-time initialisation and for
  // planning each distinct contraction. A solver iterates, so time a later one.
  evaluate();
  example::check(cudaDeviceSynchronize());
  example::flops = 0.0;

  example::Timer timer;
  evaluate();
  example::check(cudaDeviceSynchronize());
  const double seconds = timer.seconds();

  std::printf("CCD residual, o=%lld v=%lld\n", (long long)o, (long long)v);
  example::report("  one evaluation", seconds);
  std::printf("  a real solver would divide by the denominators and iterate\n");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
