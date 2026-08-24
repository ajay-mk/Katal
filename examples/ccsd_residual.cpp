// The complete spin-orbital CCSD residual: tau, the two- and four-index
// intermediates, then the singles and doubles equations.
//
// i, j, k, l, m, n index occupied spin orbitals and a, b, c, d, e, f virtual
// ones; <pq||rs> is the antisymmetrised integral.
//
// The equations are transcribed, not checked against a reference. Real
// arithmetic only: complex amplitudes need a conjugation cuEin cannot express.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "util.hpp"

using example::ein;
using Tensor = example::RandomTensor<double>;

int main(int argc, char** argv) {
  // Orbital counts, overridable so a different size needs no rebuild.
  const std::int64_t o = argc > 1 ? std::atoll(argv[1]) : 8;   // occupied
  const std::int64_t v = argc > 2 ? std::atoll(argv[2]) : 48;  // virtual

  unsigned seed = 0;
  const auto next = [&seed] { return ++seed; };

  // Amplitudes.
  const Tensor t1({o, v}, next());
  const Tensor t2({o, o, v, v}, next());

  // Fock blocks.
  const Tensor f_oo({o, o}, next());
  const Tensor f_ov({o, v}, next());
  const Tensor f_vv({v, v}, next());

  // Antisymmetrised integral blocks.
  const Tensor oooo({o, o, o, o}, next());
  const Tensor ooov({o, o, o, v}, next());
  const Tensor oovo({o, o, v, o}, next());
  const Tensor oovv({o, o, v, v}, next());
  const Tensor ovoo({o, v, o, o}, next());
  const Tensor ovov({o, v, o, v}, next());
  const Tensor ovvo({o, v, v, o}, next());
  const Tensor ovvv({o, v, v, v}, next());
  const Tensor vvvo({v, v, v, o}, next());
  const Tensor vvvv({v, v, v, v}, next());

  // Intermediates.
  const Tensor tau({o, o, v, v}, next());
  const Tensor tau_t({o, o, v, v}, next());
  const Tensor fae({v, v}, next());
  const Tensor fmi({o, o}, next());
  const Tensor fme({o, v}, next());
  const Tensor wmnij({o, o, o, o}, next());
  const Tensor wabef({v, v, v, v}, next());
  const Tensor wmbej({o, v, v, o}, next());

  // Residuals.
  const Tensor r1({o, v}, next());
  const Tensor r2({o, o, v, v}, next());

  const auto evaluate = [&] {
    // ---- tau and tau-tilde ---------------------------------------------
    // cuEin has no elementwise add, so each term accumulates with beta = 1.
    // The one-operand form of einsum copies t2 in; the outer products follow.
    ein("ijab->ijab", 1.0, t2, 0.0, tau);
    ein("ia,jb->ijab", 1.0, t1, t1, 1.0, tau);
    ein("ib,ja->ijab", -1.0, t1, t1, 1.0, tau);

    ein("ijab->ijab", 1.0, t2, 0.0, tau_t);
    ein("ia,jb->ijab", 0.5, t1, t1, 1.0, tau_t);
    ein("ib,ja->ijab", -0.5, t1, t1, 1.0, tau_t);

    // ---- two-index intermediates ---------------------------------------
    // Fae = f_ae - 1/2 f_me t_m^a + t_m^f <ma||fe> - 1/2 tau~_mn^af <mn||ef>
    ein("ae->ae", 1.0, f_vv, 0.0, fae);
    ein("me,ma->ae", -0.5, f_ov, t1, 1.0, fae);
    ein("mf,mafe->ae", 1.0, t1, ovvv, 1.0, fae);
    ein("mnaf,mnef->ae", -0.5, tau_t, oovv, 1.0, fae);

    // Fmi = f_mi + 1/2 f_me t_i^e + t_n^e <mn||ie> + 1/2 tau~_in^ef <mn||ef>
    ein("mi->mi", 1.0, f_oo, 0.0, fmi);
    ein("me,ie->mi", 0.5, f_ov, t1, 1.0, fmi);
    ein("ne,mnie->mi", 1.0, t1, ooov, 1.0, fmi);
    ein("inef,mnef->mi", 0.5, tau_t, oovv, 1.0, fmi);

    // Fme = f_me + t_n^f <mn||ef>
    ein("me->me", 1.0, f_ov, 0.0, fme);
    ein("nf,mnef->me", 1.0, t1, oovv, 1.0, fme);

    // ---- four-index intermediates --------------------------------------
    // Wmnij = <mn||ij> + P(ij) t_j^e <mn||ie> + 1/4 tau_ij^ef <mn||ef>
    ein("mnij->mnij", 1.0, oooo, 0.0, wmnij);
    ein("je,mnie->mnij", 1.0, t1, ooov, 1.0, wmnij);
    ein("ie,mnje->mnji", -1.0, t1, ooov, 1.0, wmnij);
    ein("ijef,mnef->mnij", 0.25, tau, oovv, 1.0, wmnij);

    // Wabef = <ab||ef> - P(ab) t_m^b <am||ef> + 1/4 tau_mn^ab <mn||ef>
    ein("abef->abef", 1.0, vvvv, 0.0, wabef);
    ein("mb,maef->abef", -1.0, t1, ovvv, 1.0, wabef);
    ein("ma,mbef->baef", 1.0, t1, ovvv, 1.0, wabef);
    ein("mnab,mnef->abef", 0.25, tau, oovv, 1.0, wabef);

    // Wmbej = <mb||ej> + t_j^f <mb||ef> - t_n^b <mn||ej>
    //         - (1/2 t_jn^fb + t_j^f t_n^b) <mn||ef>
    ein("mbej->mbej", 1.0, ovvo, 0.0, wmbej);
    ein("jf,mbef->mbej", 1.0, t1, ovvv, 1.0, wmbej);
    ein("nb,mnej->mbej", -1.0, t1, oovo, 1.0, wmbej);
    ein("jnfb,mnef->mbej", -0.5, t2, oovv, 1.0, wmbej);

    // ---- singles residual ----------------------------------------------
    // r1 = f_ia + t_i^e Fae - t_m^a Fmi + t_im^ae Fme
    //      - t_n^f <na||if> - 1/2 t_im^ef <ma||ef> - 1/2 t_mn^ae <nm||ei>
    ein("ia->ia", 1.0, f_ov, 0.0, r1);
    ein("ie,ae->ia", 1.0, t1, fae, 1.0, r1);
    ein("ma,mi->ia", -1.0, t1, fmi, 1.0, r1);
    ein("imae,me->ia", 1.0, t2, fme, 1.0, r1);
    ein("nf,naif->ia", -1.0, t1, ovov, 1.0, r1);
    ein("imef,maef->ia", -0.5, t2, ovvv, 1.0, r1);
    ein("mnae,nmei->ia", -0.5, t2, oovo, 1.0, r1);

    // ---- doubles residual ----------------------------------------------
    // Each permutation operator becomes one more call with the output labels
    // swapped and the sign flipped.
    ein("ijab->ijab", 1.0, oovv, 0.0, r2);

    // P(ab) t_ij^ae Fbe
    ein("ijae,be->ijab", 1.0, t2, fae, 1.0, r2);
    ein("ijbe,ae->ijba", -1.0, t2, fae, 1.0, r2);

    // -P(ij) t_im^ab Fmj
    ein("imab,mj->ijab", -1.0, t2, fmi, 1.0, r2);
    ein("jmab,mi->jiab", 1.0, t2, fmi, 1.0, r2);

    // 1/2 tau_mn^ab Wmnij  and  1/2 tau_ij^ef Wabef
    ein("mnab,mnij->ijab", 0.5, tau, wmnij, 1.0, r2);
    ein("ijef,abef->ijab", 0.5, tau, wabef, 1.0, r2);

    // P(ij)P(ab) t_im^ae Wmbej
    ein("imae,mbej->ijab", 1.0, t2, wmbej, 1.0, r2);
    ein("jmae,mbei->jiab", -1.0, t2, wmbej, 1.0, r2);
    ein("imbe,maej->ijba", -1.0, t2, wmbej, 1.0, r2);
    ein("jmbe,maei->jiba", 1.0, t2, wmbej, 1.0, r2);

    // P(ij) t_i^e <ab||ej>
    ein("ie,abej->ijab", 1.0, t1, vvvo, 1.0, r2);
    ein("je,abei->jiab", -1.0, t1, vvvo, 1.0, r2);

    // -P(ab) t_m^a <mb||ij>
    ein("ma,mbij->ijab", -1.0, t1, ovoo, 1.0, r2);
    ein("mb,maij->ijba", 1.0, t1, ovoo, 1.0, r2);
  };

  // The first pass pays for cuTENSOR's one-time initialisation and for
  // planning each distinct contraction. A solver iterates, so time a later one.
  evaluate();
  cudaDeviceSynchronize();
  example::flops = 0.0;

  example::Timer timer;
  evaluate();
  cudaDeviceSynchronize();
  const double seconds = timer.seconds();
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "a contraction failed\n");
    return 1;
  }

  std::printf("CCSD residual, o=%lld v=%lld\n", (long long)o, (long long)v);
  example::report("  one evaluation", seconds);
  std::printf("  equations transcribed, not verified against a reference\n");
  return 0;
}
