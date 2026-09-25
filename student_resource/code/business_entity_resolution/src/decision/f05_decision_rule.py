"""
f05_decision_rule.py - per-entity set selection for macro-F0.5 (Amazon ML Challenge 2026, ER task)

  f05(pred, truth)                   exact competition metric for ONE Source-1 entity
  plan6_pseudocode(ids, probs)       plan 6's function, reproduced verbatim for comparison ONLY
  gfm(ids, P, p_empty, beta)         GFM (Dembczynski et al., NIPS 2011) generalised to F-beta
  P_independent(probs)               GFM inputs if candidate match events are independent
  P_count_model(probs, count_dist)   count-tilted GFM inputs: pair model picks WHICH candidates,
                                     an entity-level count model picks HOW MANY (recommended)
  expand_count_classes(pc, probs)    4-class count output [P0,P1,P2,P3+] -> Pcount(s)

Cap candidate lists at m <= 30 (sorted by p) before calling GFM.
Inputs must be CALIBRATED probabilities (isotonic/Platt fitted on out-of-fold predictions).
Fit the count model on the number of true matches INSIDE the pruned candidate list, with
out-of-fold pair probabilities as its features; keep country out of its features (France is unseen).

Provenance: functions and self-tests 1-6 are Appendix A of the build spec, unchanged. The spec
message was truncated inside self-test 7; test 7 and the m = 40 NaN check below are rebuilt to the
spec's stated checks ("count-tilted GFM vs brute force: mismatches = 0 / 200", "no NaN at m = 40").
"""
import numpy as np


def f05(pred, truth, beta=0.5):
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0                       # correct empty prediction for a singleton
    tp = len(pred & truth)
    if tp == 0:
        return 0.0                       # covers: any prediction on a singleton, empty on a non-singleton
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * len(truth) + len(pred))


def plan6_pseudocode(candidate_ids, probabilities):
    """Plan 6, appendix D - DO NOT USE. Kept to demonstrate its two bugs."""
    if not candidate_ids:
        return []
    order = np.argsort(probabilities)[::-1]
    c = [candidate_ids[i] for i in order]
    p = [probabilities[i] for i in order]
    best, idx, m = 0.0, 0, len(p)
    for k in range(1, m + 1):
        t = sum(p[:k]); f = k - t
        val = (1.25 * t) / (0.25 * m + t + f)       # bug 2: 0.25*m (candidate count) instead of 0.25*E|T|
        if val > best:                              # bug 1: empty set is never scored
            best, idx = val, k
    return c[:idx]


def gfm(candidate_ids, P, p_empty, beta=0.5):
    """
    P[i, s-1] = Pr(candidate i is a true match AND the entity has s true matches in total), s = 1..S
    p_empty   = Pr(entity has no true match)
    Returns (best_set, expected_F). Exact for any joint distribution summarised by (P, p_empty).
    """
    P = np.asarray(P, float)
    m, S = P.shape
    b2 = beta * beta
    s = np.arange(1, S + 1)
    best_val, best_set = float(p_empty), []
    for k in range(1, m + 1):
        delta = P @ ((1 + b2) / (b2 * s + k))       # expected contribution of each candidate if |pred| = k
        top = np.argsort(-delta)[:k]
        val = float(delta[top].sum())
        if val > best_val + 1e-12:
            best_val, best_set = val, [candidate_ids[j] for j in top]
    return best_set, best_val


def _poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1); dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist


def P_independent(probs):
    p = np.asarray(probs, float); m = len(p)
    P = np.zeros((m, m))
    for i in range(m):
        P[i, :] = p[i] * _poisson_binomial(np.delete(p, i))   # column s-1  <=>  s-1 other matches
    return P, float(np.prod(1 - p))


def P_count_model(probs, count_dist):
    """
    Count-tilted Poisson-binomial GFM inputs (replaces the earlier proportional-allocation version).
    The pair model decides WHICH candidates, the count model decides HOW MANY:
        P(y_i = 1, S = s) = Pcount(S = s) * p_i * PB_{-i}(s - 1) / PB(s)
    This is the joint "independent Bernoulli(p) conditioned on the count, count drawn from Pcount",
    so it is always a valid distribution; it reduces exactly to P_independent when Pcount = PB.
    count_dist[s] = Pr(entity has s true matches INSIDE the candidate list), s = 0..len-1.
    """
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)   # 1e-6 floor: no underflow for m <= 50
    m = len(p)
    cd = np.zeros(m + 1)
    c = np.asarray(count_dist, float)
    cd[:min(len(c), m + 1)] = c[:m + 1]
    if len(c) > m + 1:                      # mass on impossible counts -> largest feasible count
        cd[m] += c[m + 1:].sum()
    cd = cd / cd.sum()
    pb = _poisson_binomial(p)
    P = np.zeros((m, m))
    for i in range(m):
        loo = _poisson_binomial(np.delete(p, i))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pb[1:] > 0, loo[:m] / pb[1:], 0.0)
        P[i, :] = cd[1:] * p[i] * ratio
    return P, float(cd[0])


def expand_count_classes(p_classes, probs):
    """Turn a 4-class count prediction [P0, P1, P2, P3plus] into Pcount(s), s = 0..m,
    spreading the 3+ mass over s >= 3 in proportion to the Poisson-binomial PB(s)."""
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6); m = len(p)
    pc = np.asarray(p_classes, float)
    out = np.zeros(m + 1)
    for s in range(min(3, m + 1)):
        out[s] = pc[s]
    if m >= 3:
        pb = _poisson_binomial(p)[3:]
        out[3:] = pc[3] * (pb / pb.sum() if pb.sum() > 0 else np.eye(1, m - 2, 0).ravel())
    else:                                   # 3+ impossible with < 3 candidates
        out[m] += pc[3:].sum()
    return out / out.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) metric sanity check against the official example (2 TP + 1 FP vs 2 truths -> 0.714)
    assert abs(f05(["a", "b", "c"], ["a", "c"]) - 1.25 * 2 / 3.5) < 1e-12
    print("official example:", round(f05(["a", "b", "c"], ["a", "c"]), 4))

    # 2) GFM == brute force over all 2^m subsets, for random joint distributions (no independence)
    from itertools import product
    m = 5
    for trial in range(300):
        ys = np.array(list(product([0, 1], repeat=m)))
        py = rng.dirichlet(np.ones(len(ys)) * 0.3)
        ids = list(range(m))
        P = np.zeros((m, m))
        for y, pr in zip(ys, py):
            sy = y.sum()
            for i in range(m):
                if y[i]:
                    P[i, sy - 1] += pr
        p0 = py[ys.sum(1) == 0].sum()
        _, v_gfm = gfm(ids, P, p0)
        v_bf = max(sum(pr * f05({i for i in range(m) if h[i]}, {i for i in range(m) if y[i]})
                       for y, pr in zip(ys, py)) for h in ys)
        assert abs(v_gfm - v_bf) < 1e-9, (v_gfm, v_bf)
    print("GFM matches brute force on 300 random joint distributions")

    def exp_f_indep(pred_idx, probs):
        """exact expected F0.5 of a prediction set under independence (enumeration)"""
        from itertools import product
        tot = 0.0
        for y in product([0, 1], repeat=len(probs)):
            pr = np.prod([p if b else 1 - p for p, b in zip(probs, y)])
            tot += pr * f05(set(pred_idx), {i for i, b in enumerate(y) if b})
        return tot

    # 3) plan 6 bug 1: likely singleton (all candidates weak)
    probs = [0.05, 0.03, 0.02]; ids = list(range(3))
    p6 = plan6_pseudocode(ids, probs)
    g, v = gfm(ids, *P_independent(probs))
    print(f"\nlikely singleton {probs}")
    print(f"  plan 6 predicts {p6}  -> expected F {exp_f_indep(p6, probs):.3f}")
    print(f"  GFM predicts    {g}   -> expected F {v:.3f}")

    # 4) plan 6 bug 2: one strong match + long tail of 19 weak candidates (m = 20)
    probs = [0.95, 0.30, 0.20, 0.15] + [0.05] * 4        # enumeration-friendly m = 8
    ids = list(range(len(probs)))
    p6 = plan6_pseudocode(ids, probs)
    g, v = gfm(ids, *P_independent(probs))
    print(f"\nstrong match + tail {probs}")
    print(f"  plan 6 predicts {p6} -> expected F {exp_f_indep(p6, probs):.3f}")
    print(f"  GFM predicts    {g}       -> expected F {v:.3f}")
    probs20 = [0.95, 0.30, 0.20, 0.15] + [0.05] * 16
    print(f"  same list padded to m=20: plan 6 predicts {len(plan6_pseudocode(list(range(20)), probs20))} candidates")

    # 5) the two regimes, exclusivity model (entity has at most 1 match)
    print("\nfirst link, entity has 0 or 1 match, one candidate with prob p:")
    for p in [0.45, 0.50, 0.55]:
        g, v = gfm(["c"], [[p]], 1 - p)
        print(f"  p={p:.2f} -> {g}")
    print("second link, first match certain, second candidate is a match with prob p:")
    for p in [0.70, 0.72, 0.73, 0.75]:
        P = np.array([[1 - p, p], [0.0, p]])       # rows: c1, c2 ; cols: total count s=1, s=2
        g, v = gfm(["c1", "c2"], P, 0.0)
        print(f"  p={p:.2f} -> {g}")

    # 6) independence vs a count model can disagree on the empty set
    probs = [0.35, 0.35]
    gi, vi = gfm(["c1", "c2"], *P_independent(probs))
    gc, vc = gfm(["c1", "c2"], *P_count_model(probs, [0.30, 0.70]))   # 0 or 1 match, mass split
    print(f"\ncandidates {probs}: independence -> {gi} ; count model (P(T=0)=0.30, T<=1) -> {gc}")

    # 7) [rebuilt] count-tilted construction is exact on the joint it implies (random cases):
    #    y ~ independent Bernoulli(p) conditioned on |y| = s, s ~ Pcount. Brute force over all
    #    2^m label vectors and all 2^m prediction sets must give the same best expected F as GFM.
    mism = 0
    for t in range(200):
        m = int(rng.integers(2, 6))
        probs = rng.uniform(0.02, 0.98, m)
        cdist = rng.dirichlet(np.ones(m + 1))
        P, p0 = P_count_model(probs, cdist)
        _, v_gfm = gfm(list(range(m)), P, p0)
        pcl = np.clip(probs, 1e-6, 1 - 1e-6)
        pb = _poisson_binomial(pcl)
        ys = np.array(list(product([0, 1], repeat=m)))
        py = np.array([cdist[y.sum()] * np.prod(np.where(y, pcl, 1 - pcl)) / pb[y.sum()] for y in ys])
        v_bf = max(sum(pr * f05({i for i in range(m) if h[i]}, {i for i in range(m) if y[i]})
                       for y, pr in zip(ys, py)) for h in ys)
        mism += abs(v_gfm - v_bf) > 1e-9
    print(f"\ncount-tilted GFM vs brute force: mismatches = {mism} / 200")
    assert mism == 0

    # 8) [rebuilt] numerical safety for long lists
    probs40 = rng.uniform(0.0, 1.0, 40)
    P, p0 = P_count_model(probs40, expand_count_classes([0.1, 0.3, 0.3, 0.3], probs40))
    g, v = gfm(list(range(40)), P, p0)
    assert np.isfinite(P).all() and np.isfinite(v)
    print(f"m = 40: no NaN/Inf (expected F {v:.4f}, |pred| = {len(g)})")
