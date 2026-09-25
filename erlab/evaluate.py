"""Metric (macro F0.5 per S1, singletons included), decision rules, rule search, expected-F0.5 set selection."""
import numpy as np
import pandas as pd

BETA2 = 0.25


def s1_metrics(sel_qi, sel_y, n_true, q_eval):
    nq = len(n_true)
    n_pred = np.bincount(sel_qi, minlength=nq)[q_eval].astype(np.float64)
    tp = np.bincount(sel_qi, weights=np.asarray(sel_y, np.float64), minlength=nq)[q_eval]
    nt = n_true[q_eval].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 1.0)
        r = np.where(nt > 0, tp / np.maximum(nt, 1), 1.0)
        f = np.where((n_pred == 0) & (nt == 0), 1.0, np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0))
    sing = nt == 0
    return dict(F05=float(f.mean()), P_micro=float(tp.sum() / max(n_pred.sum(), 1)), R_micro=float(tp.sum() / max(nt.sum(), 1)),
                P_macro=float(p.mean()), R_macro=float(r.mean()),
                singleton_acc=float((n_pred[sing] == 0).mean()) if sing.any() else float('nan'),
                false_merge_rate=float(((n_pred - tp) > 0).mean()), avg_pred=float(n_pred.mean()), n_S1=int(len(q_eval)))


def best_for_pool(pi, p):
    order = np.lexsort((-p, pi))
    ps = pi[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    flag = np.zeros(len(p), bool)
    flag[order[first]] = True
    return flag


def decide(qi, p, s3, bp, rk, prm, nq):
    """per-source thresholds; exclusivity (pool record → its best S1 only); top-n; anchor (best pair of the S1 must reach
    t_anchor); relative margin to the best pair of the S1."""
    ok = p >= np.where(s3, prm['t_s3'], prm['t_s2'])
    if prm.get('exclusive'):
        ok &= bp
    if prm.get('top_n'):
        ok &= rk <= prm['top_n']
    if prm.get('t_anchor', 0) > 0 or prm.get('rel', 0) > 0:
        mx = np.zeros(nq, np.float32)
        np.maximum.at(mx, qi[ok], p[ok])
        ok &= mx[qi] >= prm.get('t_anchor', 0)
        if prm.get('rel', 0) > 0:
            ok &= p >= prm['rel'] * mx[qi]
    return ok


class RuleEval:
    """Repeated rule evaluation on one set of S1 entities (q_eval) with candidate rows (qi, pi, p, y, s3)."""
    def __init__(self, qi, pi, p, y, s3, q_eval, n_true, min_p=0.005):
        p = np.asarray(p, np.float32)
        bp = best_for_pool(pi, p)
        rk = pd.Series(p).groupby(qi).rank(ascending=False, method='first').to_numpy()
        k = p >= min_p
        self.qi, self.pi, self.p, self.y, self.s3, self.bp, self.rk = qi[k], pi[k], p[k], y[k], s3[k], bp[k], rk[k]
        self.keep = k
        self.q_eval, self.n_true, self.nq = q_eval, n_true, len(n_true)

    def select(self, prm):
        return decide(self.qi, self.p, self.s3, self.bp, self.rk, prm, self.nq)

    def metrics(self, prm):
        ok = self.select(prm)
        return s1_metrics(self.qi[ok], self.y[ok], self.n_true, self.q_eval)


T_GRID = np.round(np.arange(0.02, 0.98, 0.01), 2)


def base_prm(t):
    return dict(t_s2=float(t), t_s3=float(t), t_anchor=0.0, exclusive=False, rel=0.0, top_n=0)


def tune_global(ev):
    res = [(t, ev.metrics(base_prm(t))['F05']) for t in T_GRID]
    t, f = max(res, key=lambda x: x[1])
    return base_prm(t), f


def tune_staged(ev):
    """global → exclusivity → per-source → anchor+expansion → relative margin (each kept only if it helps)."""
    H = []

    def run(prm, stage):
        m = ev.metrics(prm)
        H.append(dict(stage=stage, **prm, F05=m['F05']))
        return m['F05']
    best, bf = None, -1.0
    for t in T_GRID:
        f = run(base_prm(t), '1 global')
        if f > bf:
            best, bf = base_prm(t), f
    be, bfe = None, -1.0
    for t in T_GRID:
        prm = dict(base_prm(t), exclusive=True)
        f = run(prm, '2 exclusive')
        if f > bfe:
            be, bfe = prm, f
    if bfe >= bf - 1e-4:
        best, bf = be, bfe
    base = dict(best)
    grid = np.unique(np.round(np.clip(np.arange(base['t_s2'] - 0.2, base['t_s2'] + 0.201, 0.02), 0.02, 0.97), 2))
    for a in grid:
        for b in grid:
            prm = dict(base, t_s2=float(a), t_s3=float(b))
            f = run(prm, '3 per-source')
            if f > bf + 1e-6:
                best, bf = prm, f
    base = dict(best)
    lo = max(base['t_s2'], base['t_s3'])
    for ta in np.round(np.arange(lo, 0.99, 0.01), 2):
        for d in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            prm = dict(base, t_anchor=float(ta), t_s2=float(max(0.02, base['t_s2'] - d)), t_s3=float(max(0.02, base['t_s3'] - d)))
            f = run(prm, '4 anchor+expansion')
            if f > bf + 1e-6:
                best, bf = prm, f
    base = dict(best)
    for r in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        prm = dict(base, rel=r)
        f = run(prm, '5 relative')
        if f > bf + 1e-6:
            best, bf = prm, f
    return best, bf, pd.DataFrame(H)


# ---------------------------------------------------------------- expected-F0.5 optimal set selection
def _expf_groups_py(off, p, r_extra, beta2, nmax):
    """For each group (candidates sorted by p desc): choose k (0..n) maximising E[F_beta(top-k)] under independent
    Bernoulli labels + an 'unseen true match' Bernoulli(r_extra) (blocking misses). Returns chosen k per group."""
    G = off.shape[0] - 1
    best_k = np.zeros(G, np.int64)
    for g in range(G):
        s = off[g]; e = off[g + 1]
        n = e - s
        if n > nmax:
            n = nmax
        if n == 0:
            continue
        q = p[s:s + n]
        # suffix Poisson-binomial distributions: suf[k][b] = P(#true among q[k:] == b)
        suf = np.zeros((n + 1, n + 1))
        suf[n, 0] = 1.0
        for k in range(n - 1, -1, -1):
            for b in range(0, n - k + 1):
                v = suf[k + 1, b] * (1 - q[k])
                if b > 0:
                    v += suf[k + 1, b - 1] * q[k]
                suf[k, b] = v
        # k = 0: F = 1 iff no true anywhere
        best = suf[0, 0] * (1 - r_extra)
        bk = 0
        pre = np.zeros(n + 1)
        pre[0] = 1.0
        for k in range(1, n + 1):
            new = np.zeros(n + 1)
            for a in range(0, k + 1):
                v = pre[a] * (1 - q[k - 1])
                if a > 0:
                    v += pre[a - 1] * q[k - 1]
                new[a] = v
            pre = new
            ef = 0.0
            for a in range(1, k + 1):
                if pre[a] == 0.0:
                    continue
                for b in range(0, n - k + 1):
                    pb = suf[k, b]
                    if pb == 0.0:
                        continue
                    f0 = (1 + beta2) * a / (k + beta2 * (a + b))
                    f1 = (1 + beta2) * a / (k + beta2 * (a + b + 1))
                    ef += pre[a] * pb * ((1 - r_extra) * f0 + r_extra * f1)
            if ef > best:
                best = ef
                bk = k
        best_k[g] = bk
    return best_k


try:
    import numba
    _expf_groups = numba.njit(_expf_groups_py)
except Exception:
    _expf_groups = _expf_groups_py


def expf_select(qi, p, r_extra=0.02, nmax=20, exclusive_mask=None):
    """Bayes-optimal (under independence) set per S1. qi must be sorted; returns boolean selection mask."""
    p = np.asarray(p, np.float64).copy()
    if exclusive_mask is not None:
        p[~exclusive_mask] = 0.0
    order = np.lexsort((-p, qi))
    qs, ps = qi[order], p[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    off = np.r_[starts, len(qs)].astype(np.int64)
    k = _expf_groups(off, ps, float(r_extra), BETA2, int(nmax))
    rank = np.arange(len(qs)) - np.repeat(off[:-1], np.diff(off))
    sel_sorted = rank < np.repeat(k, np.diff(off))
    sel = np.zeros(len(p), bool)
    sel[order] = sel_sorted
    return sel


def fit_calibrator(p, y, kind='isotonic'):
    if kind == 'none':
        return None
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(p, y)
    return iso


def apply_calibrator(cal, p):
    return p if cal is None else cal.predict(p).astype(np.float32)


def tune_expf(qi, pi, p, y, q_eval, n_true, exclusive_opts=(False, True)):
    """Search r_extra / sharpening power / exclusivity for expected-F selection on a tune fold."""
    best, bf, H = None, -1, []
    bp = best_for_pool(pi, p)
    for ex in exclusive_opts:
        for r in (0.0, 0.01, 0.02, 0.05, 0.1):
            for g in (0.8, 1.0, 1.25, 1.5):
                sel = expf_select(qi, np.clip(p, 0, 1) ** g, r, exclusive_mask=bp if ex else None)
                m = s1_metrics(qi[sel], y[sel], n_true, q_eval)
                H.append(dict(exclusive=ex, r_extra=r, power=g, F05=m['F05']))
                if m['F05'] > bf:
                    best, bf = dict(mode='expf', exclusive=ex, r_extra=r, power=g), m['F05']
    return best, bf, pd.DataFrame(H)


def apply_rule(prm, qi, pi, p, s3, nq):
    """Apply a tuned rule (staged thresholds or expf) to arbitrary candidates (used for hold and test)."""
    if prm.get('mode') == 'expf':
        bp = best_for_pool(pi, p) if prm['exclusive'] else None
        return expf_select(qi, np.clip(p, 0, 1) ** prm['power'], prm['r_extra'], exclusive_mask=bp)
    bp = best_for_pool(pi, p)
    rk = pd.Series(p).groupby(qi).rank(ascending=False, method='first').to_numpy() if prm.get('top_n') else np.ones(len(p))
    return decide(qi, p, s3, bp, rk, prm, nq)
