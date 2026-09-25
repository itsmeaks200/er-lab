"""Model zoo on the pair-feature matrix: LightGBM (binary / lambdarank), XGBoost, CatBoost, logistic regression, MLP.
All models expose fit(Xtr, ytr, gtr, Xva, yva, gva) and predict(X) -> score in [0,1] (or rank score, calibrated later)."""
import os

import numpy as np

from .utils import log

DEFAULT_PARAMS = {
    'lgb': dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1),
    'lgb_rank': dict(objective='lambdarank', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                     lambdarank_truncation_level=20, eval_at=[5]),
    'xgb': dict(objective='binary:logistic', eval_metric='logloss', eta=0.05, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=10, tree_method='hist', max_bin=256),
    'cat': dict(learning_rate=0.08, depth=8, l2_leaf_reg=3.0, eval_metric='Logloss'),
    'lr': dict(C=1.0),
    'mlp': dict(hidden=512, layers=3, dropout=0.15, lr=2e-3, epochs=6, batch=8192, wd=1e-5),
}


def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _group_sizes(g):
    """g: qi array sorted → group sizes."""
    if g is None:
        return None
    _, cnt = np.unique(g, return_counts=True)
    return cnt


class Model:
    def __init__(self, kind, params, rounds=5000, early_stop=200, seed=42, feats=None):
        self.kind, self.seed, self.rounds, self.early_stop = kind, seed, rounds, early_stop
        self.params = dict(DEFAULT_PARAMS[kind]); self.params.update(params or {})
        self.feats = feats
        self.obj = None

    def fit(self, Xtr, ytr, gtr=None, Xva=None, yva=None, gva=None):
        k, p = self.kind, dict(self.params)
        if k in ('lgb', 'lgb_rank'):
            import lightgbm as lgb
            p.update(seed=self.seed, num_threads=os.cpu_count())
            dtr = lgb.Dataset(Xtr, ytr, group=_group_sizes(gtr) if k == 'lgb_rank' else None, free_raw_data=True)
            dva = lgb.Dataset(Xva, yva, group=_group_sizes(gva) if k == 'lgb_rank' else None, reference=dtr)
            self.obj = lgb.train(p, dtr, num_boost_round=self.rounds, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(self.early_stop, verbose=False), lgb.log_evaluation(500)])
            self.best_iter = self.obj.best_iteration
        elif k == 'xgb':
            import xgboost as xgb
            p['seed'] = self.seed
            if _gpu():
                if int(xgb.__version__.split('.')[0]) >= 2:
                    p['device'] = 'cuda'
                else:
                    p['tree_method'] = 'gpu_hist'
            p['nthread'] = os.cpu_count()
            self.obj = xgb.train(p, xgb.DMatrix(Xtr, ytr, missing=np.nan), self.rounds,
                                 evals=[(xgb.DMatrix(Xva, yva, missing=np.nan), 'va')],
                                 early_stopping_rounds=self.early_stop, verbose_eval=500)
            self.best_iter = self.obj.best_iteration
        elif k == 'cat':
            from catboost import CatBoostClassifier
            self.obj = CatBoostClassifier(iterations=self.rounds, random_seed=self.seed, od_type='Iter',
                                          od_wait=self.early_stop, task_type='GPU' if _gpu() else 'CPU',
                                          thread_count=os.cpu_count(), verbose=500,
                                          train_dir=os.path.join(os.environ.get('ER_CACHE', './cache'), 'catboost_info'), **p)
            self.obj.fit(Xtr, ytr, eval_set=(Xva, yva))
            self.best_iter = self.obj.get_best_iteration()
        elif k == 'lr':
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            n = min(len(Xtr), 3_000_000)
            idx = np.random.RandomState(self.seed).choice(len(Xtr), n, replace=False) if n < len(Xtr) else slice(None)
            self.obj = make_pipeline(SimpleImputer(strategy='constant', fill_value=-1), StandardScaler(),
                                     LogisticRegression(max_iter=500, C=p['C']))
            self.obj.fit(Xtr[idx], ytr[idx])
            self.best_iter = 0
        elif k == 'mlp':
            self._fit_mlp(Xtr, ytr, Xva, yva)
        else:
            raise ValueError(k)
        return self

    def fit_fixed(self, Xtr, ytr, gtr=None):
        """Train exactly self.rounds rounds without a validation set (refit on all labelled rows after the holdout
        phase, and out-of-fold models): no early stopping on a tiny or in-sample validation set."""
        k, p = self.kind, dict(self.params)
        rounds = max(1, int(self.rounds))
        if k in ('lgb', 'lgb_rank'):
            import lightgbm as lgb
            p.update(seed=self.seed, num_threads=os.cpu_count())
            p.pop('eval_at', None)
            self.obj = lgb.train(p, lgb.Dataset(Xtr, ytr, group=_group_sizes(gtr) if k == 'lgb_rank' else None),
                                 num_boost_round=rounds)
            self.best_iter = rounds
        elif k == 'xgb':
            import xgboost as xgb
            p['seed'] = self.seed
            if _gpu():
                if int(xgb.__version__.split('.')[0]) >= 2:
                    p['device'] = 'cuda'
                else:
                    p['tree_method'] = 'gpu_hist'
            p['nthread'] = os.cpu_count()
            self.obj = xgb.train(p, xgb.DMatrix(Xtr, ytr, missing=np.nan), rounds)
            self.best_iter = rounds - 1
        elif k == 'cat':
            from catboost import CatBoostClassifier
            self.obj = CatBoostClassifier(iterations=rounds, random_seed=self.seed, task_type='GPU' if _gpu() else 'CPU',
                                          thread_count=os.cpu_count(), verbose=500,
                                          train_dir=os.path.join(os.environ.get('ER_CACHE', './cache'), 'catboost_info'), **p)
            self.obj.fit(Xtr, ytr)
            self.best_iter = rounds
        else:                                   # lr / mlp: no early stopping anyway
            self.fit(Xtr, ytr, gtr, None, None, None)
        return self

    def predict(self, X):
        k = self.kind
        if k in ('lgb', 'lgb_rank'):
            s = self.obj.predict(X, num_iteration=self.best_iter)
            return 1 / (1 + np.exp(-s)) if k == 'lgb_rank' else s
        if k == 'xgb':
            import xgboost as xgb
            return self.obj.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, self.best_iter + 1))
        if k in ('cat', 'lr'):
            return self.obj.predict_proba(X)[:, 1]
        if k == 'mlp':
            return self._predict_mlp(X)
        raise ValueError(k)

    def importance(self):
        if self.kind in ('lgb', 'lgb_rank'):
            return self.obj.feature_importance('gain')
        return None

    # ---------------- MLP (quantile-normalised features, GPU)
    def _prep(self, X, fit=False):
        from sklearn.preprocessing import QuantileTransformer
        Xn = np.where(np.isnan(X), -1.0, X).astype(np.float32)
        nanf = np.isnan(X).astype(np.float32)
        if fit:
            n = min(len(Xn), 1_000_000)
            self.qt = QuantileTransformer(n_quantiles=256, output_distribution='normal', subsample=n,
                                          random_state=self.seed).fit(Xn[:n])
            self.nan_cols = nanf.any(axis=0)
        return np.hstack([self.qt.transform(Xn).astype(np.float32), nanf[:, self.nan_cols]])

    def _fit_mlp(self, Xtr, ytr, Xva, yva):
        import torch
        p, dev = self.params, 'cuda' if _gpu() else 'cpu'
        torch.manual_seed(self.seed)
        A = self._prep(Xtr, fit=True)
        d = A.shape[1]
        layers, w = [], d
        for _ in range(p['layers']):
            layers += [torch.nn.Linear(w, p['hidden']), torch.nn.BatchNorm1d(p['hidden']), torch.nn.SiLU(), torch.nn.Dropout(p['dropout'])]
            w = p['hidden']
        layers += [torch.nn.Linear(w, 1)]
        self.net = torch.nn.Sequential(*layers).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        steps = p['epochs'] * (len(A) // p['batch'] + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p['lr'], total_steps=steps)
        At, yt = torch.from_numpy(A), torch.from_numpy(ytr.astype(np.float32))
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(p['epochs']):
            self.net.train()
            perm = torch.randperm(len(At))
            tot = 0.0
            for s in range(0, len(At), p['batch']):
                bi = perm[s:s + p['batch']]
                xb, yb = At[bi].to(dev, non_blocking=True), yt[bi].to(dev, non_blocking=True)
                loss = lossf(self.net(xb).squeeze(1), yb)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(bi)
            log(f'   mlp epoch {ep + 1}: train logloss {tot / len(At):.4f}')
        self.best_iter = p['epochs']

    def _predict_mlp(self, X):
        import torch
        dev = 'cuda' if _gpu() else 'cpu'
        A = torch.from_numpy(self._prep(X))
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(A), 65536):
                out.append(torch.sigmoid(self.net(A[s:s + 65536].to(dev)).squeeze(1)).cpu().numpy())
        return np.concatenate(out)


def rank01(x):
    from scipy.stats import rankdata
    return (rankdata(x) / len(x)).astype(np.float32)


def blend_weights(preds_tune, y_tune, eval_fn, iters=30, seed=42):
    """Hill-climbing (forward selection with replacement) on the tune fold; eval_fn(p) -> F0.5."""
    names = list(preds_tune)
    chosen, best_f = [], -1
    for _ in range(iters):
        cand = None
        for n in names:
            trial = chosen + [n]
            p = np.mean([preds_tune[m] for m in trial], axis=0)
            f = eval_fn(p)
            if cand is None or f > cand[1]:
                cand = (n, f)
        if cand[1] <= best_f + 1e-6:
            break
        chosen.append(cand[0]); best_f = cand[1]
    w = {n: chosen.count(n) / len(chosen) for n in set(chosen)}
    return w, best_f
