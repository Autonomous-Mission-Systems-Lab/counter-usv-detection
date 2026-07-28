"""Class-conditional one-class benign-behavior models.

Per envelope, fit GMM (primary) + Mahalanobis + IsolationForest on
standardized kinematic features. Score convention: **higher = more anomalous**.
Thresholds are percentiles of held-out *benign* validation scores (FAR targets).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

Subspace = Literal["core", "core_course"]
ModelName = Literal["gmm", "mahalanobis", "isolation_forest"]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def select_envelope_rows(df: pd.DataFrame, members: dict) -> pd.DataFrame:
    """Filter rows by envelope ``members`` predicates (class / transceiver)."""
    out = df
    for col, allowed in (members or {}).items():
        if col not in out.columns:
            raise KeyError(f"envelope member key {col!r} not in columns")
        allowed_set = {str(x) for x in allowed}
        out = out.loc[out[col].astype(str).isin(allowed_set)]
    return out


@dataclass
class FittedSubspace:
    """One fitted model family on one feature subspace."""

    subspace: Subspace
    feature_names: list[str]
    scaler: StandardScaler
    model_name: ModelName
    model: Any
    # GMM only
    n_components: int | None = None
    val_loglik: float | None = None
    n_train: int = 0
    n_val: int = 0
    # Calibrated thresholds: far_target (float) → score threshold
    thresholds: dict[str, float] = field(default_factory=dict)
    val_score_percentiles: dict[str, float] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. ``X`` is raw (unscaled) feature matrix."""
        Xs = self.transform(X)
        if self.model_name == "gmm":
            return -self.model.score_samples(Xs)
        if self.model_name == "mahalanobis":
            return self.model.mahalanobis(Xs)
        if self.model_name == "isolation_forest":
            return -self.model.score_samples(Xs)
        raise ValueError(self.model_name)


@dataclass
class EnvelopeModel:
    """All subspaces + model families for one envelope."""

    name: str
    members: dict
    subspaces: dict[str, dict[str, FittedSubspace]]  # subspace → model_name → fit
    core_features: list[str]
    course_features: list[str]
    window_s: float
    primary: ModelName = "gmm"

    def choose_subspace(self, row: pd.Series | dict) -> Subspace:
        course_ok = all(
            pd.notna(row.get(c)) for c in self.course_features
        ) if self.course_features else False
        return "core_course" if course_ok and "core_course" in self.subspaces else "core"

    def score_row(
        self,
        row: pd.Series | dict,
        *,
        model_name: ModelName | None = None,
    ) -> dict[str, Any]:
        model_name = model_name or self.primary
        sub = self.choose_subspace(row)
        if sub not in self.subspaces or model_name not in self.subspaces[sub]:
            # fall back to core
            sub = "core"
        fit = self.subspaces[sub][model_name]
        feats = fit.feature_names
        x = np.array([[float(row[c]) for c in feats]], dtype="float64")
        if np.isnan(x).any():
            return {
                "score": None, "subspace": sub, "model": model_name,
                "note": "nan_features",
            }
        score = float(fit.anomaly_score(x)[0])
        return {"score": score, "subspace": sub, "model": model_name}

    def score_frame(
        self,
        df: pd.DataFrame,
        *,
        model_name: ModelName | None = None,
    ) -> pd.DataFrame:
        model_name = model_name or self.primary
        records = [self.score_row(r, model_name=model_name) for _, r in df.iterrows()]
        out = pd.DataFrame(records)
        out.index = df.index
        return out


@dataclass
class MultiHorizonEnvelope:
    """One envelope fitted at several observation horizons.

    At score time the caller passes the set of horizons whose window is
    *complete* for the contact; the longest such horizon is used, so short
    tracks still get scored at a shorter horizon instead of being dropped.
    """

    name: str
    members: dict
    primary_window_s: int
    core_features: list[str]
    course_features: list[str]
    horizons: dict[int, EnvelopeModel]  # window_s → model
    primary: ModelName = "gmm"

    def available_windows(self) -> list[int]:
        return sorted(self.horizons)

    def select_window(self, complete_windows: "set[int] | list[int]") -> int | None:
        """Longest fitted horizon whose window is complete for the contact."""
        cand = [w for w in self.horizons if w in set(complete_windows)]
        return max(cand) if cand else None

    def score_row(
        self,
        rows_by_window: dict[int, "pd.Series | dict"],
        complete_windows: "set[int] | list[int]",
        *,
        model_name: ModelName | None = None,
    ) -> dict[str, Any]:
        w = self.select_window(complete_windows)
        if w is None:
            return {"score": None, "window_s": None, "note": "no_complete_window"}
        res = self.horizons[w].score_row(rows_by_window[w], model_name=model_name)
        res["window_s"] = w
        return res


class _MahalanobisModel:
    """Thin wrapper: fit Ledoit-Wolf on already-scaled X; score = mahalanobis."""

    def __init__(self) -> None:
        self.cov_: LedoitWolf | None = None

    def fit(self, X: np.ndarray) -> "_MahalanobisModel":
        self.cov_ = LedoitWolf().fit(X)
        return self

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        assert self.cov_ is not None
        return self.cov_.mahalanobis(X)


def _matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, pd.Index]:
    sub = df.dropna(subset=cols)
    X = sub[cols].to_numpy(dtype="float64")
    return X, sub.index


def fit_gmm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    n_components_sweep: list[int],
    covariance_type: str,
    reg_covar: float,
    max_iter: int,
    n_init: int,
    seed: int,
    selection_tol_nats: float = 0.0,
) -> tuple[GaussianMixture, int, float]:
    """Fit GMMs across ``n_components_sweep`` and pick ``k`` by held-out val
    log-likelihood.

    With ``selection_tol_nats > 0`` a **knee rule** is used: the smallest ``k``
    whose val mean log-likelihood is within ``selection_tol_nats`` of the best.
    This keeps selection from pinning at the top of the sweep when extra
    components only add marginal (flat) likelihood — a more parsimonious,
    less overfit density than raw argmax.
    """
    fits: list[tuple[int, float, GaussianMixture]] = []
    for k in n_components_sweep:
        if k > len(X_train):
            continue
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )
        gmm.fit(X_train)
        ll = float(gmm.score(X_val)) if len(X_val) else float(gmm.score(X_train))
        fits.append((k, ll, gmm))
    if not fits:
        raise RuntimeError("GMM fit failed for all k")
    best_ll = max(ll for _, ll, _ in fits)
    tol = max(0.0, float(selection_tol_nats))
    for k, ll, gmm in sorted(fits, key=lambda r: r[0]):
        if ll >= best_ll - tol:
            return gmm, k, ll
    k, ll, gmm = max(fits, key=lambda r: r[1])
    return gmm, k, ll


def calibrate_thresholds(
    scores: np.ndarray,
    far_targets: list[float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map FAR target → score threshold (percentile of benign scores)."""
    scores = scores[np.isfinite(scores)]
    thresholds: dict[str, float] = {}
    pcts: dict[str, float] = {}
    if len(scores) == 0:
        return thresholds, pcts
    for far in far_targets:
        q = 100.0 * (1.0 - far)
        thr = float(np.percentile(scores, q))
        key = f"far_{far:g}"
        thresholds[key] = thr
        pcts[f"p{q:g}"] = thr
    return thresholds, pcts


def fit_envelope(
    name: str,
    members: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    cfg: dict,
) -> EnvelopeModel | None:
    """Fit all model families / subspaces for one envelope."""
    feat_cfg = cfg.get("features") or {}
    core_cols = list(feat_cfg.get("core") or [])
    course_cols = list(feat_cfg.get("course") or [])
    subspaces_cfg = list(cfg.get("subspaces") or ["core", "core_course"])
    models_cfg = cfg.get("models") or {}
    primary = str(models_cfg.get("primary") or "gmm")
    baselines = list(models_cfg.get("baselines") or [])
    families: list[ModelName] = [primary] + [b for b in baselines if b != primary]  # type: ignore
    far_targets = list((cfg.get("calibration") or {}).get("far_targets") or [0.05])
    min_rows = int(cfg.get("min_train_rows") or 50)
    seed = int(cfg.get("seed") or 1337)
    window_s = float(cfg.get("window_s") or 300)

    train_e = select_envelope_rows(train_df, members)
    val_e = select_envelope_rows(val_df, members)
    if len(train_e) < min_rows:
        print(f"  [skip] {name}: only {len(train_e)} train rows (<{min_rows})")
        return None

    gmm_cfg = models_cfg.get("gmm") or {}
    if_cfg = models_cfg.get("isolation_forest") or {}

    fitted: dict[str, dict[str, FittedSubspace]] = {}

    for subspace in subspaces_cfg:
        cols = list(core_cols) if subspace == "core" else list(core_cols) + list(course_cols)
        X_tr, idx_tr = _matrix(train_e, cols)
        X_va, idx_va = _matrix(val_e, cols)
        if len(X_tr) < min_rows:
            print(f"  [skip] {name}/{subspace}: {len(X_tr)} usable train rows")
            continue

        scaler = StandardScaler().fit(X_tr)
        Xs_tr = scaler.transform(X_tr)
        Xs_va = scaler.transform(X_va) if len(X_va) else Xs_tr[:0]

        fitted.setdefault(subspace, {})

        for family in families:
            if family == "gmm":
                model, k, ll = fit_gmm(
                    Xs_tr, Xs_va if len(Xs_va) else Xs_tr,
                    n_components_sweep=list(gmm_cfg.get("n_components_sweep") or [1, 2, 3]),
                    covariance_type=str(gmm_cfg.get("covariance_type") or "full"),
                    reg_covar=float(gmm_cfg.get("reg_covar") or 1e-6),
                    max_iter=int(gmm_cfg.get("max_iter") or 200),
                    n_init=int(gmm_cfg.get("n_init") or 3),
                    seed=seed,
                    selection_tol_nats=float(gmm_cfg.get("selection_tol_nats") or 0.0),
                )
                fs = FittedSubspace(
                    subspace=subspace,  # type: ignore[arg-type]
                    feature_names=cols,
                    scaler=scaler,
                    model_name="gmm",
                    model=model,
                    n_components=k,
                    val_loglik=ll,
                    n_train=len(X_tr),
                    n_val=len(X_va),
                )
            elif family == "mahalanobis":
                model = _MahalanobisModel().fit(Xs_tr)
                fs = FittedSubspace(
                    subspace=subspace,  # type: ignore[arg-type]
                    feature_names=cols,
                    scaler=scaler,
                    model_name="mahalanobis",
                    model=model,
                    n_train=len(X_tr),
                    n_val=len(X_va),
                )
            elif family == "isolation_forest":
                model = IsolationForest(
                    n_estimators=int(if_cfg.get("n_estimators") or 200),
                    max_samples=if_cfg.get("max_samples") or "auto",
                    contamination=if_cfg.get("contamination") or "auto",
                    random_state=seed,
                    n_jobs=-1,
                ).fit(Xs_tr)
                fs = FittedSubspace(
                    subspace=subspace,  # type: ignore[arg-type]
                    feature_names=cols,
                    scaler=scaler,
                    model_name="isolation_forest",
                    model=model,
                    n_train=len(X_tr),
                    n_val=len(X_va),
                )
            else:
                raise ValueError(family)

            # Calibrate on val (fall back to train if val empty).
            X_cal = X_va if len(X_va) else X_tr
            scores = fs.anomaly_score(X_cal)
            thr, pct = calibrate_thresholds(scores, far_targets)
            fs.thresholds = thr
            fs.val_score_percentiles = pct
            fitted[subspace][family] = fs
            print(
                f"  [{name}/{subspace}/{family}] train={len(X_tr)} val={len(X_va)}"
                + (f" k={fs.n_components} ll={fs.val_loglik:.3f}" if family == "gmm" else "")
                + f" thr_far0.05={thr.get('far_0.05', float('nan')):.3f}"
            )

    if not fitted:
        return None
    return EnvelopeModel(
        name=name,
        members=members,
        subspaces=fitted,
        core_features=core_cols,
        course_features=course_cols,
        window_s=window_s,
        primary=primary,  # type: ignore[arg-type]
    )


def save_envelope(model: "EnvelopeModel | MultiHorizonEnvelope", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_envelope(path: Path) -> "EnvelopeModel | MultiHorizonEnvelope":
    return joblib.load(path)


def envelope_summary(model: EnvelopeModel) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": model.name,
        "members": model.members,
        "window_s": model.window_s,
        "primary": model.primary,
        "core_features": model.core_features,
        "course_features": model.course_features,
        "subspaces": {},
    }
    for sub, fams in model.subspaces.items():
        out["subspaces"][sub] = {}
        for fam, fs in fams.items():
            out["subspaces"][sub][fam] = {
                "n_train": fs.n_train,
                "n_val": fs.n_val,
                "n_components": fs.n_components,
                "val_loglik": fs.val_loglik,
                "thresholds": fs.thresholds,
                "feature_names": fs.feature_names,
            }
    return out
