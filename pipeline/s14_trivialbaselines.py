"""
s14_trivialbaselines.py
-----------------------
A family of ZERO-IMAGE null models for slice-level medical imaging benchmarks, and
an audit harness that runs them on nothing but a published label table.

WHAT THIS IS FOR
    Slice-level classification benchmarks are usually reported as a slice-level AUROC
    over a test set of volumes. That number can be reached, in part or in whole, by a
    model that never sees a pixel -- because the label is correlated with WHERE a slice
    sits in its stack, with HOW the volume was acquired, or with WHICH release batch it
    came from. This module implements those null models and measures how much of a
    published number they reach.

    None of the individual phenomena here are new. Shortcut learning, leakage in
    ML-based science, and the gap between slice-level and patient-level evaluation are
    all documented failure modes. What is missing from the literature is a quantity:
    for a given published benchmark, HOW MUCH of the headline number survives when the
    pixels are taken away. That is what this tool computes, and it computes it from
    the artefact almost every dataset already publishes -- the label file.

WHAT IT NEEDS
    One tidy table (CSV/TSV/Parquet) with, at minimum:
        * a subject / patient identifier
        * a slice index (or z position)
        * a label
    and optionally:
        * the dataset's own split column
        * any number of metadata columns (scanner, protocol, batch, matrix size, ...)

    No images. No k-space. No data-use agreement for pixels. No GPU. That is the whole
    point: a benchmark can be audited by anyone who can download its label file, which
    means benchmarks whose images we will never hold can still be audited.

THE BASELINES
    1. POSITIONAL   P(label | relative slice position), binned, fitted on TRAIN slices
                    and applied to TEST slices. Relative position is
                    (slice - min_slice_in_volume) / (max - min) so volumes of different
                    depth are comparable. Reported with a bin sweep (5/10/20/50) so the
                    result cannot be dismissed as a binning artefact.
    2. METADATA     label from metadata columns alone, via a depth-limited CART. Also
                    reported per column univariately, and as the single best column,
                    because "one acquisition field beats the network" is the sentence
                    that lands.
    3. VOLUME-SIZE  label from the number of slices in the volume alone -- a proxy for
                    protocol, scanner and acquisition era.
    4. PREVALENCE   the constant predictor. Its AUROC is 0.5 by construction; it exists
                    to define the chance anchor for every other statistic here, and to
                    prove the harness is not silently rewarding a degenerate model.
    5. COMBINED     position + metadata in one tree: the ceiling reachable with no
                    pixels at all.

DISCIPLINE
    Every baseline implements fit(train) / score(rows). The harness fits on training
    rows only and scores test rows only; the split is subject-level (never a random
    slice split), it prefers the dataset's OWN split when one is published, and it
    asserts and records that the train and test subject sets are disjoint. Apparent
    (train) performance is reported next to test performance so overfitting of the
    null model itself is visible rather than assumed away.

    Each baseline is also calibrated against a PERMUTATION NULL that destroys exactly
    the association it tests and holds everything else fixed, because a baseline's null
    is not automatically 0.5. Fit a metadata model on the training folds of a dataset
    whose label lives at subject level, score the held-out fold, and the rate you fitted
    is anti-correlated with the rate you are scoring -- positives are a finite
    population, so a level that was positive-rich in training is positive-poor in the
    fold left out. Pooled over folds the noise cancels and that bias does not. On a
    synthetic dataset whose label was by construction invisible to metadata we measured
    the metadata baseline at 0.424, not 0.500. Judging it against 0.5 would have
    manufactured a below-chance "finding" out of arithmetic. Where a permutation cannot
    change anything -- shuffling labels within a volume that is single-class -- the null
    is reported as unavailable rather than as a p of 1.0.

WHAT THE OUTPUT DOES AND DOES NOT LICENCE YOU TO SAY
    A high trivial fraction says: THE EVALUATION PROTOCOL of the published benchmark is
    matched, to that extent, by a model with no access to the images. It does NOT say
    the published model learned nothing. Those are different claims and only the first
    one is supported by a label file. See TRIVIAL_FRACTION_LIMITS below and the
    "Interpretation" block of every generated card.

Usage:
    python pipeline/s14_trivialbaselines.py --self-test
    python pipeline/s14_trivialbaselines.py --labels labels.csv --name mybench
    python pipeline/s14_trivialbaselines.py --labels t2_slice_level_labels.csv \\
        --label-col PIRADS --positive-if '>2' --name rempe_t2 --published 0.861
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s04_stats  # noqa: E402  -- statistics live in one place for the whole study

# Internal column names. Prefixed so they cannot collide with a real dataset column;
# the loader asserts they are absent from the input rather than silently overwriting.
C_SUBJ = "__tb_subject"
C_SLICE = "__tb_slice"
C_LABEL = "__tb_label"
C_VOL = "__tb_volume"
C_RELPOS = "__tb_relpos"
C_NSLICES = "__tb_nslices"
C_SPLIT = "__tb_split"
RESERVED = (C_SUBJ, C_SLICE, C_LABEL, C_VOL, C_RELPOS, C_NSLICES, C_SPLIT)

DEFAULT_BINS = 20
BIN_SWEEP = (5, 10, 20, 50)

# --- column auto-detection --------------------------------------------------
# Ordered by preference. Matching is case-insensitive on the exact name first, then
# on a normalised name (non-alphanumerics stripped). Every one of these is
# overridable from the CLI; auto-detection is a convenience, never a requirement.
ALIASES: dict[str, tuple[str, ...]] = {
    "subject": (
        "subject_id", "patient_id", "subjectid", "patientid", "fastmri_pt_id",
        "pt_id", "case_id", "study_id", "patient", "subject", "id", "bcr_patient_barcode",
        "patient_uid", "seriesinstanceuid_patient", "anon_id", "pid",
    ),
    "slice": (
        "slice", "slice_idx", "slice_index", "slice_number", "slicenumber", "z",
        "z_index", "instance_number", "instancenumber", "slice_id", "img_index",
        "position", "z_position", "slice_location", "sliceloc", "frame",
    ),
    "label": (
        "label", "target", "y", "class", "diagnosis", "malignant", "malignancy",
        "case_label", "gt", "ground_truth", "lesion", "pirads", "pi_rads",
        "is_positive", "positive", "abnormal", "finding",
    ),
    "split": (
        "split", "data_split", "official_split", "set", "subset", "partition",
        "fold_type", "train_test", "datasplit", "dataset_split", "group_split",
    ),
    "volume": (
        "file", "filename", "fastmri_rawfile", "series_id", "seriesinstanceuid",
        "series", "volume_id", "volume", "scan_id", "series_uid", "study_uid",
        "studyinstanceuid", "exam_id", "acquisition_id", "raw_file",
    ),
}

# Split vocabulary. Anything unrecognised is reported and excluded, never guessed at.
SPLIT_VOCAB = {
    "train": ("train", "training", "tr", "fit", "dev", "development", "discovery",
              "learn", "trainval", "0"),
    "test": ("test", "testing", "holdout", "hold_out", "holdout_test", "eval",
             "evaluation", "external", "2"),
    "val": ("val", "valid", "validation", "tune", "tuning", "dev_val", "1"),
}

# Metadata columns are supposed to be ACQUISITION or ADMINISTRATIVE facts. Two families
# must be excluded by default or the audit is worthless:
#
#   OUTCOME-DERIVED  a column that is the label under another name, or a component of
#                    it (PI-RADS when the label is PI-RADS>2; tumour grade; receptor
#                    status). Its association with the label is tautological.
#
#   IMAGE-DERIVED    a column computed FROM the pixels (SNR, reconstruction NCC, mask
#                    fraction, phase std). Including one would break the zero-image
#                    guarantee that is the entire premise of the tool.
#
# These are name heuristics and they are FALLIBLE. Every excluded and every included
# column is printed and written to the JSON so a reader can check the call. Use
# --metadata-cols to state the set explicitly when it matters.
OUTCOME_PATTERNS = (
    "label", "target", "class", "diagnos", "pirads", "pi_rads", "gleason", "grade",
    "stage", "malign", "benign", "lesion", "tumor", "tumour", "cancer", "outcome",
    "survival", "recur", "mortality", "ground_truth", "groundtruth", "gt_", "_gt",
    "birads", "bi_rads", "response", "severity", "score_", "risk", "prognos",
)
OUTCOME_EXACT = ("er", "pr", "her2", "y", "gt", "n", "m", "t", "detection_type",
                 "reason", "menopause", "laterality")
IMAGE_DERIVED_PATTERNS = (
    "ncc", "psnr", "ssim", "snr", "mask_frac", "phase_std", "coherence", "intensity",
    "histogram", "entropy", "texture", "radiomic", "embed", "feature", "pred", "prob",
    "logit", "attr_max", "attr_norm", "qc", "dcf", "temptv", "resid", "phase_offset",
    "sharpness", "blur", "artifact_score", "motion_score",
)

TRIVIAL_FRACTION_LIMITS = [
    "The published number must be on the SAME metric, the SAME evaluation unit "
    "(slice vs patient) and a comparable test set. A slice-level baseline compared "
    "against a patient-level publication is meaningless.",
    "The fraction is undefined when the published number is at or below chance; it is "
    "reported as null in that case rather than as a large or negative number.",
    "It is NOT a decomposition. The baseline and the published model may exploit the "
    "same shortcut, different shortcuts, or overlapping ones. A fraction of 0.9 does "
    "not license 'the model learned nothing'; it licenses 'this evaluation protocol "
    "certifies a number that a pixel-blind model also reaches'.",
    "Values above 1 mean the zero-image baseline exceeded the published number. That "
    "is a real and reportable outcome, not an error, and it is left unclipped in "
    "'value' (the clipped copy exists only for plotting).",
    "The published number enters as a fixed constant: we almost never have its "
    "sampling distribution. The interval reported here propagates uncertainty in the "
    "BASELINE only and is therefore too narrow as a statement about the ratio.",
    "The baseline is fitted on the training rows of the same table. If the published "
    "model was trained on a different or larger set, the comparison is approximate.",
]


# ==========================================================================
# Loading and column resolution
# ==========================================================================

def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_table(path: Path) -> pd.DataFrame:
    """Read CSV / TSV / Parquet, and drop pandas' unnamed index column if present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label table not found: {path}")
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suf in (".tsv", ".tab"):
        df = pd.read_csv(path, sep="\t")
    else:
        # sep=None + python engine sniffs the delimiter, so a .csv that is really
        # semicolon- or tab-separated still loads.
        df = pd.read_csv(path, sep=None, engine="python")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed: ")]]
    clash = [c for c in df.columns if c in RESERVED]
    if clash:
        raise ValueError(f"input uses reserved column names {clash}; rename them")
    return df


def _pick(df: pd.DataFrame, role: str, override: str | None) -> str | None:
    """Resolve one role to a column name: explicit override, else alias search."""
    if override:
        if override not in df.columns:
            raise KeyError(f"--{role}-col {override!r} is not in the table; "
                           f"columns are {list(df.columns)}")
        return override
    lower = {str(c).lower(): c for c in df.columns}
    normed = {_norm(c): c for c in df.columns}
    for cand in ALIASES[role]:
        if cand in lower:
            return lower[cand]
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    return None


@dataclass
class ColumnMap:
    subject: str
    slice: str
    label: str
    split: str | None = None
    volume: str | None = None
    metadata: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"subject": self.subject, "slice": self.slice, "label": self.label,
                "split": self.split, "volume": self.volume,
                "metadata": list(self.metadata), "excluded": dict(self.excluded)}


def resolve_columns(df: pd.DataFrame, subject=None, slice_col=None, label=None,
                    split=None, volume=None, metadata=None, exclude=(),
                    allow_outcome=False, allow_image_derived=False) -> ColumnMap:
    """
    Work out which column plays which role, and which columns are admissible metadata.

    Roles are auto-detected from a table of common spellings and every one can be
    overridden. Metadata defaults to "everything left over that is not obviously
    outcome-derived, image-derived, a split marker, or a row identifier", with the
    reason for each exclusion recorded so the choice is auditable rather than magic.
    """
    subj = _pick(df, "subject", subject)
    sl = _pick(df, "slice", slice_col)
    lab = _pick(df, "label", label)
    if subj is None:
        raise KeyError("could not find a subject/patient id column; pass --subject-col")
    if sl is None:
        raise KeyError("could not find a slice index column; pass --slice-col")
    if lab is None:
        raise KeyError("could not find a label column; pass --label-col")
    spl = _pick(df, "split", split) if split != "" else None
    vol = _pick(df, "volume", volume) if volume != "" else None
    if vol is not None and vol in (subj, sl, lab, spl):
        vol = None

    roles = {subj, sl, lab, spl, vol} - {None}
    excluded: dict[str, str] = {}

    if metadata:
        missing = [c for c in metadata if c not in df.columns]
        if missing:
            raise KeyError(f"--metadata-cols not in table: {missing}")
        meta = list(metadata)
        for c in df.columns:
            if c not in meta:
                excluded[str(c)] = "not listed in --metadata-cols"
    else:
        meta = []
        n_subj = df[subj].nunique()
        for c in df.columns:
            cl = str(c).lower()
            if c in roles:
                excluded[str(c)] = "role column"
                continue
            if c in exclude:
                excluded[str(c)] = "excluded on the command line"
                continue
            if cl.endswith("_split") or cl.startswith("split") or cl.endswith("_fold"):
                excluded[str(c)] = "split marker"
                continue
            if not allow_outcome and (
                    cl in OUTCOME_EXACT or any(p in cl for p in OUTCOME_PATTERNS)):
                excluded[str(c)] = "name suggests outcome-derived (tautological)"
                continue
            if not allow_image_derived and any(p in cl for p in IMAGE_DERIVED_PATTERNS):
                excluded[str(c)] = "name suggests image-derived (breaks zero-image claim)"
                continue
            nun = df[c].nunique(dropna=True)
            if nun < 2:
                excluded[str(c)] = "constant"
                continue
            if nun == len(df):
                # One distinct value per row: a row counter or a UID. It carries the
                # table's own ordering, which is an artefact of how the file was
                # assembled rather than a fact about the acquisition.
                excluded[str(c)] = f"row-identifier-like ({nun} levels, {len(df)} rows)"
                continue
            if not pd.api.types.is_numeric_dtype(df[c]) and nun >= n_subj:
                # A non-numeric column with as many levels as there are subjects is a
                # per-subject identifier. On a subject-disjoint split it can only ever
                # score 0.5, so it adds noise and no information.
                excluded[str(c)] = f"identifier-like ({nun} levels, {n_subj} subjects)"
                continue
            meta.append(c)

    return ColumnMap(subject=str(subj), slice=str(sl), label=str(lab),
                     split=None if spl is None else str(spl),
                     volume=None if vol is None else str(vol),
                     metadata=[str(c) for c in meta], excluded=excluded)


# ==========================================================================
# Label binarisation
# ==========================================================================

def binarise(series: pd.Series, rule: str | None) -> tuple[np.ndarray, str]:
    """
    Turn a label column into {0,1} under an EXPLICIT, recorded rule.

    Supported rules: '>2', '>=3', '<2', '<=1', '==1', '!=1', 'in:3,4,5'.
    With no rule: a column already in {0,1} is taken as is; a column with exactly two
    distinct values maps the larger to 1; anything else is an error, because guessing
    a threshold on an ordinal scale is exactly the kind of silent choice this study
    exists to complain about.
    """
    s = series
    if rule:
        r = rule.strip()
        if r.lower().startswith("in:"):
            vals = [v.strip() for v in r[3:].split(",") if v.strip()]
            try:
                vals_typed: list = [float(v) for v in vals]
                y = s.astype(float).isin(vals_typed)
            except (TypeError, ValueError):
                y = s.astype(str).isin(vals)
            return y.astype(int).to_numpy(), f"{series.name} in {{{', '.join(vals)}}}"
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if r.startswith(op):
                rhs = r[len(op):].strip()
                try:
                    num = float(rhs)
                    lhs = pd.to_numeric(s, errors="coerce")
                except ValueError:
                    num, lhs = rhs, s.astype(str)  # type: ignore[assignment]
                y = {">=": lambda: lhs >= num, "<=": lambda: lhs <= num,
                     "==": lambda: lhs == num, "!=": lambda: lhs != num,
                     ">": lambda: lhs > num, "<": lambda: lhs < num}[op]()
                return y.fillna(False).astype(int).to_numpy(), f"{series.name} {op} {rhs}"
        raise ValueError(f"unparsable --positive-if rule {rule!r}")

    uniq = pd.unique(s.dropna())
    try:
        uset = set(float(u) for u in uniq)
    except (TypeError, ValueError):
        uset = None  # type: ignore[assignment]
    if uset is not None and uset <= {0.0, 1.0}:
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy(), \
            f"{series.name} already binary {{0,1}}"
    if len(uniq) == 2:
        order = sorted(uniq, key=lambda v: (str(type(v)), v))
        return (s == order[1]).astype(int).to_numpy(), \
            f"{series.name} == {order[1]!r} (larger of two values)"
    raise ValueError(
        f"label column {series.name!r} has {len(uniq)} distinct values "
        f"({sorted(uniq)[:10]}); pass --positive-if, e.g. --positive-if '>2'. "
        "Refusing to guess a threshold."
    )


# ==========================================================================
# Geometry: relative slice position, volume size
# ==========================================================================

def add_relative_position(df: pd.DataFrame, volume_col: str, slice_col: str,
                          out_col: str = C_RELPOS) -> pd.DataFrame:
    """
    Relative position of each slice within its OWN stack, in [0, 1].

    (slice - min) / (max - min) within the volume, so a 24-slice prostate stack and a
    38-slice one are on the same axis. Volumes with a single slice, where the span is
    zero, get 0.0: there is no interior to speak of and any other convention would
    invent structure.

    This generalises s12_rempe.add_relpos, which grouped on subject_id and hardcoded
    the 'slice' column; s12 now calls through to here.
    """
    out = df.copy()
    z = pd.to_numeric(out[slice_col], errors="coerce")
    top = z.groupby(out[volume_col]).transform("max")
    bottom = z.groupby(out[volume_col]).transform("min")
    span = (top - bottom).replace(0, 1)
    out[out_col] = (z - bottom) / span
    return out


def add_volume_size(df: pd.DataFrame, volume_col: str, out_col: str = C_NSLICES):
    """Number of LABELLED slices in each volume.

    Counted from the table rather than read from an 'n_slices' column, so it means the
    same thing on every dataset. If only a subset of slices is labelled this is the
    size of that subset -- which is itself a protocol fingerprint, and is what a model
    trained on the same table would see.
    """
    out = df.copy()
    out[out_col] = out.groupby(volume_col)[volume_col].transform("size").astype(float)
    return out


# ==========================================================================
# Splitting: the dataset's own split when it has one, else subject-level folds
# ==========================================================================

def normalise_split_values(values: pd.Series) -> tuple[pd.Series, dict]:
    """Map a split column onto {train, test, val, other}, reporting what it saw."""
    raw = values.astype(str).str.strip().str.lower()
    mapping = {}
    for canon, vocab in SPLIT_VOCAB.items():
        for v in vocab:
            mapping[v] = canon
    out = raw.map(lambda v: mapping.get(v, "other"))
    seen = {str(k): int(v) for k, v in raw.value_counts().items()}
    unknown = sorted({r for r, o in zip(raw, out) if o == "other"})
    return out, {"values_seen": seen, "unrecognised": unknown}


def subject_folds(subject_labels: pd.Series, k: int, seed: int) -> dict[str, int]:
    """
    Deterministic subject-level stratified k-fold assignment.

    Subjects (not slices) are the unit. Stratification is on the subject-level label
    (max over the subject's slices) so a small positive class is spread across folds
    instead of landing in one. Never a random slice split: that is the leak this whole
    paper is about.
    """
    rng = np.random.default_rng(seed)
    assign: dict[str, int] = {}
    for lab in sorted(subject_labels.unique()):
        subs = sorted(subject_labels.index[subject_labels == lab].astype(str).tolist())
        subs = [subs[i] for i in rng.permutation(len(subs))]
        for i, s in enumerate(subs):
            assign[s] = i % k
    return assign


# ==========================================================================
# Baselines
# ==========================================================================

class Baseline:
    """
    fit(train_rows) then score(rows). Never the other way round.

    The fitted state is derived from the training rows alone; score() refuses to run
    before fit() and records a fingerprint of the exact rows it was fitted on, so the
    JSON can be checked against the claim rather than believed.
    """

    name = "baseline"

    def __init__(self):
        self._fitted = False
        self.fit_fingerprint: str | None = None
        self.fit_n: int = 0

    def _fit(self, train: pd.DataFrame) -> None:
        raise NotImplementedError

    def _score(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> "Baseline":
        self._fit(train)
        self._fitted = True
        self.fit_n = int(len(train))
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(train.index.to_numpy(dtype=np.int64)).tobytes())
        h.update(np.ascontiguousarray(train[C_LABEL].to_numpy(dtype=np.int64)).tobytes())
        self.fit_fingerprint = h.hexdigest()[:16]
        return self

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: score() before fit(); refusing to run")
        s = np.asarray(self._score(rows), dtype=float)
        if len(s) != len(rows):
            raise RuntimeError(f"{self.name}: produced {len(s)} scores for {len(rows)} rows")
        return s

    def describe(self) -> dict:
        return {"name": self.name, "fit_rows": self.fit_n,
                "fit_fingerprint": self.fit_fingerprint}


class PrevalenceBaseline(Baseline):
    """The constant predictor: every slice gets the training prevalence.

    Its AUROC is exactly 0.5 (all scores tie), and that is the point -- it is the
    chance anchor against which every other baseline and the published number are
    measured. Its average precision is the positive rate, which is the correct chance
    anchor for AP.
    """

    name = "prevalence"

    def _fit(self, train):
        self.prior = float(train[C_LABEL].mean())

    def _score(self, rows):
        return np.full(len(rows), self.prior, dtype=float)

    def describe(self):
        d = super().describe()
        d["train_prevalence"] = self.prior
        return d


def positional_scores(train_rows: pd.DataFrame, test_rows: pd.DataFrame,
                      n_bins: int = DEFAULT_BINS, pos_col: str = "relpos",
                      label_col: str = "label") -> np.ndarray:
    """
    P(label = 1 | relative slice position), binned on equal-width bins of [0, 1],
    fitted on TRAINING rows and applied to TEST rows. Empty bins fall back to the
    training prevalence.

    This model has no access to pixels, to k-space, to phase, or to anything else about
    the patient. It knows only where a slice sits in its own stack. It is the
    generalised form of s12_rempe.positional_scores (which now calls through to here);
    the only additions are configurable label and position column names.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tr_bin = np.clip(np.digitize(np.asarray(train_rows[pos_col], dtype=float), edges) - 1,
                     0, n_bins - 1)
    y = np.asarray(train_rows[label_col], dtype=float)
    prior = float(y.mean())
    rate = np.array([y[tr_bin == b].mean() if (tr_bin == b).any() else prior
                     for b in range(n_bins)])
    te_bin = np.clip(np.digitize(np.asarray(test_rows[pos_col], dtype=float), edges) - 1,
                     0, n_bins - 1)
    return rate[te_bin]


class PositionalBaseline(Baseline):
    """Baseline 1. Wraps positional_scores in the fit/score contract."""

    def __init__(self, n_bins: int = DEFAULT_BINS):
        super().__init__()
        self.n_bins = int(n_bins)
        self.name = f"positional_{self.n_bins}bin"

    def _fit(self, train):
        self._train = train[[C_RELPOS, C_LABEL]].copy()

    def _score(self, rows):
        return positional_scores(self._train, rows, n_bins=self.n_bins,
                                 pos_col=C_RELPOS, label_col=C_LABEL)

    def describe(self):
        d = super().describe()
        d["n_bins"] = self.n_bins
        return d


class ColumnBaseline(Baseline):
    """
    Baseline for ONE column: a smoothed target encoding fitted on the training rows.

    Categorical -> P(label | level). Numeric -> P(label | quantile bin), or per exact
    value when the column takes few values (n_coils, matrix size). Levels unseen in
    training fall back to the training prevalence.

    The encoding is DIRECTIONAL: it maps to a fitted probability, so a test AUROC below
    0.5 means the association reversed out of sample and is reported as such. This is
    deliberately different from s08_belowchance.assoc_with_label, which folded the
    statistic to >= 0.5 because it fed arbitrary factorize codes to the AUC and had no
    fitted direction to preserve.
    """

    def __init__(self, col: str, n_bins: int = 10, smoothing: float = 1.0,
                 max_levels: int = 50):
        super().__init__()
        self.col = col
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)
        self.max_levels = int(max_levels)
        self.name = f"column[{col}]"

    def _keys(self, s: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            if self.edges is None:
                return np.where(np.isnan(v), "__nan__", v.astype(str)).astype(object)
            k = np.digitize(v, self.edges)
            out = k.astype(object)
            out[np.isnan(v)] = "__nan__"
            return out
        return s.astype(str).fillna("__nan__").to_numpy(dtype=object)

    def _fit(self, train):
        s = train[self.col]
        y = train[C_LABEL].to_numpy(dtype=float)
        self.prior = float(y.mean())
        self.kind = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        self.edges = None
        if self.kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            finite = v[np.isfinite(v)]
            if len(np.unique(finite)) > self.n_bins:
                qs = np.quantile(finite, np.linspace(0, 1, self.n_bins + 1)[1:-1])
                self.edges = np.unique(qs)
                if len(self.edges) == 0:
                    self.edges = None
        keys = self._keys(s)
        self.rates = {}
        df = pd.DataFrame({"k": keys, "y": y})
        for k, g in df.groupby("k", sort=False):
            self.rates[k] = float((g["y"].sum() + self.smoothing * self.prior)
                                  / (len(g) + self.smoothing))
        if self.kind == "categorical" and len(self.rates) > self.max_levels:
            keep = set(df["k"].value_counts().index[:self.max_levels])
            self.rates = {k: v for k, v in self.rates.items() if k in keep}
        self.n_levels_fitted = len(self.rates)

    def _score(self, rows):
        keys = self._keys(rows[self.col])
        return np.array([self.rates.get(k, self.prior) for k in keys], dtype=float)

    def describe(self):
        d = super().describe()
        d.update({"column": self.col, "kind": getattr(self, "kind", None),
                  "n_levels_fitted": getattr(self, "n_levels_fitted", 0)})
        return d


# --- shallow CART -----------------------------------------------------------
# Implemented here in numpy rather than pulled from scikit-learn so the tool has no
# dependency beyond numpy + pandas: an auditor should be able to run it inside a data
# enclave with no package installs. The self-test checks it against scikit-learn when
# scikit-learn happens to be importable.

def _fit_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_leaf: int,
              smoothing: float = 1.0, max_thresholds: int = 64) -> dict:
    prior = float(y.mean())

    def leaf(idx):
        c = len(idx)
        s = float(y[idx].sum())
        return {"leaf": (s + smoothing * prior) / (c + smoothing), "n": int(c)}

    def build(idx, depth):
        yi = y[idx]
        if (depth >= max_depth or len(idx) < 2 * min_leaf
                or yi.min() == yi.max()):
            return leaf(idx)
        best = None  # (impurity, feature, threshold)
        m = len(idx)
        for j in range(X.shape[1]):
            xs = X[idx, j]
            order = np.argsort(xs, kind="mergesort")
            xs_s, ys_s = xs[order], yi[order]
            uniq = np.unique(xs_s)
            if len(uniq) < 2:
                continue
            if len(uniq) > max_thresholds + 1:
                cand = np.unique(np.quantile(xs_s,
                                             np.linspace(0, 1, max_thresholds + 2)[1:-1]))
            else:
                cand = (uniq[:-1] + uniq[1:]) / 2.0
            pos = np.searchsorted(xs_s, cand, side="right")
            ok = (pos >= min_leaf) & (m - pos >= min_leaf)
            if not ok.any():
                continue
            pos = pos[ok]
            cand = cand[ok]
            csum = np.cumsum(ys_s)
            left_pos = csum[pos - 1]
            tot_pos = csum[-1]
            right_pos = tot_pos - left_pos
            pl = left_pos / pos
            pr_ = right_pos / (m - pos)
            imp = (pos * 2 * pl * (1 - pl) + (m - pos) * 2 * pr_ * (1 - pr_)) / m
            b = int(np.argmin(imp))
            if best is None or imp[b] < best[0] - 1e-12:
                best = (float(imp[b]), j, float(cand[b]))
        if best is None:
            return leaf(idx)
        _, j, thr = best
        left = idx[X[idx, j] <= thr]
        right = idx[X[idx, j] > thr]
        if len(left) < min_leaf or len(right) < min_leaf:
            return leaf(idx)
        return {"feature": j, "threshold": thr, "n": int(m),
                "left": build(left, depth + 1), "right": build(right, depth + 1)}

    return build(np.arange(len(y)), 0)


def _predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype=float)

    def walk(node, idx):
        if "leaf" in node:
            out[idx] = node["leaf"]
            return
        m = X[idx, node["feature"]] <= node["threshold"]
        walk(node["left"], idx[m])
        walk(node["right"], idx[~m])

    walk(tree, np.arange(len(X)))
    return out


def _tree_depth(node: dict) -> int:
    if "leaf" in node:
        return 0
    return 1 + max(_tree_depth(node["left"]), _tree_depth(node["right"]))


def _tree_features(node: dict, names: list[str], acc=None) -> list[str]:
    acc = [] if acc is None else acc
    if "leaf" not in node:
        acc.append(names[node["feature"]])
        _tree_features(node["left"], names, acc)
        _tree_features(node["right"], names, acc)
    return acc


class Encoder:
    """
    Numeric design matrix from mixed metadata columns, fitted on training rows.

    Numeric columns pass through (missing -> training median, plus a missing-indicator
    when training had any); categorical columns become one-hot over the training levels,
    capped at max_levels by training frequency with the tail folded into __other__.
    Levels that appear only at test time therefore land in __other__ instead of
    silently creating a column the model never saw.
    """

    def __init__(self, cols: list[str], max_levels: int = 30):
        self.cols = list(cols)
        self.max_levels = int(max_levels)
        self.spec: list[tuple] = []
        self.names: list[str] = []

    def fit(self, train: pd.DataFrame) -> "Encoder":
        self.spec, self.names = [], []
        for c in self.cols:
            s = train[c]
            if pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce")
                med = float(v.median()) if np.isfinite(v.median()) else 0.0
                has_na = bool(v.isna().any())
                self.spec.append(("num", c, med, has_na))
                self.names.append(f"{c}")
                if has_na:
                    self.names.append(f"{c}__missing")
            else:
                counts = s.astype(str).fillna("__nan__").value_counts()
                levels = list(counts.index[:self.max_levels])
                self.spec.append(("cat", c, levels, None))
                self.names.extend([f"{c}={lv}" for lv in levels])
                self.names.append(f"{c}=__other__")
        return self

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        blocks = []
        for kind, c, a, b in self.spec:
            if kind == "num":
                v = pd.to_numeric(rows[c], errors="coerce")
                blocks.append(v.fillna(a).to_numpy(dtype=float)[:, None])
                if b:
                    blocks.append(v.isna().to_numpy(dtype=float)[:, None])
            else:
                s = rows[c].astype(str).fillna("__nan__").to_numpy(dtype=object)
                oh = np.zeros((len(rows), len(a) + 1), dtype=float)
                lut = {lv: i for i, lv in enumerate(a)}
                for i, v in enumerate(s):
                    oh[i, lut.get(v, len(a))] = 1.0
                blocks.append(oh)
        if not blocks:
            return np.zeros((len(rows), 0), dtype=float)
        return np.hstack(blocks)


class TreeBaseline(Baseline):
    """
    Baselines 2, 3 and 5: a depth-limited CART on encoded metadata (and, for the
    combined arm, on relative position as an extra feature).

    Depth 3 with a minimum leaf size is deliberately weak. The claim being tested is
    "a trivial model reaches the published number", so the model has to stay trivial;
    a gradient-boosted ensemble would win more arguments and prove less.
    """

    def __init__(self, cols: list[str], name: str, use_relpos: bool = False,
                 max_depth: int = 3, min_leaf: int | None = None,
                 max_levels: int = 30):
        super().__init__()
        self.cols = list(cols)
        self.name = name
        self.use_relpos = bool(use_relpos)
        self.max_depth = int(max_depth)
        self.min_leaf = min_leaf
        self.max_levels = int(max_levels)

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.enc.transform(rows)
        if self.use_relpos:
            X = np.hstack([rows[C_RELPOS].to_numpy(dtype=float)[:, None], X])
        return X

    def _fit(self, train):
        self.enc = Encoder(self.cols, max_levels=self.max_levels).fit(train)
        self.feature_names = ((["relative_position"] if self.use_relpos else [])
                              + list(self.enc.names))
        X = self._matrix(train)
        y = train[C_LABEL].to_numpy(dtype=float)
        min_leaf = self.min_leaf or max(10, int(0.02 * len(train)))
        self.tree = _fit_tree(X, y, max_depth=self.max_depth, min_leaf=min_leaf)
        self.min_leaf_used = min_leaf

    def _score(self, rows):
        return _predict_tree(self.tree, self._matrix(rows))

    def describe(self):
        d = super().describe()
        d.update({
            "columns": self.cols, "uses_relative_position": self.use_relpos,
            "max_depth": self.max_depth, "depth_used": _tree_depth(self.tree),
            "min_leaf": self.min_leaf_used, "n_features": len(self.feature_names),
            "split_features": sorted(set(_tree_features(self.tree, self.feature_names))),
        })
        return d


# ==========================================================================
# Evaluation (statistics reused from s04_stats)
# ==========================================================================

TIE_DECIMALS = 12


def snap_ties(scores, decimals: int = TIE_DECIMALS) -> np.ndarray:
    """
    Round scores so that values which are equal in exact arithmetic compare equal.

    This is not cosmetic. Averaging a CONSTANT slice score over volumes of different
    depth does not return the constant: floating-point summation error makes the mean
    depend on how many terms were summed, at the 1e-17 level. Those differences are
    deterministic and depth-ordered, so a predictor that is genuinely constant acquires
    a perfect ordering by volume depth -- and where depth tracks the label, the CONSTANT
    predictor scores a patient-level AUROC of 0.000 or 1.000 instead of 0.5. We hit
    exactly that on the prostate DWI test arm (4 patients, all scores equal to
    0.04824120603015075, patient AUROC 0.000) before this was added.

    A null-model tool that reports 0.000 for the null model is worse than useless, so
    ties are snapped at 1e-12 -- far above the summation noise and far below any
    difference a real predictor produces.
    """
    return np.round(np.asarray(scores, dtype=float), decimals)


def patient_auc(labels, scores, subjects) -> float:
    """Mean-aggregated patient-level AUROC, with ties snapped before ranking."""
    agg = s04_stats.aggregate_by_cluster(np.asarray(labels, dtype=int),
                                         snap_ties(scores),
                                         np.asarray(subjects, dtype=object))
    return float(s04_stats.auc_midrank(agg["labels"], snap_ties(agg["scores"])))


def evaluate_scores(labels, scores, subjects, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    One set of scores read at both levels, with subject-clustered intervals.

    Slice level AND patient level, because the divergence between them is the whole
    finding. The naive (unclustered) slice interval is reported alongside so the report
    can show how much narrower the wrong interval would have been.
    """
    labels = np.asarray(labels, dtype=int)
    scores = snap_ties(scores)
    subjects = np.asarray(subjects, dtype=object)

    naive = s04_stats.naive_slice_bootstrap_auc(labels, scores, n_boot=n_boot, seed=seed)
    clustered = s04_stats.cluster_bootstrap_auc(labels, scores, subjects,
                                                n_boot=n_boot, seed=seed)
    agg = s04_stats.aggregate_by_cluster(labels, scores, subjects, how="mean")
    agg_max = s04_stats.aggregate_by_cluster(labels, scores, subjects, how="max")
    agg["scores"] = snap_ties(agg["scores"])
    agg_max["scores"] = snap_ties(agg_max["scores"])
    pat = s04_stats.cluster_bootstrap_auc(
        agg["labels"], agg["scores"], np.asarray(agg["cluster_ids"], dtype=object),
        n_boot=n_boot, seed=seed)
    return {
        "n_slices": int(len(labels)),
        "n_pos_slices": int(labels.sum()),
        "slice_prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "n_patients": int(agg["n_clusters"]),
        "n_pos_patients": int(agg["n_pos_clusters"]),
        "patient_prevalence": (float(agg["labels"].mean())
                               if agg["n_clusters"] else float("nan")),
        "slice_auc": float(s04_stats.auc_midrank(labels, scores)),
        "slice_ap": float(s04_stats.average_precision(labels, scores)),
        "slice_ci_clustered": [clustered.get("ci_lo"), clustered.get("ci_hi")],
        "slice_ci_naive": [naive.get("ci_lo"), naive.get("ci_hi")],
        "patient_auc": float(s04_stats.auc_midrank(agg["labels"], agg["scores"])),
        "patient_ci_clustered": [pat.get("ci_lo"), pat.get("ci_hi")],
        "patient_auc_maxagg": float(s04_stats.auc_midrank(agg_max["labels"],
                                                          agg_max["scores"])),
        "n_mixed_label_patients": int(agg["n_mixed_label_clusters"]),
        "ci_note": clustered.get("reason"),
    }


def trivial_fraction(baseline: float, chance: float, published: float,
                     baseline_ci=None, min_headroom: float = 0.02) -> dict:
    """
    The headline statistic: how much of a PUBLISHED number is reachable with no pixels.

        trivial_fraction = (baseline - chance) / (published - chance)

    'chance' is the value the CONSTANT (prevalence) predictor attains under the metric
    in question: exactly 0.5 for AUROC, because a predictor whose scores are all tied
    has mid-rank AUC 0.5 by arithmetic; the test positive rate for average precision.
    So the denominator is the headroom the published model actually had, and the
    numerator is the part of that headroom a pixel-blind model covers.

    The prevalence baseline is still RUN rather than assumed, and its measured value is
    reported next to this as a protocol check -- see prevalence_baseline_check in the
    payload. It is 0.5 within any single test set, but pooled out-of-fold across folds
    with different training prevalence it is not, and that deviation is itself a
    property of the evaluation protocol worth reporting.

    Read it as: "this published evaluation certifies X% of its own margin over chance
    to a model that never saw an image". Read TRIVIAL_FRACTION_LIMITS before writing a
    sentence about it -- in particular it is not evidence that the published model
    learned nothing.
    """
    out = {"baseline": float(baseline), "chance": float(chance),
           "published": float(published), "value": None, "value_clipped": None,
           "ci": None, "reason": None,
           "definition": "(baseline - chance) / (published - chance), chance = the "
                         "value of the constant/prevalence predictor under this metric"}
    denom = float(published) - float(chance)
    if not np.isfinite(denom) or denom <= min_headroom:
        out["reason"] = (f"published ({published:.3f}) is at or below chance "
                         f"({chance:.3f}); the fraction is undefined")
        return out
    v = (float(baseline) - float(chance)) / denom
    out["value"] = float(v)
    out["value_clipped"] = float(min(max(v, 0.0), 1.0))
    if baseline_ci and all(c is not None and np.isfinite(c) for c in baseline_ci):
        out["ci"] = [float((baseline_ci[0] - chance) / denom),
                     float((baseline_ci[1] - chance) / denom)]
        out["ci_note"] = ("propagates uncertainty in the baseline only; the published "
                          "number is treated as a fixed constant")
    return out


# ==========================================================================
# The audit
# ==========================================================================

def _build_baselines(colmap: ColumnMap, n_bins: int, tree_depth: int,
                     max_levels: int) -> list[Baseline]:
    b: list[Baseline] = [
        PrevalenceBaseline(),
        PositionalBaseline(n_bins=n_bins),
        ColumnBaseline(C_NSLICES),
    ]
    b[-1].name = "volume_size"
    if colmap.metadata:
        b.append(TreeBaseline(colmap.metadata, name="metadata_tree",
                              max_depth=tree_depth, max_levels=max_levels))
        b.append(TreeBaseline(colmap.metadata, name="combined_position_metadata",
                              use_relpos=True, max_depth=tree_depth,
                              max_levels=max_levels))
    else:
        b.append(TreeBaseline([], name="combined_position_metadata", use_relpos=True,
                              max_depth=tree_depth, max_levels=max_levels))
    return b


def _partition_scores(df: pd.DataFrame, parts, colmap: ColumnMap, n_bins: int,
                      tree_depth: int, max_levels: int):
    """
    Fit every baseline on each partition's TRAIN rows and score its TEST rows.

    `parts` is a list of (train_index, test_index). One element means the dataset's own
    split; k elements mean k-fold subject-level CV, whose scores are pooled into one
    out-of-fold vector. Both go through this identical code path, so nothing between an
    official-split result and a CV result can be attributed to a different estimator.

    Every partition is checked for subject overlap before anything is fitted. A model is
    never asked to score a row it was fitted on except for the explicitly labelled
    apparent-performance diagnostic.
    """
    sweep_bins = sorted(set(BIN_SWEEP) | {n_bins})
    uni_cols = list(colmap.metadata) + [C_NSLICES]

    pool: dict = {"labels": [], "subjects": [], "relpos": [],
                  "main": {}, "sweep": {}, "uni": {}}
    fits, apparent, per_part = [], {}, []

    for pi, (tr_idx, te_idx) in enumerate(parts):
        tr, te = df.loc[tr_idx], df.loc[te_idx]
        overlap = set(tr[C_SUBJ]) & set(te[C_SUBJ])
        if overlap:
            raise AssertionError(
                f"partition {pi}: {len(overlap)} subject(s) in both train and test "
                f"(e.g. {sorted(overlap)[:3]}); a subject-level split is mandatory")
        if te[C_LABEL].nunique() < 2 or tr[C_LABEL].nunique() < 2:
            per_part.append({"partition": pi, "skipped": "single-class train or test"})
            continue
        per_part.append({
            "partition": pi, "n_train_slices": int(len(tr)),
            "n_train_subjects": int(tr[C_SUBJ].nunique()),
            "n_test_slices": int(len(te)), "n_test_subjects": int(te[C_SUBJ].nunique()),
            "train_prevalence": float(tr[C_LABEL].mean()),
        })
        pool["labels"].append(te[C_LABEL].to_numpy())
        pool["subjects"].append(te[C_SUBJ].astype(str).to_numpy())
        pool["relpos"].append(te[C_RELPOS].to_numpy())

        part_fits = []
        for bl in _build_baselines(colmap, n_bins, tree_depth, max_levels):
            bl.fit(tr)
            pool["main"].setdefault(bl.name, []).append(bl.score(te))
            apparent.setdefault(bl.name, []).append(
                float(s04_stats.auc_midrank(tr[C_LABEL].to_numpy(), bl.score(tr))))
            part_fits.append(bl.describe())
        fits.append(part_fits)

        for nb in sweep_bins:
            pool["sweep"].setdefault(nb, []).append(
                PositionalBaseline(n_bins=nb).fit(tr).score(te))
        for c in uni_cols:
            cb = ColumnBaseline(c).fit(tr)
            pool["uni"].setdefault(c, {"scores": [], "kind": cb.kind,
                                       "levels": cb.n_levels_fitted})
            pool["uni"][c]["scores"].append(cb.score(te))

    if not pool["labels"]:
        return None
    return pool, fits, apparent, per_part


def _evaluate_partitions(df: pd.DataFrame, parts, colmap: ColumnMap, n_bins: int,
                         tree_depth: int, max_levels: int, n_boot: int, seed: int,
                         description: str) -> dict | None:
    """Score, pool, and turn into statistics. One dict per evaluation protocol."""
    got = _partition_scores(df, parts, colmap, n_bins, tree_depth, max_levels)
    if got is None:
        return None
    pool, fits, apparent, per_part = got
    y = np.concatenate(pool["labels"])
    subj = np.concatenate(pool["subjects"])
    rel = np.concatenate(pool["relpos"])

    out: dict = {
        "description": description,
        "n_partitions": len(parts),
        "subject_disjoint": True,
        "per_partition": per_part,
        "baselines": {},
        "fits": fits,
    }
    for bname, parts_scores in pool["main"].items():
        ev = evaluate_scores(y, np.concatenate(parts_scores), subj,
                             n_boot=n_boot, seed=seed)
        ap = apparent.get(bname, [])
        ev["apparent_slice_auc_on_train"] = float(np.mean(ap)) if ap else None
        out["baselines"][bname] = ev

    out["positional_bin_sweep"] = {}
    for nb, parts_scores in sorted(pool["sweep"].items()):
        sc = np.concatenate(parts_scores)
        out["positional_bin_sweep"][str(nb)] = {
            "slice_auc": float(s04_stats.auc_midrank(y, snap_ties(sc))),
            "patient_auc": patient_auc(y, sc, subj),
        }
    # A fit-free positional score: -(|relative position - 0.5|), i.e. "how close to the
    # middle of the stack". If this alone separates the classes, no estimator choice at
    # all is doing the work and the binning objection is closed.
    out["centrality_no_fit_slice_auc"] = float(
        s04_stats.auc_midrank(y, -np.abs(rel - 0.5)))

    uni = []
    for c, d in pool["uni"].items():
        sc = np.concatenate(d["scores"])
        uni.append({
            "column": "n_slices_per_volume" if c == C_NSLICES else c,
            "kind": d["kind"], "n_levels_fitted": d["levels"],
            "slice_auc": float(s04_stats.auc_midrank(y, snap_ties(sc))),
            "patient_auc": patient_auc(y, sc, subj),
            "subject_level_constant": bool(
                df.groupby(C_SUBJ)[c].nunique(dropna=False).max() == 1),
        })
    # A metadata column that separates the classes perfectly out of sample is far more
    # likely to BE the label under another name than to be an astonishing confound.
    # (Our own brain and knee cohorts are exactly this case: their label is defined
    # from an acquisition field, so that field reproduces it by construction.) Flag it
    # rather than let it walk into a headline.
    for r in uni:
        if np.isfinite(r["slice_auc"]) and r["slice_auc"] >= 0.999:
            r["tautology_suspect"] = (
                "separates the test set perfectly; check this column is not the label "
                "under another name before reporting it as a confound")
    uni.sort(key=lambda r: -(r["slice_auc"] if np.isfinite(r["slice_auc"]) else 0.0))
    out["metadata_univariate"] = uni
    out["tautology_suspects"] = [r["column"] for r in uni if "tautology_suspect" in r]
    if uni:
        out["best_single_column"] = dict(uni[0])
        # "One acquisition field beats the network" is the sentence this number exists
        # to support, so the number has to carry its own health warning: it is a
        # MAXIMUM over however many columns were screened, with no multiplicity
        # correction, and the maximum of K noisy statistics sits above their common
        # mean even when every one of them is null.
        out["best_single_column"]["selection_caveat"] = (
            f"maximum over {len(uni)} columns screened on this test set, with no "
            "multiplicity correction; optimistically biased. The pre-specified "
            "baselines above are the defensible headline.")
        out["best_single_column"]["n_columns_screened"] = len(uni)
    return out


# ==========================================================================
# Permutation calibration: where does each baseline sit when there is nothing
# to find?
# ==========================================================================
#
# A baseline's null is NOT automatically 0.5. Fit a metadata model on the training
# folds of a dataset whose label lives at SUBJECT level and score the held-out fold,
# and the per-level rate you fitted is anti-correlated with the per-level rate you
# are scoring: the positives are a finite population, so a level that was
# positive-rich in training is positive-poor in the fold that was left out. Pooled
# over folds the noise cancels but that bias does not, and the metadata baseline
# lands reliably BELOW 0.5. We measured -0.077 on a synthetic dataset whose label
# was, by construction, invisible to metadata.
#
# That is a property of the protocol, not of the data, and a tool that assumed 0.5
# would report a spurious sign on every metadata result. So each baseline is
# calibrated against a permutation null that destroys exactly the association being
# tested and preserves everything else:
#
#   within_volume_label_permutation   shuffles labels inside each volume. Kills the
#                                     position-label link; keeps prevalence, subject
#                                     clustering, depth and all metadata. The null
#                                     for the positional baseline.
#
#   metadata_block_permutation        gives each subject another subject's metadata
#                                     row, all columns moved together so their joint
#                                     structure survives. Kills the metadata-label
#                                     link; keeps labels, positions and clustering.
#                                     The null for the metadata and volume-size
#                                     baselines.
#
# The combined baseline is reported against both, because it can draw on either.

# Which nulls are even meaningful for which baseline. A baseline can legitimately have
# more than one candidate; the conservative one (the highest null) is used.
NULLS_FOR_BASELINE = {
    "volume_size": ("metadata_block_permutation",),
    "metadata_tree": ("metadata_block_permutation",),
    "combined_position_metadata": ("within_volume_label_permutation",
                                   "metadata_block_permutation"),
}


def _permute_labels_within_volume(df: pd.DataFrame, rng) -> pd.DataFrame:
    out = df.copy()
    lab = out[C_LABEL].to_numpy().copy()
    for _, idx in out.groupby(C_VOL).indices.items():
        if len(idx) > 1:
            lab[idx] = rng.permutation(lab[idx])
    out[C_LABEL] = lab
    return out


def _permute_metadata_across_subjects(df: pd.DataFrame, cols: list[str],
                                      rng) -> pd.DataFrame:
    """
    Hand each subject another subject's metadata BLOCK: every metadata column moves
    together, and the donor's rows are matched to the recipient's by slice order.

    Transplanting whole blocks rather than one summary value per subject is not a
    detail. A column that varies slice to slice (a per-slice TR, say) is a channel
    through which the model can overfit its training folds, and a null that quietly
    flattened it to a single value per subject would remove that channel and report a
    null that is too kind -- 0.505 instead of 0.425 on our synthetic image-driven
    dataset, which would have turned a null result into an apparent below-chance
    finding. Matching by slice order keeps the within-subject variation intact while
    destroying the subject-level association with the label. Where depths differ the
    donor's last row is repeated.
    """
    move = list(cols) + [C_NSLICES]
    out = df.copy()
    order = np.lexsort((out[C_SLICE].to_numpy(), out[C_SUBJ].to_numpy().astype(str)))
    subj_sorted = out[C_SUBJ].to_numpy().astype(str)[order]
    subs, starts = np.unique(subj_sorted, return_index=True)
    ends = np.append(starts[1:], len(order))
    blocks = {s: order[a:b] for s, a, b in zip(subs, starts, ends)}
    donor = rng.permutation(subs)

    rowmap = np.arange(len(out))
    for s, d in zip(subs, donor):
        ri, di = blocks[s], blocks[d]
        rowmap[ri] = di[np.minimum(np.arange(len(ri)), len(di) - 1)]
    for c in move:
        v = out[c].to_numpy()
        out[c] = v[rowmap]
    return out


def calibrate_nulls(df: pd.DataFrame, parts, colmap: ColumnMap, n_bins: int,
                    tree_depth: int, max_levels: int, n_perm: int,
                    seed: int) -> dict:
    """Point AUCs only -- a bootstrap inside a permutation loop buys nothing."""
    kinds = {
        "within_volume_label_permutation":
            lambda d, r: _permute_labels_within_volume(d, r),
        "metadata_block_permutation":
            lambda d, r: _permute_metadata_across_subjects(d, colmap.metadata, r),
    }
    # A permutation that cannot change anything is not a null. Shuffling labels inside
    # a volume does nothing when every volume is single-class -- which is the norm for
    # a subject-level diagnosis -- and reporting the resulting "null" (identical to the
    # observed value, p = 1.0) as if it were a calibration would be a lie of omission.
    n_mixed_volumes = int((df.groupby(C_VOL)[C_LABEL].nunique() > 1).sum())
    n_meta_blocks = int(df.groupby(C_SUBJ)[C_SUBJ].size().shape[0])
    preconditions = {
        "within_volume_label_permutation": {
            "applicable": n_mixed_volumes > 0,
            "reason": (None if n_mixed_volumes > 0 else
                       "every volume is single-class, so shuffling labels within a "
                       "volume is a no-op; this null carries no information here"),
            "n_mixed_label_volumes": n_mixed_volumes,
        },
        "metadata_block_permutation": {
            "applicable": bool(colmap.metadata) and n_meta_blocks > 1,
            "reason": (None if colmap.metadata else
                       "no metadata columns, so there is no metadata block to permute"),
            "n_subject_blocks": n_meta_blocks,
        },
    }
    out: dict = {"preconditions": preconditions}
    for kind, fn in kinds.items():
        if not preconditions[kind]["applicable"]:
            out[kind] = {}
            continue
        draws: dict[str, list] = {}
        for i in range(n_perm):
            rng = np.random.default_rng(seed + 9871 * (i + 1))
            dp = fn(df, rng)
            if dp[C_LABEL].nunique() < 2:
                continue
            got = _partition_scores(dp, parts, colmap, n_bins, tree_depth, max_levels)
            if got is None:
                continue
            pool = got[0]
            y = np.concatenate(pool["labels"])
            subj = np.concatenate(pool["subjects"])
            for bname, ps in pool["main"].items():
                sc = np.concatenate(ps)
                draws.setdefault(bname, []).append(
                    (float(s04_stats.auc_midrank(y, snap_ties(sc))),
                     patient_auc(y, sc, subj)))
        out[kind] = {
            b: {
                "n_perm": len(v),
                "slice_mean": float(np.mean([x[0] for x in v])),
                "slice_q025": float(np.quantile([x[0] for x in v], 0.025)),
                "slice_q975": float(np.quantile([x[0] for x in v], 0.975)),
                # Patient AUC is NaN whenever a permutation leaves the patient
                # level single-class, which is routine for a subject-level label.
                "patient_mean": (float(np.mean(_pat)) if
                                 (_pat := [x[1] for x in v if np.isfinite(x[1])])
                                 else None),
                "slice_draws": [float(x[0]) for x in v],
            }
            for b, v in draws.items() if v
        }
    return out


def attach_null_calibration(ev: dict, nulls: dict) -> None:
    """
    Fold the permutation nulls into each baseline's entry, with a p-value.

    The prevalence baseline is skipped: it IS the null, and calibrating it against
    itself would produce a meaningless p of 1.0 dressed up as a test. Where a baseline
    has more than one applicable null the highest is used, so the bar is the hardest
    one available rather than the most flattering.
    """
    pre = nulls.get("preconditions", {})
    ev["permutation_nulls"] = nulls
    for bname, b in ev["baselines"].items():
        if bname == "prevalence":
            b["null"] = {"kind": "not_applicable",
                         "note": "the prevalence baseline is itself the null model"}
            continue
        kinds = NULLS_FOR_BASELINE.get(
            bname, ("within_volume_label_permutation",) if bname.startswith("positional")
            else ("metadata_block_permutation",))
        cands = [(k, nulls.get(k, {}).get(bname)) for k in kinds]
        cands = [(k, nd) for k, nd in cands
                 if nd and pre.get(k, {}).get("applicable", True)]
        if not cands:
            b["null"] = {
                "kind": "unavailable",
                "note": "; ".join(
                    filter(None, [pre.get(k, {}).get("reason") for k in kinds]))
                or "no applicable permutation null for this baseline",
                "judge_against": 0.5,
            }
            continue
        kind, nd = max(cands, key=lambda kv: kv[1]["slice_mean"])
        draws = nd["slice_draws"]
        # One-sided permutation p with the observed value counted in its own reference
        # set: the form that cannot return an impossible zero.
        n_ge = sum(1 for d in draws if d >= b["slice_auc"])
        b["null"] = {
            "kind": kind,
            "n_perm": nd["n_perm"],
            "slice_mean": nd["slice_mean"],
            "slice_95pct_range": [nd["slice_q025"], nd["slice_q975"]],
            "patient_mean": nd["patient_mean"],
            "slice_excess_over_null": float(b["slice_auc"] - nd["slice_mean"]),
            "patient_excess_over_null": (
                None if nd["patient_mean"] is None
                else float(b["patient_auc"] - nd["patient_mean"])),
            "p_one_sided": float((n_ge + 1) / (nd["n_perm"] + 1)),
            "p_resolution": float(1.0 / (nd["n_perm"] + 1)),
            "exceeds_null": bool(b["slice_auc"] > nd["slice_q975"]),
        }
    for kind, block in nulls.items():
        if kind == "preconditions":
            continue
        for b in block.values():
            b.pop("slice_draws", None)


def audit(labels_path: Path, name: str | None = None, n_boot: int = 2000,
          seed: int = 0, n_bins: int = DEFAULT_BINS, cv_folds: int = 5,
          tree_depth: int = 3, max_levels: int = 30, published: float | None = None,
          published_metric: str = "slice_auc", published_label: str = "",
          val_as: str = "exclude", positive_if: str | None = None,
          n_perm: int = 20, relpos_col: str | None = None, **colargs) -> dict:
    """Run the whole family on one label table and return the payload.

    relpos_col: name of a column that ALREADY holds the slice's relative position in
    its stack. Some benchmarks publish that quantity directly (DeepLesion's
    `Normalized_lesion_location` z, LUNA16's world z in mm) and publish too few rows
    per volume for the harness to recover it by within-volume min/max rescaling. When
    it is given, the column is used verbatim as the positional feature, is rescaled to
    [0, 1] only if it falls outside that range, and is removed from the metadata pool
    so it cannot be counted twice.
    """
    labels_path = Path(labels_path)
    raw = load_table(labels_path)
    if relpos_col is not None:
        if relpos_col not in raw.columns:
            raise KeyError(f"--relpos-col {relpos_col!r} is not in the table")
        colargs = dict(colargs)
        colargs["exclude"] = tuple(colargs.get("exclude") or ()) + (relpos_col,)
    colmap = resolve_columns(raw, **colargs)
    if relpos_col is not None and relpos_col in colmap.metadata:
        colmap.metadata.remove(relpos_col)
        colmap.excluded[str(relpos_col)] = "supplied as the relative-position column"

    df = raw.copy()
    y, rule = binarise(df[colmap.label], positive_if)
    df[C_LABEL] = y
    df[C_SUBJ] = df[colmap.subject].astype(str)
    df[C_SLICE] = pd.to_numeric(df[colmap.slice], errors="coerce")
    n_bad = int(df[C_SLICE].isna().sum())
    if n_bad:
        df = df[df[C_SLICE].notna()].copy()
    df[C_VOL] = (df[C_SUBJ] if colmap.volume is None
                 else df[C_SUBJ] + "|" + df[colmap.volume].astype(str))
    if relpos_col is None:
        df = add_relative_position(df, C_VOL, C_SLICE)
        relpos_provenance = (f"computed within volume as (slice - min)/(max - min) on "
                             f"{colmap.slice!r}")
    else:
        r = pd.to_numeric(df[relpos_col], errors="coerce")
        n_bad_r = int(r.isna().sum())
        if n_bad_r:
            df = df[r.notna()].copy()
            r = r[r.notna()]
        lo, hi = float(r.min()), float(r.max())
        if lo < 0.0 or hi > 1.0:
            df[C_RELPOS] = (r - lo) / (hi - lo if hi > lo else 1.0)
            relpos_provenance = (f"supplied column {relpos_col!r}, linearly rescaled "
                                 f"from its observed range [{lo:.4g}, {hi:.4g}] to [0, 1]")
        else:
            df[C_RELPOS] = r
            relpos_provenance = (f"supplied column {relpos_col!r}, already in [0, 1] "
                                 f"(observed [{lo:.4g}, {hi:.4g}]); used verbatim")
        if n_bad_r:
            relpos_provenance += f"; {n_bad_r} rows dropped for a non-numeric value"
    df = add_volume_size(df, C_VOL)
    df = df.reset_index(drop=True)
    if df[C_LABEL].nunique() < 2:
        raise ValueError(f"label rule {rule!r} produces a single class; nothing to audit")

    payload: dict = {
        "tool": "s14_trivialbaselines",
        "version": "1.0",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": name or labels_path.stem,
        "labels_file": str(labels_path),
        "labels_sha256": _sha256(labels_path),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "columns": colmap.to_json(),
        "relative_position_provenance": relpos_provenance,
        "label_rule": rule,
        "n_rows": int(len(df)),
        "n_rows_dropped_bad_slice_index": n_bad,
        "n_subjects": int(df[C_SUBJ].nunique()),
        "n_volumes": int(df[C_VOL].nunique()),
        "slice_prevalence": float(df[C_LABEL].mean()),
        "patient_prevalence": float(df.groupby(C_SUBJ)[C_LABEL].max().mean()),
        "slices_per_volume": {
            "min": int(df.groupby(C_VOL).size().min()),
            "median": float(df.groupby(C_VOL).size().median()),
            "max": int(df.groupby(C_VOL).size().max()),
        },
        "settings": {"n_boot": n_boot, "seed": seed, "n_bins": n_bins,
                     "cv_folds": cv_folds, "tree_depth": tree_depth,
                     "max_levels": max_levels, "val_as": val_as,
                     "null_permutations": n_perm},
        "evaluations": {},
        "warnings": [],
    }

    # --- evaluation 1: the dataset's OWN split, if it published one ---------
    # Respecting the published split is what makes the comparison with a published
    # number legitimate. Only when there is no usable split do we construct one, and
    # then it is subject-level, never a random slice split.
    headline_key = None
    if colmap.split is not None:
        canon, info = normalise_split_values(df[colmap.split])
        payload["split_column_report"] = info
        if val_as == "train":
            canon = canon.replace("val", "train")
        elif val_as == "test":
            canon = canon.replace("val", "test")
        tr_idx = df.index[canon == "train"]
        te_idx = df.index[canon == "test"]
        if len(tr_idx) and len(te_idx):
            ev = _evaluate_partitions(
                df, [(tr_idx, te_idx)], colmap, n_bins, tree_depth, max_levels,
                n_boot, seed,
                description=(f"the dataset's own {colmap.split!r} column; validation "
                             f"rows: {val_as}"))
            if ev is not None:
                if n_perm:
                    attach_null_calibration(ev, calibrate_nulls(
                        df, [(tr_idx, te_idx)], colmap, n_bins, tree_depth,
                        max_levels, n_perm, seed))
                payload["evaluations"]["official_split"] = ev
                headline_key = "official_split"
                n_excl = int(((canon == "val") | (canon == "other")).sum())
                if n_excl:
                    payload["warnings"].append(
                        f"{n_excl} rows sit in neither arm (validation or unrecognised "
                        f"split value) and were dropped; change with --val-as")
        else:
            payload["warnings"].append(
                f"split column {colmap.split!r} does not yield both a train and a test "
                "arm; falling back to subject-level cross-validation")

    # --- evaluation 2: subject-level CV, pooled out of fold -----------------
    if cv_folds and cv_folds > 1 and df[C_SUBJ].nunique() >= cv_folds:
        assign = subject_folds(df.groupby(C_SUBJ)[C_LABEL].max(), cv_folds, seed)
        fold = df[C_SUBJ].map(assign)
        parts = [(df.index[fold != k], df.index[fold == k]) for k in range(cv_folds)]
        ev = _evaluate_partitions(
            df, parts, colmap, n_bins, tree_depth, max_levels, n_boot, seed,
            description=f"{cv_folds}-fold subject-level CV, pooled out-of-fold")
        if ev is not None:
            if n_perm:
                attach_null_calibration(ev, calibrate_nulls(
                    df, parts, colmap, n_bins, tree_depth, max_levels, n_perm, seed))
            payload["evaluations"]["subject_cv"] = ev
            if headline_key is None:
                headline_key = "subject_cv"

    if headline_key is None:
        raise RuntimeError("no evaluable split could be constructed for this table")
    payload["headline_evaluation"] = headline_key

    # --- the headline comparison against a published number ----------------
    ev = payload["evaluations"][headline_key]
    prev = ev["baselines"]["prevalence"]
    chance = {"slice_auc": 0.5, "patient_auc": 0.5,
              "slice_ap": prev.get("slice_prevalence", float("nan"))}
    ci_key = "slice_ci_clustered" if published_metric.startswith("slice") \
        else "patient_ci_clustered"
    best_name, best_val = None, -np.inf
    for bname, b in ev["baselines"].items():
        if bname == "prevalence":
            continue
        v = b.get(published_metric)
        if v is not None and np.isfinite(v) and v > best_val:
            best_name, best_val = bname, float(v)
    # Protocol check. A constant predictor has mid-rank AUROC of exactly 0.5 inside any
    # one test set. Pooled out-of-fold it need not: folds with a higher training
    # prevalence emit a higher constant, so the pooled ranking carries fold identity and
    # the "chance" model scores off 0.5 without seeing anything at all. When that shows
    # up it is a property of the pooling scheme, and it is reported rather than hidden.
    prev_check = {
        "measured_slice_auc": prev.get("slice_auc"),
        "measured_patient_auc": prev.get("patient_auc"),
        "expected": 0.5,
        "note": "a constant predictor is exactly 0.5 within one test set",
    }
    dev = abs(float(prev.get("slice_auc", 0.5)) - 0.5)
    if dev > 0.01:
        prev_check["deviation_flag"] = (
            f"the constant predictor scores {prev['slice_auc']:.3f}, {dev:.3f} away "
            "from 0.5. Pooling out-of-fold predictions across folds whose training "
            "prevalence differs makes fold identity rankable on its own. Treat this "
            "as the floor of what any pooled number here can mean.")
        payload["warnings"].append(prev_check["deviation_flag"])

    summary = {
        "metric": published_metric,
        "evaluation": headline_key,
        "chance": chance.get(published_metric),
        "chance_definition": ("0.5 for AUROC (constant predictor, mid-rank); the test "
                              "positive rate for average precision"),
        "prevalence_baseline_check": prev_check,
        "best_zero_image_baseline": best_name,
        "best_zero_image_value": None if best_name is None else best_val,
        "best_zero_image_ci": (None if best_name is None
                               else ev["baselines"][best_name].get(ci_key)),
    }
    if published is not None and best_name is not None:
        summary["published"] = float(published)
        summary["published_label"] = published_label
        summary["trivial_fraction"] = trivial_fraction(
            best_val, chance.get(published_metric, 0.5), float(published),
            baseline_ci=summary["best_zero_image_ci"])
        hi = (summary["best_zero_image_ci"] or [None, None])[1]
        summary["baseline_reaches_published"] = bool(
            hi is not None and hi >= float(published))
    summary["limits"] = TRIVIAL_FRACTION_LIMITS
    payload["headline"] = summary

    b0 = next(iter(ev["baselines"].values()))
    if b0["n_patients"] < 10 or min(b0["n_pos_patients"],
                                    b0["n_patients"] - b0["n_pos_patients"]) < 3:
        payload["warnings"].append(
            f"the headline test arm holds {b0['n_patients']} subjects "
            f"({b0['n_pos_patients']} positive). Patient-level AUROC on this few "
            "subjects is close to uninformative and its interval will be degenerate; "
            "read the slice-level row and the cross-validated evaluation instead.")
    if not colmap.metadata:
        payload["warnings"].append(
            "no admissible metadata columns in this label file; the metadata and "
            "combined baselines are position-only. That is a property of the file, "
            "not a null result about metadata confounding.")
    if ev.get("tautology_suspects"):
        payload["warnings"].append(
            "column(s) " + ", ".join(ev["tautology_suspects"]) + " separate the test "
            "set perfectly. Verify they are not the label under another name before "
            "reporting them as confounds.")
    payload["warnings"].append(
        "metadata columns are taken at face value as acquisition/administrative "
        "fields. Check payload['columns'] before claiming the baselines are "
        "image-blind: an image-derived column would break that guarantee.")
    return payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ==========================================================================
# Reporting
# ==========================================================================

def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _ci(pair, nd=3):
    if not pair or pair[0] is None or pair[1] is None:
        return "-"
    return f"[{_f(pair[0], nd)}, {_f(pair[1], nd)}]"


def render_card(p: dict) -> str:
    """One-page markdown card: the thing that goes in the supplement, one per dataset."""
    hk = p["headline_evaluation"]
    ev = p["evaluations"][hk]
    h = p["headline"]
    L = []
    A = L.append
    A(f"# Zero-image baseline card: {p['dataset']}")
    A("")
    A(f"*{p['tool']} v{p['version']} - {p['generated_utc']}*")
    A("")
    A("## Dataset")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| label file | `{Path(p['labels_file']).name}` (sha256 `{p['labels_sha256']}`) |")
    A(f"| rows (slices) | {p['n_rows']} |")
    A(f"| subjects | {p['n_subjects']} |")
    A(f"| volumes | {p['n_volumes']} |")
    A(f"| slices per volume | {p['slices_per_volume']['min']}-"
      f"{p['slices_per_volume']['max']} (median {_f(p['slices_per_volume']['median'], 1)}) |")
    A(f"| positive rule | `{p['label_rule']}` |")
    if p.get("relative_position_provenance"):
        A(f"| relative position | {p['relative_position_provenance']} |")
    A(f"| prevalence | {_f(p['slice_prevalence'])} slice / "
      f"{_f(p['patient_prevalence'])} patient |")
    A(f"| evaluation | {ev.get('description', hk)} |")
    A(f"| metadata columns used | {len(p['columns']['metadata'])} |")
    A("")
    A("## Baselines (fitted on train slices, applied to test slices, no pixels)")
    A("")
    A("| baseline | slice AUROC | slice 95% CI (subject-clustered) | patient AUROC | "
      "patient 95% CI | permutation null | excess over null |")
    A("|---|---|---|---|---|---|---|")
    for bname, b in ev["baselines"].items():
        nl = b.get("null") or {}
        A(f"| {bname} | {_f(b['slice_auc'])} | {_ci(b['slice_ci_clustered'])} | "
          f"{_f(b['patient_auc'])} | {_ci(b['patient_ci_clustered'])} | "
          f"{_f(nl.get('slice_mean'))} | {_f(nl.get('slice_excess_over_null'))} |")
    A("")
    A("`permutation null` is the value THIS baseline reaches when the association it "
      "tests is destroyed and everything else is held fixed. It is not always 0.5: "
      "out-of-fold metadata models on subject-level labels sit below chance by "
      "construction. Judge a baseline against its own null, not against 0.5.")
    A("")
    b0 = next(iter(ev["baselines"].values()))
    A(f"Test set: {b0['n_slices']} slices from {b0['n_patients']} subjects, "
      f"{b0['n_pos_slices']} positive slices / {b0['n_pos_patients']} positive subjects.")
    A("")
    if "positional_bin_sweep" in ev:
        A("## Bin sensitivity (positional baseline)")
        A("")
        A("| bins | " + " | ".join(ev["positional_bin_sweep"].keys()) + " | no-fit centrality |")
        A("|---|" + "---|" * (len(ev["positional_bin_sweep"]) + 1))
        A("| slice AUROC | "
          + " | ".join(_f(v["slice_auc"]) for v in ev["positional_bin_sweep"].values())
          + f" | {_f(ev.get('centrality_no_fit_slice_auc'))} |")
        A("| patient AUROC | "
          + " | ".join(_f(v["patient_auc"]) for v in ev["positional_bin_sweep"].values())
          + " | - |")
        A("")
        A("`no-fit centrality` is -(|relative position - 0.5|): no training data at all.")
        A("")
    if ev.get("metadata_univariate"):
        A("## Single metadata columns (each fitted on train alone)")
        A("")
        A(f"Screened {ev.get('best_single_column', {}).get('n_columns_screened', 0)} "
          "columns. The top row is a MAXIMUM over that many statistics with no "
          "multiplicity correction, so it is optimistically biased; the pre-specified "
          "baselines above are the defensible headline.")
        A("")
        A("| column | kind | slice AUROC | patient AUROC | constant within subject |")
        A("|---|---|---|---|---|")
        for r in ev["metadata_univariate"][:10]:
            flag = " **(tautology suspect)**" if "tautology_suspect" in r else ""
            A(f"| `{r['column']}`{flag} | {r['kind']} | {_f(r['slice_auc'])} | "
              f"{_f(r['patient_auc'])} | {'yes' if r['subject_level_constant'] else 'no'} |")
        A("")
    A("## Headline")
    A("")
    A(f"- chance anchor for {h['metric']}: **{_f(h['chance'])}** "
      f"({h['chance_definition']})")
    pc = h["prevalence_baseline_check"]
    A(f"- protocol check - the constant predictor actually scored "
      f"{_f(pc['measured_slice_auc'])} (slice) / {_f(pc['measured_patient_auc'])} "
      f"(patient)" + ("  **<- see warnings**" if "deviation_flag" in pc else ""))
    A(f"- best zero-image baseline: **{h['best_zero_image_baseline']}** = "
      f"**{_f(h['best_zero_image_value'])}** {_ci(h['best_zero_image_ci'])}")
    if "published" in h:
        A(f"- published number ({h.get('published_label') or 'user-supplied'}): "
          f"**{_f(h['published'])}**")
        tf = h["trivial_fraction"]
        if tf["value"] is None:
            A(f"- trivial fraction: not defined - {tf['reason']}")
        else:
            A(f"- **trivial fraction = {_f(tf['value'])}** {_ci(tf.get('ci'))}")
            A("")
            A(f"  > {int(round(100 * tf['value_clipped']))}% of this benchmark's margin "
              "over chance is reached by a model that never sees a pixel.")
        A(f"- baseline CI reaches the published number: "
          f"{'YES' if h.get('baseline_reaches_published') else 'no'}")
    else:
        A("- no published number supplied (`--published`); trivial fraction not computed")
    A("")
    A("## Interpretation and limits")
    A("")
    A("A high trivial fraction is a statement about the EVALUATION PROTOCOL, not about "
      "the published model's internals. It says a pixel-blind model reaches that much "
      "of the reported margin under the same protocol. It does not say the model "
      "learned nothing, and it must not be written that way.")
    A("")
    for lim in h["limits"]:
        A(f"- {lim}")
    if p.get("warnings"):
        A("")
        A("## Warnings")
        A("")
        for w in p["warnings"]:
            A(f"- {w}")
    A("")
    A("## Provenance")
    A("")
    A("```")
    A(p["command"])
    A("```")
    A("")
    A(f"Columns: subject=`{p['columns']['subject']}` slice=`{p['columns']['slice']}` "
      f"label=`{p['columns']['label']}` split=`{p['columns']['split']}` "
      f"volume=`{p['columns']['volume']}`")
    A("")
    A(f"Metadata used ({len(p['columns']['metadata'])}): "
      + (", ".join(f"`{c}`" for c in p["columns"]["metadata"]) or "none"))
    A("")
    return "\n".join(L)


def print_console(p: dict) -> None:
    hk = p["headline_evaluation"]
    ev = p["evaluations"][hk]
    h = p["headline"]
    w = 92
    print("=" * w)
    print(f"ZERO-IMAGE BASELINES  {p['dataset']}")
    print("=" * w)
    print(f"  {p['n_rows']} slices / {p['n_subjects']} subjects / {p['n_volumes']} volumes"
          f"   prevalence {p['slice_prevalence']:.3f} slice, "
          f"{p['patient_prevalence']:.3f} patient")
    print(f"  label rule: {p['label_rule']}")
    if p.get("relative_position_provenance"):
        print(f"  relative position: {p['relative_position_provenance']}")
    print(f"  evaluation: {ev.get('description', hk)}")
    print(f"  metadata columns: {len(p['columns']['metadata'])}"
          + (f"  ({', '.join(p['columns']['metadata'][:8])}"
             + (", ..." if len(p['columns']['metadata']) > 8 else "") + ")"
             if p['columns']['metadata'] else ""))
    print()
    print(f"  {'baseline':<30}{'slice AUC':>10}  {'slice 95% CI':<18}"
          f"{'pat AUC':>8}  {'patient 95% CI':<18}{'null':>7}{'excess':>8}")
    print("  " + "-" * (w - 4))
    for bname, b in ev["baselines"].items():
        nl = b.get("null") or {}
        print(f"  {bname:<30}{_f(b['slice_auc']):>10}  "
              f"{_ci(b['slice_ci_clustered']):<18}{_f(b['patient_auc']):>8}  "
              f"{_ci(b['patient_ci_clustered']):<18}"
              f"{_f(nl.get('slice_mean')):>7}"
              f"{_f(nl.get('slice_excess_over_null')):>8}")
    if any(b.get("null") for b in ev["baselines"].values()):
        print("  'null' is this baseline's own permutation null, which is NOT always "
              "0.5; 'excess' is observed - null.")
    if "positional_bin_sweep" in ev:
        print()
        print("  bin sweep (slice AUC): " + "  ".join(
            f"{k}={_f(v['slice_auc'])}" for k, v in ev["positional_bin_sweep"].items())
            + f"   no-fit centrality={_f(ev.get('centrality_no_fit_slice_auc'))}")
    if ev.get("metadata_univariate"):
        print()
        bs = ev.get("best_single_column", {})
        print(f"  best single metadata columns (slice AUC / patient AUC)"
              f"  [max over {bs.get('n_columns_screened', '?')} columns, "
              "not multiplicity-corrected]:")
        for r in ev["metadata_univariate"][:5]:
            print(f"      {r['column']:<28}{_f(r['slice_auc']):>8}  "
                  f"{_f(r['patient_auc']):>8}   ({r['kind']}, "
                  f"{r['n_levels_fitted']} levels)"
                  + ("   <<< TAUTOLOGY SUSPECT" if "tautology_suspect" in r else ""))
    print()
    print(f"  chance anchor          {_f(h['chance'])}   "
          f"(constant predictor measured at "
          f"{_f(h['prevalence_baseline_check']['measured_slice_auc'])})")
    print(f"  best zero-image        {_f(h['best_zero_image_value'])} "
          f"{_ci(h['best_zero_image_ci'])}  ({h['best_zero_image_baseline']})")
    if "published" in h:
        tf = h["trivial_fraction"]
        print(f"  published              {_f(h['published'])}"
              + (f"  ({h['published_label']})" if h.get("published_label") else ""))
        if tf["value"] is None:
            print(f"  TRIVIAL FRACTION       undefined -- {tf['reason']}")
        else:
            print(f"  TRIVIAL FRACTION       {_f(tf['value'])} {_ci(tf.get('ci'))}"
                  f"   -> {round(100 * tf['value_clipped'])}% of the margin over chance "
                  "needs no pixels")
    for wmsg in p.get("warnings", []):
        print(f"  ! {wmsg}")
    print()


# ==========================================================================
# Self-test
# ==========================================================================

def _synth(kind: str, n_subjects: int = 160, n_slices: int = 24, seed: int = 0):
    """
    Synthetic label tables where the right answer is known in advance.

    positional   label is a deterministic function of relative slice position only.
                 The positional baseline MUST recover it (slice AUROC ~ 1.0).
    random       label is coin flips per slice. Everything MUST sit at ~0.5.
    image_driven label is a per-subject property with no positional and no metadata
                 correlate -- the stand-in for a genuine imaging finding. Every
                 baseline MUST sit at ~0.5. This is the case that proves the tool does
                 not simply always fire.
    metadata     label is a deterministic function of one metadata field (a release
                 batch). The metadata baseline MUST recover it while the positional
                 baseline stays at ~0.5.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        depth = int(rng.integers(n_slices - 6, n_slices + 7))
        batch = f"batch{i % 4}"
        scanner = rng.choice(["A", "B", "C"])
        subj_pos = int(rng.integers(0, 2))
        for s in range(depth):
            rel = s / (depth - 1)
            if kind == "positional":
                lab = int(0.4 <= rel <= 0.6)
            elif kind == "random":
                lab = int(rng.random() < 0.3)
            elif kind == "image_driven":
                lab = subj_pos
            elif kind == "metadata":
                lab = int(batch in ("batch0", "batch1"))
            else:
                raise ValueError(kind)
            rows.append({"patient_id": f"p{i:04d}", "slice": s, "label": lab,
                         "batch": batch, "scanner": scanner,
                         "TR": float(rng.normal(4000, 200))})
    return pd.DataFrame(rows)


def self_test(n_boot: int = 400, quick: bool = False) -> int:
    failures = []

    def check(nm, cond, detail=""):
        if cond:
            print(f"  PASS  {nm}")
        else:
            print(f"  FAIL  {nm}   {detail}")
            failures.append(nm)

    print("s14_trivialbaselines self-test")
    print("-" * 78)

    # --- unit: relative position ------------------------------------------
    d = pd.DataFrame({"v": ["a"] * 5 + ["b"] * 3, "z": [10, 11, 12, 13, 14, 0, 5, 10]})
    r = add_relative_position(d, "v", "z")[C_RELPOS].to_numpy()
    check("relpos spans [0,1] per volume regardless of depth or origin",
          np.allclose(r[:5], [0, .25, .5, .75, 1]) and np.allclose(r[5:], [0, .5, 1]),
          str(r))
    one = add_relative_position(pd.DataFrame({"v": ["a"], "z": [7]}), "v", "z")
    check("single-slice volume yields a finite relpos",
          bool(np.isfinite(one[C_RELPOS]).all()), str(one[C_RELPOS].tolist()))

    # --- unit: volume size -------------------------------------------------
    vs = add_volume_size(d, "v")[C_NSLICES].to_numpy()
    check("volume size counts labelled slices per volume",
          np.allclose(vs, [5, 5, 5, 5, 5, 3, 3, 3]), str(vs))

    # --- unit: label binarisation -----------------------------------------
    y, rule = binarise(pd.Series([1, 2, 3, 4, 5], name="PIRADS"), ">2")
    check("binarise '>2' on PI-RADS", list(y) == [0, 0, 1, 1, 1] and ">" in rule, rule)
    y2, _ = binarise(pd.Series([0, 1, 0], name="l"), None)
    check("binarise passes {0,1} through", list(y2) == [0, 1, 0])
    try:
        binarise(pd.Series([1, 2, 3], name="PIRADS"), None)
        check("binarise refuses to guess a threshold", False)
    except ValueError:
        check("binarise refuses to guess a threshold", True)

    # --- unit: column detection -------------------------------------------
    fm = pd.DataFrame({"fastmri_pt_id": [1], "slice": [1], "PIRADS": [3],
                       "data_split": ["training"], "fastmri_rawfile": ["f.h5"],
                       "folder": ["b1"], "scanner": ["x"]})
    cm = resolve_columns(fm)
    check("auto-detects fastMRI-style column spellings",
          (cm.subject, cm.slice, cm.split, cm.volume)
          == ("fastmri_pt_id", "slice", "data_split", "fastmri_rawfile"), str(cm))
    check("PI-RADS is excluded from metadata as outcome-derived",
          "PIRADS" not in cm.metadata, str(cm.metadata))
    pic = pd.DataFrame({"patient_id": ["a"], "slice_idx": [2], "case_label": [1],
                        "manufacturer": ["S"]})
    cm2 = resolve_columns(pic, label="case_label")
    check("auto-detects PI-CAI-style spellings with an explicit label",
          (cm2.subject, cm2.slice, cm2.label) == ("patient_id", "slice_idx", "case_label"),
          str(cm2))
    cm3 = resolve_columns(fm, subject="folder")
    check("explicit override beats auto-detection", cm3.subject == "folder")

    # --- unit: split vocabulary -------------------------------------------
    can, info = normalise_split_values(pd.Series(["training", "TEST", "val", "weird"]))
    check("split vocabulary maps train/test/val and flags the rest",
          list(can) == ["train", "test", "val", "other"] and info["unrecognised"] == ["weird"],
          str(info))

    # --- unit: no leakage in the target encoder ---------------------------
    tr = pd.DataFrame({"c": ["a", "a", "b", "b"], C_LABEL: [1, 1, 0, 0]})
    te = pd.DataFrame({"c": ["a", "b", "zzz"]})
    cb = ColumnBaseline("c", smoothing=0.0).fit(tr)
    sc = cb.score(te)
    check("unseen level falls back to the training prior",
          abs(sc[2] - 0.5) < 1e-9 and sc[0] > sc[1], str(sc))
    check("score() before fit() raises",
          _raises(lambda: ColumnBaseline("c").score(te)))

    # --- unit: tree agrees with scikit-learn where available --------------
    rng = np.random.default_rng(1)
    Xt = rng.normal(size=(600, 4))
    yt = ((Xt[:, 0] > 0.3) ^ (Xt[:, 1] > -0.2)).astype(float)
    tree = _fit_tree(Xt, yt, max_depth=3, min_leaf=10, smoothing=0.0)
    ptr = _predict_tree(tree, Xt)
    mine = s04_stats.auc_midrank(yt.astype(int), ptr)
    try:
        from sklearn.tree import DecisionTreeClassifier
        sk = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                    random_state=0).fit(Xt, yt)
        theirs = s04_stats.auc_midrank(yt.astype(int), sk.predict_proba(Xt)[:, 1])
        check("built-in CART matches scikit-learn on an XOR problem",
              abs(mine - theirs) < 0.02, f"ours {mine:.4f} sklearn {theirs:.4f}")
    except ImportError:
        check("built-in CART separates an XOR problem", mine > 0.85, f"{mine:.4f}")

    # --- unit: the constant-predictor aggregation artefact ------------------
    # Averaging one constant over volumes of different depth does not return the
    # constant in floating point, and the error is depth-ordered. Without snapping,
    # the NULL MODEL scored a patient-level AUROC of 0.000 on the real prostate DWI
    # test arm. This check reproduces the exact configuration.
    depths = [28, 30, 31, 33]
    const = 0.04824120603015075
    sc_c = np.concatenate([np.full(k, const) for k in depths])
    cl_c = np.concatenate([[str(i)] * k for i, k in enumerate(depths)])
    lb_c = np.concatenate([[i % 2] * k for i, k in enumerate(depths)])
    raw_agg = s04_stats.aggregate_by_cluster(lb_c, sc_c, cl_c)
    check("the constant-score aggregation artefact is real (guard is needed)",
          len(set(raw_agg["scores"].tolist())) > 1,
          "means of a constant came out identical; guard may be redundant")
    check("snapping restores the constant predictor to exactly 0.5 at patient level",
          abs(patient_auc(lb_c, sc_c, cl_c) - 0.5) < 1e-12,
          _f(patient_auc(lb_c, sc_c, cl_c), 6))

    # --- unit: trivial fraction algebra -----------------------------------
    tf = trivial_fraction(0.851, 0.5, 0.861)
    check("trivial fraction algebra", abs(tf["value"] - 0.351 / 0.361) < 1e-9,
          str(tf["value"]))
    tf2 = trivial_fraction(0.6, 0.5, 0.51)
    check("trivial fraction is undefined when published is at chance",
          tf2["value"] is None, str(tf2))
    tf3 = trivial_fraction(0.9, 0.5, 0.8)
    check("trivial fraction above 1 is kept unclipped and flagged",
          tf3["value"] > 1 and tf3["value_clipped"] == 1.0, str(tf3["value"]))

    # --- unit: a slice-level split must be refused -------------------------
    dfx = pd.DataFrame({C_SUBJ: ["a", "a", "b", "b"], C_LABEL: [0, 1, 0, 1],
                        C_RELPOS: [0., 1., 0., 1.], C_NSLICES: [2.] * 4})
    check("train/test sharing a subject raises",
          _raises(lambda: _run_one_split(dfx, [0, 2], [1, 3],
                                         ColumnMap("s", "z", "l"), 5, 2, 10, 20, 0)))

    # --- end to end: the three scenarios that decide whether the tool works -
    # Seeds are LITERAL, not hash()-derived: Python randomises string hashing per
    # process, so a hash-seeded self-test is not reproducible across runs and a
    # borderline check would pass or fail depending on the interpreter's mood.
    # image_driven gets more subjects because its label lives at subject level, so the
    # effective sample size is the subject count, not the slice count.
    scenarios = {
        "positional": ("purely positional label", "positional", 11, 160),
        "random": ("purely random label", "random", 22, 160),
        "image_driven": ("subject-level (image-driven) label", "image_driven", 33, 400),
        "metadata": ("purely metadata-driven label", "metadata", 44, 160),
    }
    tmp = Path(__file__).resolve().parent.parent / "pipeline_out" / "_s14_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    nb = 150 if quick else n_boot
    results = {}
    for key, (desc, kind, sd, nsub) in scenarios.items():
        f = tmp / f"synthetic_{key}.csv"
        _synth(kind, n_subjects=nsub, seed=sd).to_csv(f, index=False)
        p = audit(f, name=f"synthetic_{key}", n_boot=nb, cv_folds=5, seed=0)
        results[key] = p
        ev = p["evaluations"][p["headline_evaluation"]]
        b = {k: v["slice_auc"] for k, v in ev["baselines"].items()}
        print(f"\n  [{desc}]  " + "  ".join(f"{k}={_f(v)}" for k, v in b.items()))

    ev = results["positional"]["evaluations"]["subject_cv"]["baselines"]
    check("positional label: positional baseline recovers it (slice AUROC >= 0.98)",
          ev[f"positional_{DEFAULT_BINS}bin"]["slice_auc"] >= 0.98,
          _f(ev[f"positional_{DEFAULT_BINS}bin"]["slice_auc"], 4))
    check("positional label: patient level is degenerate (every patient is positive)",
          results["positional"]["patient_prevalence"] == 1.0,
          _f(results["positional"]["patient_prevalence"]))

    ev = results["random"]["evaluations"]["subject_cv"]["baselines"]
    for bn, b in ev.items():
        check(f"random label: {bn} sits at chance (|AUROC-0.5| < 0.05)",
              abs(b["slice_auc"] - 0.5) < 0.05, _f(b["slice_auc"], 4))

    # THE CASE THAT MATTERS. A label that only the pixels could explain must leave
    # every zero-image baseline at chance -- otherwise the tool always fires and its
    # positive results mean nothing. The load-bearing assertion is the interval, not
    # the point estimate: with a subject-level label the effective n is the number of
    # subjects, so the point estimate has real sampling noise and demanding it sit
    # inside a hairline of 0.5 would be a check on luck rather than on the tool.
    ev = results["image_driven"]["evaluations"]["subject_cv"]["baselines"]
    for bn, b in ev.items():
        nl = b["null"]
        check(f"image-driven label: {bn} is nowhere near recovering the label",
              b["slice_auc"] < 0.60, _f(b["slice_auc"], 4))
        if nl.get("p_one_sided") is None:
            continue
        check(f"image-driven label: {bn} does not beat its own permutation null",
              not nl["exceeds_null"] and nl["p_one_sided"] > 0.05,
              f"AUROC {_f(b['slice_auc'], 4)} vs null {_f(nl['slice_mean'], 4)} "
              f"p={nl['p_one_sided']:.3f}")
    # A subject-level label makes the within-volume permutation a no-op. The tool must
    # say the null is unavailable rather than report the observed value back as its own
    # null with p = 1.0, which is what it did before preconditions were added.
    pre = results["image_driven"]["evaluations"]["subject_cv"]["permutation_nulls"][
        "preconditions"]
    check("image-driven label: the within-volume null is declared inapplicable",
          pre["within_volume_label_permutation"]["applicable"] is False
          and "no-op" in pre["within_volume_label_permutation"]["reason"],
          str(pre["within_volume_label_permutation"]))
    check("image-driven label: an inapplicable null is reported as unavailable, "
          "not faked",
          ev[f"positional_{DEFAULT_BINS}bin"]["null"]["kind"] == "unavailable",
          str(ev[f"positional_{DEFAULT_BINS}bin"]["null"]))
    check("the prevalence baseline is never calibrated against itself",
          ev["prevalence"]["null"]["kind"] == "not_applicable")
    hf = results["image_driven"]["headline"]["trivial_fraction"] if "trivial_fraction" \
        in results["image_driven"]["headline"] else None
    check("image-driven label: no published number was supplied, so no fraction claimed",
          hf is None)

    ev = results["metadata"]["evaluations"]["subject_cv"]["baselines"]
    check("metadata label: metadata tree recovers it (slice AUROC >= 0.98)",
          ev["metadata_tree"]["slice_auc"] >= 0.98, _f(ev["metadata_tree"]["slice_auc"], 4))
    check("metadata label: metadata tree beats its own null decisively",
          ev["metadata_tree"]["null"]["exceeds_null"]
          and ev["metadata_tree"]["null"]["p_one_sided"] <= 0.05,
          str(ev["metadata_tree"]["null"]))
    check("metadata label: positional baseline stays at chance",
          abs(ev[f"positional_{DEFAULT_BINS}bin"]["slice_auc"] - 0.5) < 0.05,
          _f(ev[f"positional_{DEFAULT_BINS}bin"]["slice_auc"], 4))
    pev = results["positional"]["evaluations"]["subject_cv"]["baselines"]
    check("positional label: positional baseline beats its own null decisively",
          pev[f"positional_{DEFAULT_BINS}bin"]["null"]["exceeds_null"]
          and pev[f"positional_{DEFAULT_BINS}bin"]["null"]["p_one_sided"] <= 0.05,
          str(pev[f"positional_{DEFAULT_BINS}bin"]["null"]))
    check("positional label: the positional null is near 0.5 as it should be",
          abs(pev[f"positional_{DEFAULT_BINS}bin"]["null"]["slice_mean"] - 0.5) < 0.05,
          _f(pev[f"positional_{DEFAULT_BINS}bin"]["null"]["slice_mean"], 4))
    best = results["metadata"]["evaluations"]["subject_cv"]["best_single_column"]
    check("metadata label: the best single column is named and is the batch field",
          best["column"] == "batch" and best["slice_auc"] >= 0.98,
          f"{best['column']} {_f(best['slice_auc'])}")
    worst_pos = results["image_driven"]["evaluations"]["subject_cv"]["best_single_column"]
    check("image-driven label: even the best single column stays near chance",
          abs(worst_pos["slice_auc"] - 0.5) < 0.10,
          f"{worst_pos['column']} {_f(worst_pos['slice_auc'])}")
    # The constant predictor is exactly 0.5 within one test set. Pooled out of fold it
    # need not be, and the tool has to say so rather than quietly anchor on it.
    one = _synth("random", n_subjects=40, seed=7)
    one[C_LABEL] = one["label"]
    pv = PrevalenceBaseline().fit(one.iloc[:500])
    sc = pv.score(one.iloc[500:])
    check("constant predictor is exactly 0.5 within a single test set",
          abs(s04_stats.auc_midrank(one[C_LABEL].to_numpy()[500:], sc) - 0.5) < 1e-12)
    devs = {k: r["evaluations"]["subject_cv"]["baselines"]["prevalence"]["slice_auc"]
            for k, r in results.items()}
    check("pooled out-of-fold constant predictor stays near 0.5",
          all(abs(v - 0.5) < 0.05 for v in devs.values()),
          str({k: round(v, 4) for k, v in devs.items()}))
    flagged = [k for k, r in results.items()
               if "deviation_flag" in r["headline"]["prevalence_baseline_check"]]
    check("any pooled deviation of the constant predictor is flagged in warnings",
          all(any("constant predictor scores" in w for w in results[k]["warnings"])
              for k in flagged), str(flagged))
    check("chance anchor for AUROC is 0.5 and not the measured pooled value",
          all(r["headline"]["chance"] == 0.5 for r in results.values()))

    # --- card renders and JSON round-trips ---------------------------------
    card = render_card(results["positional"])
    check("card renders with a headline section", "## Headline" in card and len(card) > 800)
    check("card states the protocol-not-model caveat",
          "EVALUATION PROTOCOL" in card)
    txt = json.dumps(results["random"], default=_json_default)
    check("payload is JSON-serialisable", len(txt) > 500 and json.loads(txt))

    print("-" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all self-test checks passed")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage:")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="run synthetic datasets with known answers and exit")
    p.add_argument("--quick", action="store_true", help="fewer bootstrap replicates")
    p.add_argument("--labels", help="label table (csv/tsv/parquet)")
    p.add_argument("--name", help="dataset name for the outputs")
    p.add_argument("--out-dir", default=None,
                   help="where to write <name>.json and <name>.md "
                        "(default pipeline_out/trivial_baselines)")
    p.add_argument("--no-write", action="store_true", help="console output only")

    g = p.add_argument_group("columns (all optional; auto-detected otherwise)")
    g.add_argument("--subject-col")
    g.add_argument("--slice-col")
    g.add_argument("--label-col")
    g.add_argument("--split-col", help="use '' to ignore an existing split column")
    g.add_argument("--volume-col", help="use '' to treat each subject as one volume")
    g.add_argument("--relpos-col",
                   help="column already holding relative position in the stack "
                        "(e.g. a published normalised z); used instead of deriving it")
    g.add_argument("--metadata-cols", help="comma-separated; overrides auto-selection")
    g.add_argument("--exclude-cols", default="", help="comma-separated, dropped from metadata")
    g.add_argument("--positive-if", help="e.g. '>2', '>=3', '==1', 'in:3,4,5'")
    g.add_argument("--allow-outcome-cols", action="store_true",
                   help="keep columns whose names look outcome-derived (tautological)")
    g.add_argument("--allow-image-derived-cols", action="store_true",
                   help="keep columns whose names look image-derived (breaks the "
                        "zero-image guarantee -- only for diagnostics)")

    g = p.add_argument_group("protocol")
    g.add_argument("--val-as", default="exclude", choices=("exclude", "train", "test"),
                   help="what to do with rows marked validation (default exclude)")
    g.add_argument("--cv-folds", type=int, default=5,
                   help="subject-level CV folds; 0 disables (default 5)")
    g.add_argument("--bins", type=int, default=DEFAULT_BINS)
    g.add_argument("--tree-depth", type=int, default=3)
    g.add_argument("--max-levels", type=int, default=30)
    g.add_argument("--n-boot", type=int, default=2000)
    g.add_argument("--null-permutations", type=int, default=20,
                   help="permutation draws used to calibrate each baseline's null; "
                        "0 disables (default 20). A baseline's null is not always 0.5 "
                        "-- see the module docstring.")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("comparison with a published number")
    g.add_argument("--published", type=float, help="the published performance number")
    g.add_argument("--published-metric", default="slice_auc",
                   choices=("slice_auc", "patient_auc", "slice_ap"))
    g.add_argument("--published-label", default="", help="citation for the number")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.self_test:
        return self_test(n_boot=150 if a.quick else 400, quick=a.quick)
    if not a.labels:
        print("nothing to do: pass --labels FILE or --self-test", file=sys.stderr)
        return 2

    payload = audit(
        Path(a.labels), name=a.name, n_boot=200 if a.quick else a.n_boot, seed=a.seed,
        n_bins=a.bins, cv_folds=a.cv_folds, tree_depth=a.tree_depth,
        max_levels=a.max_levels, n_perm=a.null_permutations, published=a.published,
        published_metric=a.published_metric, published_label=a.published_label,
        val_as=a.val_as, relpos_col=a.relpos_col,
        subject=a.subject_col, slice_col=a.slice_col, label=a.label_col,
        split=a.split_col, volume=a.volume_col,
        metadata=[c for c in (a.metadata_cols or "").split(",") if c.strip()] or None,
        exclude=tuple(c.strip() for c in a.exclude_cols.split(",") if c.strip()),
        allow_outcome=a.allow_outcome_cols,
        allow_image_derived=a.allow_image_derived_cols,
        positive_if=a.positive_if,
    )
    print_console(payload)
    if not a.no_write:
        out = Path(a.out_dir) if a.out_dir else (
            Path(__file__).resolve().parent.parent / "pipeline_out" / "trivial_baselines")
        out.mkdir(parents=True, exist_ok=True)
        stem = payload["dataset"]
        (out / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, default=_json_default))
        (out / f"{stem}.md").write_text(render_card(payload))
        print(f"  wrote {out / (stem + '.json')}")
        print(f"  wrote {out / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
