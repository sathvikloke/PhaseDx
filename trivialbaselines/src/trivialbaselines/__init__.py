"""
trivialbaselines -- zero-image null models for slice-labelled imaging benchmarks.

Fit a family of models that never see a pixel -- slice position, acquisition metadata,
volume depth, the constant predictor -- on a benchmark's own published label file, and
measure how much of the benchmark's headline number they reach.

    from trivialbaselines import audit, render_card
    payload = audit("labels.csv", name="mybench", published=0.861)
    print(render_card(payload))

or from a shell::

    trivial-baselines --labels labels.csv --name mybench --published 0.861

What a result licenses you to say is narrow and is spelled out in
``TRIVIAL_FRACTION_LIMITS`` and in the "Interpretation" block of every generated card:
a high trivial fraction is a statement about an EVALUATION PROTOCOL, never about what a
published model did or did not learn.
"""

from .core import (
    TRIVIAL_FRACTION_LIMITS,
    Baseline,
    ColumnBaseline,
    PositionalBaseline,
    PrevalenceBaseline,
    TreeBaseline,
    audit,
    evaluate_scores,
    load_table,
    main,
    patient_auc,
    positional_scores,
    print_console,
    render_card,
    resolve_columns,
    self_test,
    trivial_fraction,
)
from .stratified import position_strata, stratified_auc

__version__ = "1.0.0"

__all__ = [
    "TRIVIAL_FRACTION_LIMITS",
    "Baseline",
    "ColumnBaseline",
    "PositionalBaseline",
    "PrevalenceBaseline",
    "TreeBaseline",
    "audit",
    "evaluate_scores",
    "load_table",
    "main",
    "patient_auc",
    "position_strata",
    "positional_scores",
    "print_console",
    "render_card",
    "resolve_columns",
    "self_test",
    "stratified_auc",
    "trivial_fraction",
    "__version__",
]
