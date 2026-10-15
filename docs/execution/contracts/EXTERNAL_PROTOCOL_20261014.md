# Outcome-blind external protocol amendment — Codex decision, October 14, 2026

This amendment is made before any project access to CQ500 images or individual outcome rows. It changes a planned comparison for a documented representation limitation; it does not retroactively preregister earlier work or claim the external experiment ran. Historical design and batches remain unchanged. No acquisition or clinical inference is authorized by this document.

## Comparator and primary question — decided

The original-counter geometry comparator is not demonstrably reproducible externally. Retain that failed applicability finding. Authorize the proposed **transported physical-rank geometry comparator with frozen reconstructed G2_tree parameters**, explicitly named `G2_tree_transported_rank_reconstructed_frozen`, for the planned external comparison. Do not call this original-G-star validation. No fitting or recalibration is allowed.

Primary: examination-weighted AUROC of image seed42 maximum-slice logit minus AUROC of this transported comparator, on the identical eligible paired set, with both absolute AUROCs and positive/negative counts. Its interpretation is image-versus-transported-geometry discrimination under a fixed deployment rule, not causal shortcut detection or clinical benefit. Seed123 remains descriptive on its own paired set, with its set overlap and counts explicit; never promote it on results. Freeze model bytes, canonical float16 precision, aggregation and adapted representation before inference.

## Candidate population and series selection — technical contract decided, clinical mapping open

The intended source roster precedes extraction or successful parsing. Each candidate remains present through source availability, header interpretation, clinical compatibility, score availability and reference-label availability. Ambiguous source-exam/StudyUID assignment is not silently resolved by selecting the first UID. All incomplete and unknown reasons remain visible.

Do not use the development feature minima/maxima as external inclusion rules: they are support flags. Technical failures such as invalid orientation, uninterpretable geometry, duplicate conflicts or unsupported decoding remain explicit. Source or clinical eligibility must be distinguished from model scoring failures and support flags. Missing images/indeterminate eligibility are reported in full-source flow; completion bounds apply only to the eligible labeled population actually defined, not to unknown eligibility or labels.

The clinical series requirement is a noncontrast axial soft-reconstruction series covering the brain. Source eligibility guarantees at least one qualifying series, not that every series qualifies. Use a separate, outcome-blind `series_selection.csv` with source exam, StudyUID, SeriesUID, eligibility and reason, evidence/source, selector and review status. Clear, source-authenticated rules may populate it; ambiguous kernel names, contrast status or coverage require qualified blinded review. A reviewer must not see model scores or hemorrhage labels. Missing contrast tag alone is not proof of noncontrast; kernel names require documented vendor/source mapping. Unknown eligibility is not false, and must remain a separate status.

Once clinically qualifying series are identified, select one by largest superior-inferior physical coverage, then smallest absolute difference of median physical gap from5mm, then lexical SeriesUID. This is a fixed engineering tie-break for the transported single-series pipeline, not a claim of clinical optimality. No new resampling is introduced; historical image preprocessing remains fixed. The tie-break must not run until clinical qualifying status is resolved. Freeze this rule before outcomes and report its changed unit relative to the historical all-included-series pipeline.

## Reference and grouping — decided with explicit dependencies

Use the direct per-reader any-ICH finding and majority of the three readers, including chronic hemorrhage. Two agreeing valid reads determine a majority; otherwise preserve missing/indeterminate reference. This missing-read policy is our explicit analysis rule, not a claim that the source used it. Authenticate column names and source-exam key mapping before opening individual values; do not infer schema from an unrelated mirror. A complete standard three-reader schema must not silently accept arbitrary duplicate or extra reader columns.

Verified person linkage, if source-supported, governs resampling and retains all examinations within each sampled person. If unavailable, authorize examination-cluster fallback with unresolved repeat dependence prominently disclosed. Do not call the fallback a person-level95% interval or imply nominal coverage under unknown dependence. A sidecar flag alone cannot authenticate linkage; bind the actual mapping and its source evidence. Do not infer patients from unvalidated identifier hashes.

Primary sampling recipe remains2000 paired draws, seed2026101327, linear2.5/97.5 percentiles, with all undefined draws counted and never redrawn. The clinical execution contract must enforce it. Point-only synthetic tests use no RNG. Keep deterministic missing-prediction completion bounds distinct from sampling intervals; retain both the eligible labeled denominator and paired subset. Restore seed123 descriptive outputs without additional bootstrap intervals. If grouping or reference assumptions remain unresolved, label the conditional scope explicitly rather than hiding it.

The planned pooled target is the released enriched case mix. Preserve B1/B2 source membership if authenticated, and show counts/flow by collection batch. Do not add new stratum performance endpoints, prevalence estimates for routine practice, threshold claims or hypothesis tests in this amendment.

## Primary-source facts informing the decision

CQ500's paper describes a consecutive collection batch plus an enriched batch, allows follow-up scans, and requires a compatible axial noncontrast soft-kernel series. Its reference uses independent radiologist assessments, majority decisions and positive chronic hemorrhage. These support the population, series and grouping cautions above; they do not authenticate released column names or our engineering tie-break. [Primary methods](https://arxiv.org/html/1803.05854).

The curator's own account states that the dataset and reads are public under CC-BY-NC-SA. The Academic Torrents record lists CC-BY-NC-SA4.0 and28.66GB, with empty additional-terms metadata. A separate EULA is unverified, not an established requirement. Do not import unrelated Qure commercial-service terms into this dataset. [Curator release statement](https://www.qure.ai/us/blog/challenges-of-development-validation-of-deep-learning-for-radiology), [release record](https://academictorrents.com/details/47e9d8aab761e75fd0a81982fa62bddf3a173831).

## Decision boundary

Geometry adaptation, conditional grouping fallback, endpoint family and engineering tie-break are now adjudicated for implementation. Clinical ambiguous-series classification, exact label schema and source mapping, institutional determination and actual acquisition remain unresolved. Existing storage approval is retained. Prepare a concrete source-identified, bounded acquisition plan before asking about the transport/current-scope decision. No request to accept unknown terms and no repeated storage permission is justified.
