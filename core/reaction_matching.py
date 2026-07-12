"""Pairwise reaction comparison (port of Java ReactionComparator).

Both sides are enriched ReactionRecord-shaped entries. Metrics treat
``baseline`` as ground truth and ``candidate`` as the system under test:

- FP = unmatched candidate reactions
- FN = unmatched baseline reactions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core.compound_matching import canonicalize_name
from core.reaction_parsing import ReactionEntry

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised only when rdkit missing
    Chem = None  # type: ignore[assignment]

# --- HITL-amended axis weights (INJ-149) ---------------------------------
W_NAME = 0.40
W_SMILES = 0.15
W_REACTANTS = 0.15
W_PROCEDURE = 0.20
W_YIELD = 0.05
W_CONDITIONS = 0.05

TP_THRESHOLD = 0.55

SYNONYM_NEAR_MISS_LOW = 0.40
SYNONYM_NEAR_MISS_HIGH = 0.70

_ATOM_MAP_PATTERN = re.compile(r":\d+]")

MatchType = Literal["CONTENT_MATCH", "FALSE_POSITIVE", "FALSE_NEGATIVE"]


@dataclass(frozen=True)
class AxisScores:
    product_name: float | None
    product_smiles: float | None
    reactant_jaccard: float | None
    procedure_cosine: float | None
    yield_score: float | None
    conditions: float | None


@dataclass(frozen=True)
class AxisCoverage:
    product_name: float
    product_smiles: float
    reactant_jaccard: float
    procedure_cosine: float
    yield_score: float
    conditions: float


@dataclass(frozen=True)
class ReactionMatchDetail:
    match_type: MatchType
    baseline_index: int | None
    candidate_index: int | None
    baseline: ReactionEntry | None
    candidate: ReactionEntry | None
    composite_score: float | None = None
    axis_scores: AxisScores | None = None
    label_match: bool | None = None
    product_smiles_match: bool | None = None
    reactant_jaccard: float | None = None
    procedure_cosine_similarity: float | None = None
    yield_difference: float | None = None
    reaction_class_match: bool | None = None


@dataclass
class ReactionBenchmarkReport:
    baseline_label: str
    candidate_label: str
    baseline_reaction_count: int
    candidate_reaction_count: int
    non_synthetic_skipped_baseline: int
    non_synthetic_skipped_candidate: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    synonym_near_miss: int
    avg_reactant_jaccard: float | None = None
    perfect_reactant_match_pct: float | None = None
    yield_mae: float | None = None
    yield_within_5ppt_pct: float | None = None
    reaction_class_accuracy: float | None = None
    avg_procedure_similarity: float | None = None
    legacy_label_precision: float | None = None
    legacy_label_recall: float | None = None
    axis_coverage: AxisCoverage | None = None
    match_details: list[ReactionMatchDetail] = field(default_factory=list)


@dataclass
class _PairScore:
    baseline_index: int
    candidate_index: int
    axis_name: float | None = None
    axis_smiles: float | None = None
    axis_reactants: float | None = None
    axis_procedure: float | None = None
    axis_yield: float | None = None
    axis_conditions: float | None = None
    composite: float | None = None
    label_match: bool = False
    product_smiles_match: bool = False


def levenshtein_distance(a: str, b: str) -> int:
    """Iterative Levenshtein distance (port of BenchmarkTextNormalizer)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    n = len(a)
    m = len(b)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def levenshtein_ratio(canon_a: str | None, canon_b: str | None) -> float | None:
    """1 - editDistance / max(len(a), len(b)) on already-canonical strings."""
    if canon_a is None or canon_b is None:
        return None
    max_len = max(len(canon_a), len(canon_b))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(canon_a, canon_b) / max_len)


def strip_atom_maps(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    return _ATOM_MAP_PATTERN.sub("]", smiles)


def canonicalize_smiles(smiles: str | None) -> str | None:
    """Canonicalize SMILES via RDKit; return None on missing/invalid input."""
    if smiles is None or not str(smiles).strip():
        return None
    if Chem is None:
        raise ImportError(
            "rdkit is required for reaction SMILES comparison. "
            "Install it with: pip install rdkit"
        )
    cleaned = strip_atom_maps(str(smiles).strip())
    if cleaned is None or not cleaned:
        return None
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compose_weighted_sum(
    axis_name: float | None,
    axis_smiles: float | None,
    axis_reactants: float | None,
    axis_procedure: float | None,
    axis_yield: float | None,
    axis_conditions: float | None,
) -> float | None:
    """Skip-aware weighted-sum composite; null axes are dropped and weights renormalized."""
    num = 0.0
    den = 0.0
    if axis_name is not None:
        num += W_NAME * axis_name
        den += W_NAME
    if axis_smiles is not None:
        num += W_SMILES * axis_smiles
        den += W_SMILES
    if axis_reactants is not None:
        num += W_REACTANTS * axis_reactants
        den += W_REACTANTS
    if axis_procedure is not None:
        num += W_PROCEDURE * axis_procedure
        den += W_PROCEDURE
    if axis_yield is not None:
        num += W_YIELD * axis_yield
        den += W_YIELD
    if axis_conditions is not None:
        num += W_CONDITIONS * axis_conditions
        den += W_CONDITIONS
    if den == 0.0:
        return None
    return num / den


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return 0.0 if union == 0 else intersection / union


def cosine_similarity(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def precision(tp: int, fp: int) -> float:
    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)


def f1_score(prec: float, rec: float) -> float:
    return 0.0 if (prec + rec) == 0.0 else 2.0 * prec * rec / (prec + rec)


def score_name(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    left = canonicalize_name(baseline.product_name)
    right = canonicalize_name(candidate.product_name)
    if left is None or right is None:
        return None
    if left == right:
        return 1.0
    return levenshtein_ratio(left, right)


def score_product_smiles(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    left = canonicalize_smiles(baseline.product_smiles)
    right = canonicalize_smiles(candidate.product_smiles)
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def _canonical_smiles_set(smiles_list: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            out.add(canonical)
    return out


def _canonical_name_set(names: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for name in names:
        canonical = canonicalize_name(name)
        if canonical is not None:
            out.add(canonical)
    return out


def score_reactants(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    baseline_smi = _canonical_smiles_set(baseline.reactant_smiles)
    candidate_smi = _canonical_smiles_set(candidate.reactant_smiles)
    if baseline_smi and candidate_smi:
        left, right = baseline_smi, candidate_smi
    else:
        left = _canonical_name_set(baseline.reactant_names)
        right = _canonical_name_set(candidate.reactant_names)

    if not left and not right:
        return None
    if not left or not right:
        return 0.0
    return jaccard_similarity(left, right)


def score_procedure(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    if baseline.procedure_vector is None or candidate.procedure_vector is None:
        return None
    return cosine_similarity(baseline.procedure_vector, candidate.procedure_vector)


def score_yield(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    left = baseline.product_yield_pct
    right = candidate.product_yield_pct
    if left is None or right is None:
        return None
    diff = abs(left - right)
    if diff <= 2.0:
        return 1.0
    if diff <= 5.0:
        return 0.75
    if diff <= 10.0:
        return 0.40
    return 0.0


def atmosphere_bucket(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if "argon" in text or text == "ar":
        return "inert"
    if "nitrogen" in text or text == "n2":
        return "inert"
    if text == "air":
        return "air"
    if "hydrogen" in text or text == "h2":
        return "h2"
    if "ammonia" in text or text == "nh3":
        return "nh3"
    if "vacuum" in text:
        return "vac"
    return None


_ROOM_TEMP_C = 25.0


def score_temperature(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    baseline_room = baseline.room_temperature is True
    candidate_room = candidate.room_temperature is True
    if baseline_room and candidate_room:
        return 1.0

    baseline_temp = _ROOM_TEMP_C if baseline_room else baseline.temperature_c
    candidate_temp = _ROOM_TEMP_C if candidate_room else candidate.temperature_c
    if baseline_temp is None or candidate_temp is None:
        return None

    diff = abs(baseline_temp - candidate_temp)
    if diff <= 5.0:
        return 1.0
    if diff <= 10.0:
        return 0.5
    return 0.0


def score_time(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    left = baseline.time_h
    right = candidate.time_h
    if left is None or right is None:
        return None
    larger = max(abs(left), abs(right))
    if larger == 0.0:
        return 1.0
    rel = abs(left - right) / larger
    return 1.0 if rel <= 0.20 else 0.0


def score_atmosphere(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    left = atmosphere_bucket(baseline.atmosphere)
    right = atmosphere_bucket(candidate.atmosphere)
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def score_conditions(baseline: ReactionEntry, candidate: ReactionEntry) -> float | None:
    scores = [
        score_temperature(baseline, candidate),
        score_time(baseline, candidate),
        score_atmosphere(baseline, candidate),
    ]
    present = [s for s in scores if s is not None]
    if not present:
        return None
    return sum(present) / len(present)


def normalize_label_for_diagnostic(section_label: str | None, step_label: str | None) -> str | None:
    if section_label is None and step_label is None:
        return None
    section = "" if section_label is None else section_label.lower().strip()
    step = "" if step_label is None else step_label.lower().strip()
    key = section + ("" if not step else f" | {step}")
    return key or None


def labels_agree(baseline: ReactionEntry, candidate: ReactionEntry) -> bool:
    left = normalize_label_for_diagnostic(baseline.section_label, baseline.step_label)
    right = normalize_label_for_diagnostic(candidate.section_label, candidate.step_label)
    return left is not None and left == right


def product_smiles_agree(baseline: ReactionEntry, candidate: ReactionEntry) -> bool:
    left = canonicalize_smiles(baseline.product_smiles)
    right = canonicalize_smiles(candidate.product_smiles)
    return left is not None and right is not None and left == right


def _normalized_substring_match(left: str, right: str) -> bool:
    def words(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9]", " ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return {w for w in cleaned.split(" ") if w}

    a_words = words(left)
    b_words = words(right)
    if not a_words or not b_words:
        return False
    shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    return longer.issuperset(shorter)


def _score_pair(
    baseline_index: int,
    candidate_index: int,
    baseline: ReactionEntry,
    candidate: ReactionEntry,
) -> _PairScore:
    ps = _PairScore(baseline_index=baseline_index, candidate_index=candidate_index)
    ps.axis_name = score_name(baseline, candidate)
    ps.axis_smiles = score_product_smiles(baseline, candidate)
    ps.axis_reactants = score_reactants(baseline, candidate)
    ps.axis_procedure = score_procedure(baseline, candidate)
    ps.axis_yield = score_yield(baseline, candidate)
    ps.axis_conditions = score_conditions(baseline, candidate)
    ps.composite = compose_weighted_sum(
        ps.axis_name,
        ps.axis_smiles,
        ps.axis_reactants,
        ps.axis_procedure,
        ps.axis_yield,
        ps.axis_conditions,
    )
    ps.label_match = labels_agree(baseline, candidate)
    ps.product_smiles_match = product_smiles_agree(baseline, candidate)
    return ps


def _filter_synthetic(
    entries: list[ReactionEntry],
) -> tuple[list[ReactionEntry], int]:
    kept: list[ReactionEntry] = []
    skipped = 0
    for entry in entries:
        if entry.non_synthetic:
            skipped += 1
            continue
        kept.append(entry)
    return kept, skipped


def _count_legacy_label_overlap(
    baseline: list[ReactionEntry],
    candidate: list[ReactionEntry],
) -> int:
    baseline_keys = {
        key
        for entry in baseline
        if (key := normalize_label_for_diagnostic(entry.section_label, entry.step_label)) is not None
    }
    matched: set[str] = set()
    overlap = 0
    for entry in candidate:
        key = normalize_label_for_diagnostic(entry.section_label, entry.step_label)
        if key is not None and key in baseline_keys and key not in matched:
            overlap += 1
            matched.add(key)
    return overlap


def _axis_scores_from_pair(ps: _PairScore) -> AxisScores:
    return AxisScores(
        product_name=ps.axis_name,
        product_smiles=ps.axis_smiles,
        reactant_jaccard=ps.axis_reactants,
        procedure_cosine=ps.axis_procedure,
        yield_score=ps.axis_yield,
        conditions=ps.axis_conditions,
    )


def _build_match_detail(
    match_type: MatchType,
    baseline: ReactionEntry | None,
    candidate: ReactionEntry | None,
    baseline_index: int | None,
    candidate_index: int | None,
    ps: _PairScore | None = None,
) -> ReactionMatchDetail:
    yield_diff = None
    class_match = None
    if (
        baseline is not None
        and candidate is not None
        and baseline.product_yield_pct is not None
        and candidate.product_yield_pct is not None
    ):
        yield_diff = abs(baseline.product_yield_pct - candidate.product_yield_pct)
    if (
        baseline is not None
        and candidate is not None
        and baseline.reaction_class is not None
        and candidate.reaction_class is not None
    ):
        class_match = _normalized_substring_match(
            baseline.reaction_class, candidate.reaction_class
        )

    return ReactionMatchDetail(
        match_type=match_type,
        baseline_index=baseline_index,
        candidate_index=candidate_index,
        baseline=baseline,
        candidate=candidate,
        composite_score=None if ps is None else ps.composite,
        axis_scores=None if ps is None else _axis_scores_from_pair(ps),
        label_match=None if ps is None else ps.label_match,
        product_smiles_match=None if ps is None else ps.product_smiles_match,
        reactant_jaccard=None if ps is None else ps.axis_reactants,
        procedure_cosine_similarity=None if ps is None else ps.axis_procedure,
        yield_difference=yield_diff,
        reaction_class_match=class_match,
    )


def _compute_axis_coverage(tp_pairs: list[_PairScore]) -> AxisCoverage | None:
    if not tp_pairs:
        return None
    n = len(tp_pairs)
    return AxisCoverage(
        product_name=sum(1 for p in tp_pairs if p.axis_name is not None) / n,
        product_smiles=sum(1 for p in tp_pairs if p.axis_smiles is not None) / n,
        reactant_jaccard=sum(1 for p in tp_pairs if p.axis_reactants is not None) / n,
        procedure_cosine=sum(1 for p in tp_pairs if p.axis_procedure is not None) / n,
        yield_score=sum(1 for p in tp_pairs if p.axis_yield is not None) / n,
        conditions=sum(1 for p in tp_pairs if p.axis_conditions is not None) / n,
    )


def compare_reactions(
    baseline: list[ReactionEntry],
    candidate: list[ReactionEntry],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> ReactionBenchmarkReport:
    """Compare one candidate model against a baseline (pairwise vs baseline).

    Filters ``non_synthetic`` records on both sides, scores all pairs on six
    axes, then assigns true positives via sorted-greedy bipartite matching
    with composite ≥ ``TP_THRESHOLD`` (0.55).
    """
    baseline_kept, skipped_baseline = _filter_synthetic(baseline)
    candidate_kept, skipped_candidate = _filter_synthetic(candidate)

    n_baseline = len(baseline_kept)
    n_candidate = len(candidate_kept)

    all_pairs: list[_PairScore] = []
    for i, cand in enumerate(candidate_kept):
        for j, base in enumerate(baseline_kept):
            all_pairs.append(_score_pair(j, i, base, cand))

    candidates = sorted(
        (p for p in all_pairs if p.composite is not None and p.composite >= TP_THRESHOLD),
        key=lambda p: p.composite or 0.0,
        reverse=True,
    )

    matched_baseline: set[int] = set()
    matched_candidate: set[int] = set()
    tp_pairs: list[_PairScore] = []
    match_details: list[ReactionMatchDetail] = []

    for pair in candidates:
        if pair.candidate_index in matched_candidate or pair.baseline_index in matched_baseline:
            continue
        matched_candidate.add(pair.candidate_index)
        matched_baseline.add(pair.baseline_index)
        tp_pairs.append(pair)
        match_details.append(
            _build_match_detail(
                "CONTENT_MATCH",
                baseline_kept[pair.baseline_index],
                candidate_kept[pair.candidate_index],
                pair.baseline_index,
                pair.candidate_index,
                pair,
            )
        )

    synonym_near_miss = 0
    tp_pair_keys = {(p.candidate_index, p.baseline_index) for p in tp_pairs}
    for pair in all_pairs:
        axis_a = pair.axis_name
        if axis_a is None:
            continue
        if not (SYNONYM_NEAR_MISS_LOW <= axis_a < SYNONYM_NEAR_MISS_HIGH):
            continue
        won = (pair.candidate_index, pair.baseline_index) in tp_pair_keys
        if not won:
            synonym_near_miss += 1

    for i, cand in enumerate(candidate_kept):
        if i in matched_candidate:
            continue
        match_details.append(
            _build_match_detail("FALSE_POSITIVE", None, cand, None, i)
        )

    for j, base in enumerate(baseline_kept):
        if j in matched_baseline:
            continue
        match_details.append(
            _build_match_detail("FALSE_NEGATIVE", base, None, j, None)
        )

    tp = len(tp_pairs)
    fp = sum(1 for m in match_details if m.match_type == "FALSE_POSITIVE")
    fn = sum(1 for m in match_details if m.match_type == "FALSE_NEGATIVE")
    prec = precision(tp, fp)
    rec = recall(tp, fn)

    jaccards: list[float] = []
    yield_pairs: list[tuple[float, float]] = []
    class_booleans: list[float] = []
    procedure_sims: list[float] = []
    for detail in match_details:
        if detail.match_type != "CONTENT_MATCH":
            continue
        if detail.reactant_jaccard is not None:
            jaccards.append(detail.reactant_jaccard)
        if (
            detail.baseline is not None
            and detail.candidate is not None
            and detail.baseline.product_yield_pct is not None
            and detail.candidate.product_yield_pct is not None
        ):
            yield_pairs.append(
                (detail.candidate.product_yield_pct, detail.baseline.product_yield_pct)
            )
        if detail.reaction_class_match is not None:
            class_booleans.append(1.0 if detail.reaction_class_match else 0.0)
        if detail.procedure_cosine_similarity is not None:
            procedure_sims.append(detail.procedure_cosine_similarity)

    perfect_reactant_pct = (
        None
        if not jaccards
        else 100.0 * sum(1 for value in jaccards if value == 1.0) / len(jaccards)
    )
    yield_mae = (
        None
        if not yield_pairs
        else sum(abs(a - b) for a, b in yield_pairs) / len(yield_pairs)
    )
    yield_within_5 = (
        None
        if not yield_pairs
        else 100.0 * sum(1 for a, b in yield_pairs if abs(a - b) <= 5.0) / len(yield_pairs)
    )

    legacy_overlap = _count_legacy_label_overlap(baseline_kept, candidate_kept)
    legacy_fp = max(0, n_candidate - legacy_overlap)
    legacy_fn = max(0, n_baseline - legacy_overlap)

    return ReactionBenchmarkReport(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_reaction_count=n_baseline,
        candidate_reaction_count=n_candidate,
        non_synthetic_skipped_baseline=skipped_baseline,
        non_synthetic_skipped_candidate=skipped_candidate,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=prec,
        recall=rec,
        f1=f1_score(prec, rec),
        synonym_near_miss=synonym_near_miss,
        avg_reactant_jaccard=(sum(jaccards) / len(jaccards)) if jaccards else None,
        perfect_reactant_match_pct=perfect_reactant_pct,
        yield_mae=yield_mae,
        yield_within_5ppt_pct=yield_within_5,
        reaction_class_accuracy=(
            sum(class_booleans) / len(class_booleans) if class_booleans else None
        ),
        avg_procedure_similarity=(
            sum(procedure_sims) / len(procedure_sims) if procedure_sims else None
        ),
        legacy_label_precision=precision(legacy_overlap, legacy_fp),
        legacy_label_recall=recall(legacy_overlap, legacy_fn),
        axis_coverage=_compute_axis_coverage(tp_pairs),
        match_details=match_details,
    )
