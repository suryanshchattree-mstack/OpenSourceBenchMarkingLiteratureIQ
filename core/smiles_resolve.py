"""Live name→SMILES resolve: PubChem then OPSIN, RDKit canonicalize, Markush skip."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, MutableMapping
from typing import Optional

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

OPSIN_SMILES_URL = "https://opsin.ch.cam.ac.uk/opsin/{name}.smi"
DEFAULT_HTTP_TIMEOUT_S = 8.0

ResolveNameFn = Callable[[str], Optional[str]]

# Bare halogen / Markush class tokens (whole name).
_BARE_MARKUSH = re.compile(
    r"(?i)^(hal|halogen|alkyl|aryl|heteroaryl|alkoxy|r\d*|x|y|z)$"
)
# "compound of formula …", "formula (I)", "of formula II", etc.
_FORMULA_MARKUSH = re.compile(
    r"(?i)\b(?:compound\s+of\s+)?formula\s*\(?\s*[ivx\d]+\s*\)?"
)
_SHORT_R_GROUP = re.compile(r"(?i)^[RXYZM]+[-]?(?:OH|OR|OM|H|X)?$")


def looks_like_markush(name: str | None) -> bool:
    """Cheap heuristic to skip Markush / class noise before network lookups."""
    if name is None:
        return True
    text = str(name).strip()
    if not text:
        return True
    if len(text) <= 5 and _SHORT_R_GROUP.match(text):
        return True
    lower = text.lower()
    if _BARE_MARKUSH.match(lower):
        return True
    if _FORMULA_MARKUSH.search(lower):
        return True
    if (
        lower.startswith("r-")
        or lower.startswith("r1")
        or lower.startswith("r2")
        or lower.startswith("r3")
        or "alkoxy" in lower
        or "aryl group" in lower
        or "alkyl group" in lower
        or "r group" in lower
        or "protecting group" in lower
    ):
        return True
    return False


def canonicalize_smiles_rdkit(smiles: str | None) -> str | None:
    """RDKit-canonicalize a SMILES string; return None on missing/invalid input."""
    if smiles is None or not str(smiles).strip():
        return None
    if Chem is None:
        raise ImportError(
            "rdkit is required for SMILES canonicalization. Install it with: pip install rdkit"
        )
    cleaned = str(smiles).strip()
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def pubchem_name_to_smiles(name: str, *, timeout_s: float = DEFAULT_HTTP_TIMEOUT_S) -> str | None:
    """Resolve a chemical name via PubChem (pubchempy). Returns None on miss/error."""
    del timeout_s  # pubchempy uses urllib without a caller-set timeout hook
    try:
        import pubchempy as pcp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pubchempy is required for live PubChem name→SMILES. "
            "Install it with: pip install pubchempy"
        ) from exc
    try:
        compounds = pcp.get_compounds(name, "name")
    except Exception:
        return None
    if not compounds:
        return None
    compound = compounds[0]
    smiles = getattr(compound, "canonical_smiles", None) or getattr(
        compound, "isomeric_smiles", None
    )
    if smiles is None or not str(smiles).strip():
        return None
    return str(smiles).strip()


def opsin_name_to_smiles(
    name: str,
    *,
    timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> str | None:
    """Resolve an IUPAC-ish name via Cambridge OPSIN HTTP ``.smi`` endpoint."""
    quoted = urllib.parse.quote(name.strip(), safe="")
    url = OPSIN_SMILES_URL.format(name=quoted)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    if not body or body.lower().startswith("<!doctype") or body.lower().startswith("<html"):
        return None
    # OPSIN may return plain SMILES or "SMILES\n..." — take first non-empty line.
    for line in body.splitlines():
        text = line.strip()
        if text and not text.startswith("<"):
            return text
    return None


def resolve_name_to_smiles(
    name: str | None,
    *,
    cache: MutableMapping[str, str | None] | None = None,
    pubchem_fn: ResolveNameFn | None = None,
    opsin_fn: ResolveNameFn | None = None,
) -> str | None:
    """
    Resolve ``name`` → canonical SMILES.

    Order: Markush skip → cache → PubChem → OPSIN → RDKit canonicalize.
    Cache stores ``name → smiles|None`` (including negative results).
    """
    if name is None or not str(name).strip():
        return None
    key = str(name).strip()
    if looks_like_markush(key):
        if cache is not None:
            cache[key] = None
        return None
    if cache is not None and key in cache:
        return cache[key]

    pubchem = pubchem_fn or pubchem_name_to_smiles
    opsin = opsin_fn or opsin_name_to_smiles

    raw_smiles: str | None = None
    try:
        raw_smiles = pubchem(key)
    except Exception:
        raw_smiles = None
    if raw_smiles is None:
        try:
            raw_smiles = opsin(key)
        except Exception:
            raw_smiles = None

    canonical = canonicalize_smiles_rdkit(raw_smiles) if raw_smiles else None
    if cache is not None:
        cache[key] = canonical
    return canonical
