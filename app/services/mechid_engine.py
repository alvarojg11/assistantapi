from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional


ASTResult = Literal["Susceptible", "Intermediate", "Resistant"]


class MechIDEngineError(RuntimeError):
    pass


class _StreamlitStop(Exception):
    pass


def _create_pandas_stub() -> types.ModuleType:
    module = types.ModuleType("pandas")
    module.DataFrame = lambda rows=None, *args, **kwargs: rows if rows is not None else []
    return module


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def _create_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    ctx = _DummyContext()

    def _noop(*args, **kwargs):
        return None

    def _selectbox(_label, options, index: int = 0, **kwargs):
        if not options:
            return None
        idx = max(0, min(index, len(options) - 1))
        return options[idx]

    def _radio(_label, options, index: int = 0, **kwargs):
        return _selectbox(_label, options, index=index, **kwargs)

    def _multiselect(_label, options, default=None, **kwargs):
        return list(default or [])

    def _checkbox(_label, value: bool = False, **kwargs):
        return value

    def _text_input(_label, value: str = "", **kwargs):
        return value

    def _text_area(_label, value: str = "", **kwargs):
        return value

    def _button(*args, **kwargs):
        return False

    def _columns(spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return tuple(_DummyContext() for _ in range(count))

    def _tabs(names, **kwargs):
        return tuple(_DummyContext() for _ in names)

    def _expander(*args, **kwargs):
        return _DummyContext()

    def _container(*args, **kwargs):
        return _DummyContext()

    def _form(*args, **kwargs):
        return _DummyContext()

    def _cache_data(func=None, **kwargs):
        if func is None:
            return lambda inner: inner
        return func

    def _stop():
        raise _StreamlitStop()

    module.set_page_config = _noop
    module.markdown = _noop
    module.caption = _noop
    module.subheader = _noop
    module.info = _noop
    module.error = _noop
    module.success = _noop
    module.warning = _noop
    module.write = _noop
    module.title = _noop
    module.dataframe = _noop
    module.table = _noop
    module.metric = _noop
    module.download_button = _button
    module.file_uploader = lambda *args, **kwargs: None
    module.selectbox = _selectbox
    module.radio = _radio
    module.multiselect = _multiselect
    module.checkbox = _checkbox
    module.text_input = _text_input
    module.text_area = _text_area
    module.button = _button
    module.columns = _columns
    module.tabs = _tabs
    module.expander = _expander
    module.container = _container
    module.form = _form
    module.cache_data = _cache_data
    module.stop = _stop
    module.sidebar = _DummyContext()
    module.session_state = {}
    module.__getattr__ = lambda _name: _noop
    return module


def _mechid_app_path() -> Path:
    env_path = os.getenv("MECHID_SOURCE_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[1] / "data" / "mechid" / "app_gnr.py",
            here.parents[3] / "App Micro mechanisms" / "app_gnr.py",
            Path.cwd() / "app" / "data" / "mechid" / "app_gnr.py",
            Path.cwd() / "App Micro mechanisms" / "app_gnr.py",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise MechIDEngineError(f"MechID source not found. Tried: {attempted}")


@lru_cache(maxsize=1)
def load_mechid_module():
    app_path = _mechid_app_path()

    spec = importlib.util.spec_from_file_location("mechid_app_gnr_api", app_path)
    if spec is None or spec.loader is None:
        raise MechIDEngineError(f"Could not load MechID source from {app_path}")

    module = importlib.util.module_from_spec(spec)
    previous_streamlit = sys.modules.get("streamlit")
    previous_pandas = sys.modules.get("pandas")
    sys.modules["streamlit"] = _create_streamlit_stub()
    if previous_pandas is None:
        sys.modules["pandas"] = _create_pandas_stub()
    try:
        try:
            spec.loader.exec_module(module)
        except _StreamlitStop:
            # Import may intentionally halt after UI setup; core functions are already defined.
            pass
    except ModuleNotFoundError as exc:
        raise MechIDEngineError(f"MechID dependency missing: {exc.name}. Install backend requirements.") from exc
    except Exception as exc:  # pragma: no cover - defensive runtime wrapper
        raise MechIDEngineError(f"Failed to import MechID logic: {exc}") from exc
    finally:
        if previous_streamlit is not None:
            sys.modules["streamlit"] = previous_streamlit
        else:
            sys.modules.pop("streamlit", None)
        if previous_pandas is not None:
            sys.modules["pandas"] = previous_pandas
        else:
            sys.modules.pop("pandas", None)

    required_attrs = [
        "ORGANISM_REGISTRY",
        "PANEL",
        "RULES",
        "normalize_org",
        "apply_cascade",
        "run_mechanisms_and_therapy_for",
        "_collect_mech_ref_keys",
    ]
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise MechIDEngineError(f"Imported MechID module is missing required attribute: {attr}")
    return module


def list_mechid_organisms() -> List[str]:
    module = load_mechid_module()
    return sorted(module.ORGANISM_REGISTRY.keys())


def organism_panel(organism: str) -> List[str]:
    module = load_mechid_module()
    normalized = normalize_organism(organism)
    return list(module.PANEL.get(normalized, ()))


def normalize_result(value: str | None) -> ASTResult | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text in {"s", "susceptible", "sensitive"}:
        return "Susceptible"
    if text in {"i", "intermediate"}:
        return "Intermediate"
    if text in {"r", "resistant", "nonsusceptible", "non-susceptible"}:
        return "Resistant"
    raise MechIDEngineError(f"Unsupported susceptibility value: {value}")


def normalize_organism(organism: str) -> str:
    module = load_mechid_module()
    candidate = module.normalize_org(organism)
    if candidate in module.ORGANISM_REGISTRY:
        return candidate

    lowered = organism.strip().lower()
    aliases = {
        "e coli": "Escherichia coli",
        "e. coli": "Escherichia coli",
        "ecoli": "Escherichia coli",
        "kleb": "Klebsiella pneumoniae",
        "klebsiella": "Klebsiella pneumoniae",
        "pseudomonas": "Pseudomonas aeruginosa",
        "psa": "Pseudomonas aeruginosa",
        "acinetobacter": "Acinetobacter baumannii complex",
        "steno": "Stenotrophomonas maltophilia",
        "mrsa": "Staphylococcus aureus",
        "staph aureus": "Staphylococcus aureus",
        "s aureus": "Staphylococcus aureus",
        "vre faecium": "Enterococcus faecium",
        "vre faecalis": "Enterococcus faecalis",
        "pneumococcus": "Streptococcus pneumoniae",
    }
    if lowered in aliases and aliases[lowered] in module.ORGANISM_REGISTRY:
        return aliases[lowered]

    for name in module.ORGANISM_REGISTRY:
        if lowered == name.lower() or lowered in name.lower():
            return name

    raise MechIDEngineError(f"Unsupported or unrecognized organism: {organism}")


def canonical_antibiotic_aliases(organism: str) -> Dict[str, str]:
    panel = organism_panel(organism)
    aliases: Dict[str, str] = {}
    custom = {
        "pip/tazo": "Piperacillin/Tazobactam",
        "piptazo": "Piperacillin/Tazobactam",
        "zosyn": "Piperacillin/Tazobactam",
        "ctx": "Ceftriaxone",
        "ceftriax": "Ceftriaxone",
        "mero": "Meropenem",
        "erta": "Ertapenem",
        "imi": "Imipenem",
        "gent": "Gentamicin",
        "tobra": "Tobramycin",
        "amik": "Amikacin",
        "cipro": "Ciprofloxacin",
        "levo": "Levofloxacin",
        "tmp-smx": "Trimethoprim/Sulfamethoxazole",
        "bactrim": "Trimethoprim/Sulfamethoxazole",
        "nitro": "Nitrofurantoin",
        "vanc": "Vancomycin",
        "dapto": "Daptomycin",
        "linezolid": "Linezolid",
        "pen g": "Penicillin",
        "oxa": "Nafcillin/Oxacillin",
    }
    for ab in panel:
        key = ab.lower()
        aliases[key] = ab
        aliases[key.replace("/", " / ")] = ab
        aliases[key.replace("/", "")] = ab
    for alias, canonical in custom.items():
        if canonical in panel:
            aliases[alias] = canonical
    return aliases


def _canonicalize_results_for_organism(organism: str, results: Dict[str, str | None]) -> Dict[str, ASTResult]:
    aliases = canonical_antibiotic_aliases(organism)
    canonical: Dict[str, ASTResult] = {}
    for raw_name, raw_value in results.items():
        value = normalize_result(raw_value)
        if value is None:
            continue
        key = raw_name.strip().lower()
        antibiotic = aliases.get(key)
        if antibiotic is None:
            for alias, canonical_name in aliases.items():
                if key == alias or key in alias or alias in key:
                    antibiotic = canonical_name
                    break
        if antibiotic is None:
            raise MechIDEngineError(f"Unsupported antibiotic for {organism}: {raw_name}")
        canonical[antibiotic] = value
    return canonical


def analyze_mechid(
    *,
    organism: str,
    susceptibility_results: Dict[str, str | None],
    tx_context: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    module = load_mechid_module()
    normalized_org = normalize_organism(organism)
    normalized_results = _canonicalize_results_for_organism(normalized_org, susceptibility_results)
    rules = module.RULES.get(normalized_org, {"intrinsic_resistance": [], "cascade": []})
    inferred = module.apply_cascade(rules, normalized_results)

    final = defaultdict(lambda: None)
    for name, value in {**inferred, **normalized_results}.items():
        final[name] = value
    for antibiotic in rules.get("intrinsic_resistance", []):
        final[antibiotic] = "Resistant"

    mechanisms, banners, greens, therapy = module.run_mechanisms_and_therapy_for(
        normalized_org,
        final,
        tx_context=tx_context,
    )
    references = module._collect_mech_ref_keys(normalized_org, list(mechanisms or []) + list(therapy or []), banners or [])

    rows: List[Dict[str, str]] = []
    seen_antibiotics = set(final.keys()) | set(normalized_results.keys())
    for antibiotic in sorted(seen_antibiotics):
        result = final[antibiotic]
        if result is None:
            continue
        source = "user"
        if antibiotic in rules.get("intrinsic_resistance", []):
            source = "intrinsic_rule"
        elif antibiotic in inferred and antibiotic not in normalized_results:
            source = "cascade_rule"
        rows.append({"antibiotic": antibiotic, "result": result, "source": source})

    return {
        "organism": normalized_org,
        "submitted_results": normalized_results,
        "inferred_results": inferred,
        "final_results": {row["antibiotic"]: row["result"] for row in rows},
        "rows": rows,
        "mechanisms": list(mechanisms or []),
        "cautions": list(banners or []),
        "favorable_signals": list(greens or []),
        "therapy_notes": list(therapy or []),
        "references": references,
        "panel": organism_panel(normalized_org),
    }
