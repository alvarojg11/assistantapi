from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional


ASTResult = Literal["Susceptible", "Intermediate", "Resistant"]

SUPPLEMENTAL_AGENT_ALIASES: Dict[str, List[str]] = {
    "Ceftazidime/Avibactam": [
        "ceftazidime/avibactam",
        "ceftazidime avibactam",
        "ceftazidime-avibactam",
        "caz avi",
        "caz-avi",
        "caz/avi",
        "avycaz",
    ],
    "Meropenem/Vaborbactam": [
        "meropenem/vaborbactam",
        "meropenem vaborbactam",
        "meropenem-vaborbactam",
        "mero vabor",
        "mero-vabor",
        "mero/vabor",
        "vabomere",
    ],
    "Imipenem/Cilastatin/Relebactam": [
        "imipenem/cilastatin/relebactam",
        "imipenem cilastatin relebactam",
        "imipenem-cilastatin-relebactam",
        "imipenem relebactam",
        "imi rele",
        "imi-rele",
        "imi/rele",
        "recarbrio",
    ],
    "Cefiderocol": [
        "cefiderocol",
        "fetroja",
    ],
    "Aztreonam": [
        "aztreonam",
        "azt",
    ],
}


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
        aliases[key.replace("/", "-")] = ab
        aliases[key.replace("/", " ")] = ab
    for alias, canonical in custom.items():
        if canonical in panel:
            aliases[alias] = canonical
    for canonical, alias_list in SUPPLEMENTAL_AGENT_ALIASES.items():
        aliases[canonical.lower()] = canonical
        for alias in alias_list:
            aliases[alias] = canonical
    return aliases


def resolve_antibiotic_name(organism: str, raw_name: str) -> str | None:
    aliases = canonical_antibiotic_aliases(organism)
    key = raw_name.strip().lower()
    if not key:
        return None
    if key in aliases:
        return aliases[key]

    normalized_key = re.sub(r"[^a-z0-9]+", "", key)
    normalized_aliases = sorted(
        ((alias, canonical) for alias, canonical in aliases.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, canonical in normalized_aliases:
        alias_norm = re.sub(r"[^a-z0-9]+", "", alias)
        if not alias_norm:
            continue
        if normalized_key == alias_norm:
            return canonical
    for alias, canonical in normalized_aliases:
        alias_norm = re.sub(r"[^a-z0-9]+", "", alias)
        if not alias_norm:
            continue
        if normalized_key in alias_norm or alias_norm in normalized_key:
            return canonical
    return None


def _canonicalize_results_for_organism(organism: str, results: Dict[str, str | None]) -> Dict[str, ASTResult]:
    canonical: Dict[str, ASTResult] = {}
    for raw_name, raw_value in results.items():
        value = normalize_result(raw_value)
        if value is None:
            continue
        antibiotic = resolve_antibiotic_name(organism, raw_name)
        if antibiotic is None:
            raise MechIDEngineError(f"Unsupported antibiotic for {organism}: {raw_name}")
        canonical[antibiotic] = value
    return canonical


def _has_carbapenem_resistance(final_results: Dict[str, ASTResult | None]) -> bool:
    return any(
        final_results.get(agent) == "Resistant"
        for agent in ("Ertapenem", "Meropenem", "Imipenem", "Doripenem")
    )


def _normalize_carbapenemase_result(raw_value: object) -> str:
    text = str(raw_value or "").strip().lower()
    if not text or text == "not specified":
        return "Not specified"
    if text in {"positive", "detected", "present"}:
        return "Positive"
    if text in {"negative", "not detected"}:
        return "Negative"
    if text in {"pending", "not tested", "not tested / pending"}:
        return "Not tested / pending"
    return "Not specified"


def _normalize_carbapenemase_class(raw_value: object) -> str:
    text = str(raw_value or "").strip().lower()
    if not text or text == "not specified":
        return "Not specified"
    if text == "kpc":
        return "KPC"
    if text in {"oxa-48", "oxa 48", "oxa-48-like", "oxa 48 like"}:
        return "OXA-48-like"
    if text == "ndm":
        return "NDM"
    if text == "vim":
        return "VIM"
    if text == "imp":
        return "IMP"
    if text in {"other", "unknown", "other / unknown", "mbl", "metallo-beta-lactamase", "metallo beta lactamase"}:
        return "Other / Unknown"
    return "Not specified"


def _cre_carbapenemase_additions(
    module,
    organism: str,
    final_results: Dict[str, ASTResult | None],
    tx_context: Dict[str, object] | None,
) -> Dict[str, List[str]]:
    additions = {
        "mechanisms": [],
        "cautions": [],
        "therapy_notes": [],
    }
    enterobacterales = set(getattr(module, "ENTEROBACTERALES", ()))
    if organism not in enterobacterales or not _has_carbapenem_resistance(final_results):
        return additions

    tx_context = tx_context or {}
    carb_result = _normalize_carbapenemase_result(tx_context.get("carbapenemaseResult"))
    carb_class = _normalize_carbapenemase_class(tx_context.get("carbapenemaseClass"))

    if carb_result == "Not specified" and carb_class != "Not specified":
        carb_result = "Positive"
    if carb_result == "Not specified":
        return additions

    if carb_result == "Not tested / pending":
        additions["therapy_notes"].append(
            "CRE with carbapenemase testing pending: request phenotypic or molecular carbapenemase testing because treatment differs by enzyme class."
        )
        return additions

    if carb_result == "Negative":
        additions["mechanisms"].append("Non-carbapenemase CRE phenotype is more likely than a carbapenemase-mediated CRE pattern.")
        additions["therapy_notes"].append(
            "Non-carbapenemase CRE pattern: if Imipenem or Meropenem remains susceptible and MIC and infection site support use, consider optimized extended-infusion carbapenem dosing. If all carbapenems are non-susceptible, prioritize another confirmed active agent."
        )
        return additions

    additions["mechanisms"].append(
        "Carbapenemase-positive CRE pattern confirmed on top of carbapenem resistance."
    )
    aztreonam_status = final_results.get("Aztreonam")
    cefiderocol_status = final_results.get("Cefiderocol")

    if carb_class == "KPC":
        additions["mechanisms"].append("KPC carbapenemase detected.")
        additions["therapy_notes"].append(
            "KPC carbapenemase pattern: use Meropenem/Vaborbactam, Ceftazidime/Avibactam, or Imipenem/Cilastatin/Relebactam if one is reported susceptible and appropriate for the infection site."
        )
        additions["cautions"].append(
            "Choose among the active KPC-directed beta-lactam options based on site, severity, renal function, and local formulary reporting."
        )
        return additions

    if carb_class == "OXA-48-like":
        additions["mechanisms"].append("OXA-48-like carbapenemase detected.")
        additions["therapy_notes"].append(
            "OXA-48-like carbapenemase pattern: use Ceftazidime/Avibactam if susceptible. Cefiderocol can be considered when it is reported susceptible and fits the infection site."
        )
        additions["cautions"].append(
            "OXA-48-like enzymes are not inhibited by Vaborbactam or Relebactam, so do not assume Meropenem/Vaborbactam or Imipenem/Cilastatin/Relebactam will cover this pattern."
        )
        return additions

    if carb_class in {"NDM", "VIM", "IMP"}:
        additions["mechanisms"].append(f"{carb_class} metallo-beta-lactamase carbapenemase detected.")
        additions["therapy_notes"].append(
            f"{carb_class} carbapenemase pattern: use Ceftazidime/Avibactam plus Aztreonam. Cefiderocol is the main alternative only when the isolate is reported susceptible and clinically appropriate."
        )
        additions["cautions"].append(
            "Do not rely on Ceftazidime/Avibactam alone for a metallo-beta-lactamase producer."
        )
        if aztreonam_status is None:
            additions["cautions"].append(
                "Aztreonam is not reported yet; request it because it helps refine metallo-beta-lactamase treatment planning."
            )
        elif aztreonam_status == "Resistant":
            additions["cautions"].append(
                "Aztreonam resistance on top of an MBL pattern suggests co-produced resistance mechanisms; confirm the AST before relying on aztreonam-based strategies."
            )
        if carb_class == "NDM":
            additions["cautions"].append(
                "For NDM CRE, do not assume cefiderocol activity without isolate-specific testing, and watch for co-mechanisms that can erode activity."
            )
            if cefiderocol_status in {"Intermediate", "Resistant"}:
                additions["cautions"].append("Cefiderocol is non-susceptible here, so avoid it.")
            elif cefiderocol_status is None:
                additions["cautions"].append(
                    "Cefiderocol is not reported yet; request it before considering cefiderocol for NDM CRE."
                )
        return additions

    additions["therapy_notes"].append(
        "Carbapenemase-positive CRE pattern with class not yet clarified: confirm the carbapenemase class with the lab because treatment differs by enzyme class. If an MBL is suspected, Ceftazidime/Avibactam plus Aztreonam is often the leading option."
    )
    return additions


def _carbapenem_discordance_additions(
    organism: str,
    final_results: Dict[str, ASTResult | None],
) -> Dict[str, List[str]]:
    additions = {
        "mechanisms": [],
        "cautions": [],
        "therapy_notes": [],
    }
    meropenem = final_results.get("Meropenem")
    imipenem = final_results.get("Imipenem")
    if meropenem == "Resistant" and imipenem == "Susceptible":
        additions["mechanisms"].append(
            "Discordant carbapenem pattern: Meropenem resistant but Imipenem susceptible."
        )
        additions["therapy_notes"].append(
            "Discordant carbapenem pattern: avoid relying on Imipenem or Meropenem alone when Meropenem is resistant. Prioritize another confirmed active agent or a mechanism-directed newer option when available."
        )
        additions["cautions"].append(
            "Meropenem-resistant / Imipenem-susceptible discordance can be unstable and should not be treated as routine carbapenem susceptibility."
        )
    elif imipenem == "Resistant" and meropenem == "Susceptible":
        additions["mechanisms"].append(
            "Discordant carbapenem pattern: Imipenem resistant but Meropenem susceptible."
        )
        additions["therapy_notes"].append(
            "Discordant carbapenem pattern: do not assume the remaining susceptible carbapenem is a reliable default. Confirm the mechanism and prioritize another clearly active agent when possible."
        )
        additions["cautions"].append(
            "Discordant carbapenem susceptibility should prompt caution before using a carbapenem as definitive therapy."
        )
    return additions


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
    additions = _cre_carbapenemase_additions(module, normalized_org, final, tx_context)
    discordance_additions = _carbapenem_discordance_additions(normalized_org, final)
    additions["mechanisms"].extend(discordance_additions["mechanisms"])
    additions["cautions"].extend(discordance_additions["cautions"])
    additions["therapy_notes"].extend(discordance_additions["therapy_notes"])
    mechanisms = list(additions["mechanisms"]) + list(mechanisms or [])
    banners = list(banners or []) + list(additions["cautions"])
    therapy = list(additions["therapy_notes"]) + list(therapy or [])
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
