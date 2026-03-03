from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import TypeAdapter

from ..data.demo_probid_modules import DEMO_MODULES
from ..schemas import ModuleSummary, SyndromeModule


class InMemoryModuleStore:
    def __init__(self) -> None:
        self._modules: Dict[str, SyndromeModule] = {m.id: m for m in self._load_initial_modules()}

    def _load_initial_modules(self) -> List[SyndromeModule]:
        data_path = Path(__file__).resolve().parents[1] / "data" / "probid_modules.json"
        if not data_path.exists():
            return DEMO_MODULES

        raw = json.loads(data_path.read_text())
        adapter = TypeAdapter(List[SyndromeModule])
        return adapter.validate_python(raw)

    def list_summaries(self) -> List[ModuleSummary]:
        return [
            ModuleSummary(
                id=m.id,
                name=m.name,
                itemCount=len(m.items),
                presetCount=len(m.pretest_presets),
            )
            for m in self._modules.values()
        ]

    def get(self, module_id: str) -> SyndromeModule | None:
        return self._modules.get(module_id)

    def upsert_many(self, modules: List[SyndromeModule]) -> List[str]:
        ids: List[str] = []
        for module in modules:
            self._modules[module.id] = module
            ids.append(module.id)
        return ids
