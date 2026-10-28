"""Paths, environment variables, and constants.

The prepared microdata files are copied into this repository's `data/` and read
from there. When a local copy is missing, the path falls back to the outputs of
`preprocessing/` (`preprocessing_root`).

Person-level rows (`data/interim/`) are not pushed to origin (.gitignore).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[2]  # src/sfac_persona/config.py → repo root


class SFACSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SFP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    project_root: Path = _DEFAULT_ROOT
    data_root: Path = _DEFAULT_ROOT / "data"

    # ── Microdata preparation outputs (fallback when the local copy is missing) ─
    preprocessing_root: Path = _DEFAULT_ROOT / "preprocessing"
    survey_year: int = 2024
    # General public. Culture-interest and disability panels are separate editions.
    microdata_group: str = "general"
    weight_var: str = "WT1_2"

    # ── Public-release rules (within the foundation's derivative approval) ─
    cell_floor: int = 30  # smaller unweighted cells are dropped from public tables
    geo_max_level: str = "living_area"  # dong-level (SQ2_NAME) is not used

    # ── LLM (local Ollama first) ──────────────────────────────────
    llm_budget_usd: float = 0.0
    llm_default_model: str = "qwen3.8:27b-mtp-q4_K_M"
    llm_judge_model: str = "gemma4:12b"
    llm_backend: str = "ollama_native"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = "EMPTY"
    llm_max_concurrent: int = 4

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    zenodo_token: str | None = Field(default=None, alias="ZENODO_TOKEN")

    def _prefer(self, local: Path, fallback: Path) -> Path:
        return local if local.exists() else fallback

    # ── Derived paths ────────────────────────────────────────────
    @property
    def microdata_dir(self) -> Path:
        """xlsx fallback. Local raw is not kept — usually look in the preprocessing repository."""
        local = self.data_root / "raw" / "microdata" / str(self.survey_year)
        return self._prefer(
            local,
            self.preprocessing_root / "data" / "raw" / "microdata" / str(self.survey_year),
        )

    @property
    def microdata_cache(self) -> Path:
        """Pickle produced by official_13. Prefer the local copy."""
        name = f"{self.survey_year}_{self.microdata_group}.pkl"
        local = self.data_root / "interim" / "microdata" / name
        return self._prefer(
            local,
            self.preprocessing_root / "data" / "interim" / "microdata" / name,
        )

    @property
    def preprocessing_processed(self) -> Path:
        return self.preprocessing_root / "data" / "processed"

    @property
    def reference_dir(self) -> Path:
        """Published tables, population, and living-area crosswalk (owned here)."""
        return self.data_root / "reference"

    @property
    def harmonized_long(self) -> Path:
        # Frozen observed rows from the old harmonization track's
        # survey_harmonized_long.csv. Column names are the same.
        return self.reference_dir / "published_tables_2018_2024.csv"

    @property
    def value_labels_csv(self) -> Path:
        local = self.data_root / "processed" / "microdata" / "microdata_value_labels.csv"
        return self._prefer(
            local,
            self.preprocessing_processed / "microdata" / "microdata_value_labels.csv",
        )

    @property
    def official_person_dir(self) -> Path:
        """Cleaned person-level parquet from the preprocessing repository (local copy)."""
        local = self.data_root / "interim" / "official"
        return self._prefer(
            local,
            self.preprocessing_root / "data" / "interim" / "official",
        )

    @property
    def population_prior_csv(self) -> Path:
        return self.reference_dir / "culture_nowcast_population_prior.csv"

    @property
    def official_value_labels_csv(self) -> Path:
        local = self.data_root / "processed" / "official" / "official_value_labels.csv"
        return self._prefer(
            local,
            self.preprocessing_processed / "official" / "official_value_labels.csv",
        )

    def era_grounding_dir(self, wave: int) -> Path:
        return self.grounding_dir / f"era_{wave}"

    @property
    def living_area_crosswalk_csv(self) -> Path:
        return self.reference_dir / "seoul_living_area_crosswalk.csv"

    @property
    def grounding_dir(self) -> Path:
        return self.data_root / "grounding"

    @property
    def prompts_dir(self) -> Path:
        return self.data_root / "prompts"

    @property
    def synthetic_dir(self) -> Path:
        return self.data_root / "synthetic"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


@lru_cache(maxsize=1)
def get_settings() -> SFACSettings:
    return SFACSettings()


settings = get_settings()
