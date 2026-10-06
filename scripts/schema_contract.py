"""Schema contract for incoming GHGRP raw files.

The contract describes the RAW file, bilingual headers included. It runs
BEFORE any cleaning. A new, renamed, or missing column is a hard stop:
ECCC changes its releases (the 2017 reporting-threshold change is the
precedent), and the pipeline must never silently adapt to a new schema.

Column kinds: "string" (kept as text), "integer" (whole numbers),
"number" (may carry decimals).
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_clean2 = importlib.import_module("02_clean")

# The 17 expected raw columns, in the order ECCC has published them.
# Short names are the v1 cleaning names, frozen.
CONTRACT = [
    {"raw": "GHGRP ID No. / No d'identification du PDGES", "short": "facility_id", "kind": "string"},
    {"raw": "Reference Year / Année de référence", "short": "year", "kind": "integer"},
    {"raw": "Facility Name / Nom de l'installation", "short": "facility_name", "kind": "string"},
    {"raw": "Facility City or District or Municipality / Ville ou District ou Municipalité de l'installation", "short": "city", "kind": "string"},
    {"raw": "Facility Province or Territory / Province ou territoire de l'installation", "short": "province", "kind": "string"},
    {"raw": "Facility NPRI ID / Numéro d'identification de l'INRP", "short": "npri_id", "kind": "number"},
    {"raw": "Facility NAICS Code / Code SCIAN de l'installation", "short": "naics_code", "kind": "integer"},
    {"raw": "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais", "short": "sector", "kind": "string"},
    {"raw": "French Facility NAICS Code Description / Description du code SCIAN de l'installation en français", "short": "sector_fr", "kind": "string"},
    {"raw": "Reporting Company Legal Name / Dénomination sociale de la société déclarante", "short": "company_legal", "kind": "string"},
    {"raw": "Reporting Company Trade Name / Nom commercial de la société déclarante", "short": "company_trade", "kind": "string"},
    {"raw": "CO2 (tonnes)", "short": "co2", "kind": "number"},
    {"raw": "CH4 (tonnes)", "short": "ch4_t", "kind": "number"},
    {"raw": "CH4 (tonnes CO2e / tonnes éq. CO2)", "short": "ch4_co2e", "kind": "number"},
    {"raw": "N2O (tonnes)", "short": "n2o_t", "kind": "number"},
    {"raw": "N2O (tonnes CO2e / tonnes éq. CO2)", "short": "n2o_co2e", "kind": "number"},
    {"raw": "Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2)", "short": "total_emissions", "kind": "number"},
]

EXPECTED_COLUMNS = [c["raw"] for c in CONTRACT]
RENAME = {c["raw"]: c["short"] for c in CONTRACT}
SHORT_COLUMNS = [c["short"] for c in CONTRACT]
NUMERIC_SHORT = [c["short"] for c in CONTRACT if c["kind"] in ("integer", "number")]

# The rename mapping must stay identical to the frozen v1 cleaning rules.
assert RENAME == _clean2.RENAME, "schema contract rename drifted from 02_clean"


class SchemaMismatch(Exception):
    """Raised when an incoming raw file does not match the contract."""


def check_schema(df, release_id: str = "") -> dict:
    """Diff a raw frame against the contract.

    Returns a report dict. Raises SchemaMismatch on any missing or extra
    column, naming the exact diff. A pure column-order change is reported
    as a warning, not an error, because the rename is name-based.
    """
    actual = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual]
    extra = [c for c in actual if c not in EXPECTED_COLUMNS]
    order_changed = (not missing and not extra
                     and actual != EXPECTED_COLUMNS)
    report = {
        "release_id": release_id,
        "expected_columns": len(EXPECTED_COLUMNS),
        "actual_columns": len(actual),
        "missing_columns": missing,
        "extra_columns": extra,
        "order_changed": order_changed,
        "schema_ok": not missing and not extra,
    }
    if missing or extra:
        raise SchemaMismatch(
            f"Schema mismatch on release '{release_id}': "
            f"missing={missing}, extra={extra}. "
            f"Human review required before cleaning."
        )
    return report
# Schema v2: ECCC 2004-2024 release (82 columns, retrieved 2026-10-06).
# Only the 17 pipeline columns carry short names and kinds. The rest
# are documented as present-but-unused so future ECCC changes to
# THEM also trigger a review.
CONTRACT_V2 = [
    {"raw": "GHGRP ID No. / No d'identification du PDGES", "short": "facility_id", "kind": "string", "pipeline": True},
    {"raw": 'Reference Year / Année de référence', "short": "year", "kind": "integer", "pipeline": True},
    {"raw": "Facility Name / Nom de l'installation", "short": "facility_name", "kind": "string", "pipeline": True},
    {"raw": "Facility Location / Emplacement de l'installation", "short": None, "kind": "unused", "pipeline": False},
    {"raw": "Facility City or District or Municipality / Ville ou District ou Municipalité de l'installation", "short": "city", "kind": "string", "pipeline": True},
    {"raw": "Facility Province or Territory / Province ou territoire de l'installation", "short": "province", "kind": "string", "pipeline": True},
    {"raw": "Facility Postal Code / Code postal de l'installation", "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Latitude', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Longitude', "short": None, "kind": "unused", "pipeline": False},
    {"raw": "Facility NPRI ID / Numéro d'identification de l'INRP", "short": "npri_id", "kind": "number", "pipeline": True},
    {"raw": "Facility NAICS Code / Code SCIAN de l'installation", "short": "naics_code", "kind": "integer", "pipeline": True},
    {"raw": "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais", "short": "sector", "kind": "string", "pipeline": True},
    {"raw": "French Facility NAICS Code Description / Description du code SCIAN de l'installation en français", "short": "sector_fr", "kind": "string", "pipeline": True},
    {"raw": 'Reporting Company Legal Name / Dénomination sociale de la société déclarante', "short": "company_legal", "kind": "string", "pipeline": True},
    {"raw": 'Reporting Company Trade Name / Nom commercial de la société déclarante', "short": "company_trade", "kind": "string", "pipeline": True},
    {"raw": "Reporting Company Business Number / Numéro d'entreprise de la société déclarante", "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Reporting Company DUNS Number / Numéro DUNS de la société déclarante', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Name / Nom du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Position / Poste ou Titre du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Telephone / Numéro de téléphone du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Extension / Poste téléphonique du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Email / Adresse électronique du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Mailing Address / Adresse postale du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact City or District or Municipality / Ville ou District ou Municipalité du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Province or Territory / Province ou Territoire du responsable des renseignements au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Public Contact Postal Code / Code postal du responsable des renseignement au public', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'CO2 (tonnes)', "short": "co2", "kind": "number", "pipeline": True},
    {"raw": 'CH4 (tonnes)', "short": "ch4_t", "kind": "number", "pipeline": True},
    {"raw": 'CH4 (tonnes CO2e / tonnes éq. CO2)', "short": "ch4_co2e", "kind": "number", "pipeline": True},
    {"raw": 'N2O (tonnes)', "short": "n2o_t", "kind": "number", "pipeline": True},
    {"raw": 'N2O (tonnes CO2e / tonnes éq. CO2)', "short": "n2o_co2e", "kind": "number", "pipeline": True},
    {"raw": 'CO2 from biomass combustion (tonnes) / CO2 issu de la combustion de biomasse (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-23 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-23 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-32 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-32 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-41 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-41 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-43-10mee (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-43-10mee (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-125 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-125 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-134 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-134 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-134a (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-134a (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-143 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-143 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-143a (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-143a (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-152a (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-152a (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-227ea (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-227ea (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-236fa (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-236fa (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-245ca (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC-245ca (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'HFC Total (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'CF4 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'CF4 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C2F6 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C2F6 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C3F8 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C3F8 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C4F10 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C4F10 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C4F8 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C4F8 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C5F12 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C5F12 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C6F14 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'C6F14 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'PFC Total (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'SF6 (tonnes)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'SF6 (tonnes CO2e / tonnes éq. CO2)', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2) - excl. CO2bio', "short": "total_emissions", "kind": "number", "pipeline": True},
    {"raw": 'GHGRP Quantification Requirements / Exigences de quantification du PDGES', "short": None, "kind": "unused", "pipeline": False},
    {"raw": "Emission Factors / Coefficients d'émission", "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Engineering Estimates / Estimations techniques', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Mass Balance / Bilan massique', "short": None, "kind": "unused", "pipeline": False},
    {"raw": 'Monitoring or Direct Measurement / Surveillance ou mesure directe', "short": None, "kind": "unused", "pipeline": False},
]


EXPECTED_COLUMNS_V2 = [c["raw"] for c in CONTRACT_V2]
RENAME_V2 = {c["raw"]: c["short"] for c in CONTRACT_V2 if c["pipeline"]}
SHORT_COLUMNS_V2 = [c["short"] for c in CONTRACT_V2 if c["pipeline"]]
assert SHORT_COLUMNS_V2 == SHORT_COLUMNS, \
    "v2 pipeline columns must map to the same short names as v1"

SCHEMAS = {
    "v1": {"expected": EXPECTED_COLUMNS, "rename": RENAME},
    "v2": {"expected": EXPECTED_COLUMNS_V2, "rename": RENAME_V2},
}


def check_schema_versioned(df, release_id: str = "",
                           schema_version: str = "v1") -> dict:
    """Diff a raw frame against a named schema version."""
    if schema_version not in SCHEMAS:
        raise ValueError(f"unknown schema_version: {schema_version}")
    expected = SCHEMAS[schema_version]["expected"]
    actual = list(df.columns)
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    order_changed = (not missing and not extra
                     and actual != expected)
    report = {
        "release_id": release_id,
        "schema_version": schema_version,
        "expected_columns": len(expected),
        "actual_columns": len(actual),
        "missing_columns": missing,
        "extra_columns": extra,
        "order_changed": order_changed,
        "schema_ok": not missing and not extra,
    }
    if missing or extra:
        raise SchemaMismatch(
            f"Schema mismatch on release '{release_id}' "
            f"(schema {schema_version}): "
            f"missing={missing}, extra={extra}. "
            f"Human review required before cleaning."
        )
    return report
