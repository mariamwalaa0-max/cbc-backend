from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# Paths
# =========================================================
STAGE1_MODEL_PATH = "./cbc_stage1_model.joblib"
STAGE2_MODEL_PATH = "./cbc_stage2_model.joblib"
STAGE2_LABEL_ENCODER_PATH = "./cbc_stage2_label_encoder.joblib"

FEATURE_COLUMNS_PATH = "./cbc_feature_columns.joblib"
FEATURE_MEDIANS_PATH = "./cbc_feature_medians.joblib"

MEDICAL_ONTOLOGY_PATH = "./medical_ontology.json"

# =========================================================
# API Schemas
# =========================================================
Flag = str  # LOW | NORMAL | HIGH | UNKNOWN


class PredictRequest(BaseModel):
    cbc_values: Dict[str, float] = Field(
        ...,
        description="Structured CBC numeric values. Can be canonical keys or common aliases (HGB/HCT/PLT/Neut%...).",
    )
    cbc_flags: Optional[Dict[str, Flag]] = Field(
        default=None,
        description="Optional lab flags per feature: LOW/NORMAL/HIGH/UNKNOWN.",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (symptoms, diagnosis_hint, etc.)",
    )
    top_k: int = Field(default=3, ge=1, le=5)


class PredictResponse(BaseModel):
    stage1: Dict[str, Any]
    path: str  # "CBC" or "NON_CBC"
    top_predictions: List[Dict[str, Any]]
    ontology_support: List[Dict[str, Any]]
    urgent_attention: bool
    recommended_tests: List[Dict[str, Any]]
    specialty: List[str]
    red_flags: List[str]
    warnings: List[str]
    disclaimer: str


# =========================================================
# Feature mapping
# =========================================================
MODEL_COLS = [
    "wbc",
    "rbc",
    "hemoglobin",
    "hematocrit",
    "mcv",
    "mch",
    "mchc",
    "platelets",
    "lymp_pct",
    "neut_pct",
    "lymp_abs",
    "neut_abs",
]

ALIASES = {
    "wbc": ["wbc", "white_blood_cells", "whitebloodcells", "wbc_count"],
    "rbc": ["rbc", "red_blood_cells", "redbloodcells", "rbc_count"],
    "hemoglobin": ["hemoglobin", "hgb", "hb"],
    "hematocrit": ["hematocrit", "hct"],
    "mcv": ["mcv"],
    "mch": ["mch"],
    "mchc": ["mchc"],
    "platelets": ["platelets", "plt", "platelet_count"],
    "lymp_pct": [
        "lymp_pct",
        "lymph_pct",
        "lymphocytes_percent",
        "lymphocytes%",
        "lymph_%",
        "lymph%",
        "lymphs_pct",
    ],
    "neut_pct": ["neut_pct", "neutrophils_percent", "neutrophils%", "neut_%", "neut%", "neutro_pct"],
    "lymp_abs": ["lymp_abs", "lymph_abs", "lymphocytes_abs", "absolute_lymphocytes", "lymphocytes_absolute", "alc"],
    "neut_abs": ["neut_abs", "neutrophils_abs", "absolute_neutrophils", "neutrophils_absolute", "anc"],
}


def _norm(s: str) -> str:
    """Normalize string keys for matching"""
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "pct")
    )


ALIAS_TO_CANON: Dict[str, str] = {}
for canon, names in ALIASES.items():
    for n in names:
        ALIAS_TO_CANON[_norm(n)] = canon


def map_input_to_model_features(raw_values: Dict[str, Any]) -> Dict[str, float]:
    """
    Converts user keys -> canonical model keys.
    Ignores unknown keys.
    """
    out: Dict[str, float] = {}
    for k, v in (raw_values or {}).items():
        nk = _norm(str(k))
        canon = ALIAS_TO_CANON.get(nk)

        # Small heuristics
        if canon is None:
            canon = ALIAS_TO_CANON.get(nk.replace("percent", "pct"))

        if canon is None:
            continue

        try:
            out[canon] = float(v)
        except Exception:
            continue
    return out


def validate_numeric_ranges(values: Dict[str, float]) -> List[str]:
    """
    Validate that numeric values are within reasonable clinical ranges.
    Returns list of warnings.
    """
    warnings = []
    
    ranges = {
        "wbc": (0.1, 100),
        "rbc": (1.0, 8.0),
        "hemoglobin": (3.0, 22.0),
        "hematocrit": (10.0, 70.0),
        "mcv": (50.0, 130.0),
        "mch": (20.0, 40.0),
        "mchc": (25.0, 40.0),
        "platelets": (5.0, 1000.0),
        "lymp_pct": (0.0, 100.0),
        "neut_pct": (0.0, 100.0),
        "lymp_abs": (0.0, 20.0),
        "neut_abs": (0.0, 50.0),
    }
    
    for key, (min_val, max_val) in ranges.items():
        if key in values:
            val = values[key]
            if val < min_val or val > max_val:
                warnings.append(
                    f"{key.upper()} value {val} is outside typical range ({min_val}-{max_val})"
                )
    
    return warnings


def build_feature_vector(
    raw_values: Dict[str, Any], 
    feature_columns: List[str], 
    medians: Dict[str, float]
) -> pd.DataFrame:
    """
    - Map raw inputs -> canonical columns
    - Create a 1-row dataframe with exactly the required feature_columns
    - Fill missing with medians
    """
    mapped = map_input_to_model_features(raw_values)
    row: Dict[str, float] = {}

    for col in feature_columns:
        if col in mapped:
            row[col] = mapped[col]
        else:
            row[col] = float(medians.get(col, 0.0))

    df = pd.DataFrame([row], columns=feature_columns)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# =========================================================
# App + Lifespan loading
# =========================================================
ARTIFACTS: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ARTIFACTS["stage1_model"] = joblib.load(STAGE1_MODEL_PATH)
        ARTIFACTS["stage2_model"] = joblib.load(STAGE2_MODEL_PATH)
        ARTIFACTS["label_encoder"] = joblib.load(STAGE2_LABEL_ENCODER_PATH)

        ARTIFACTS["feature_columns"] = joblib.load(FEATURE_COLUMNS_PATH)
        ARTIFACTS["feature_medians"] = joblib.load(FEATURE_MEDIANS_PATH)

        with open(MEDICAL_ONTOLOGY_PATH, "r", encoding="utf-8") as f:
            ARTIFACTS["medical_ontology"] = json.load(f)

        # Load stage1 threshold if available
        ARTIFACTS["stage1_threshold"] = 0.6  # Default, override if stored

        # Optional sanity check
        cols = list(ARTIFACTS["feature_columns"])
        if cols != MODEL_COLS:
            print("⚠️  WARNING: Loaded feature_columns != expected MODEL_COLS")
            print("Loaded :", cols)
            print("Expect :", MODEL_COLS)

        print("✅ Artifacts loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

    yield

    ARTIFACTS.clear()
    print("🧹 Artifacts cleared")


app = FastAPI(
    title="CBC Decision Support API",
    version="1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  "http://localhost:4173",      
  "http://127.0.0.1:4173",      
  "http://localhost:8080",
  "http://127.0.0.1:8080",
  "http://localhost:8081",
  "http://127.0.0.1:8081",
  "http://localhost:8082",
  "http://127.0.0.1:8082",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(ARTIFACTS.keys())}


# =========================================================
# Predict Endpoint
# =========================================================
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if not ARTIFACTS:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    warnings = []

    # ======================================================
    # 1️⃣ Validate input
    # ======================================================
    
    # Map to canonical keys first
    values_mapped = map_input_to_model_features(req.cbc_values)
    
    # Validate numeric ranges
    range_warnings = validate_numeric_ranges(values_mapped)
    warnings.extend(range_warnings)
    
    # Check minimum required fields
    required = ["hemoglobin", "wbc", "platelets", "mcv"]
    missing_required = [f for f in required if f not in values_mapped]

    if missing_required:
        warnings.append(
            f"Missing key CBC fields: {', '.join(missing_required)}. "
            "Predictions may be unreliable."
        )

    # ======================================================
    # 2️⃣ Align features
    # ======================================================
    feature_columns = ARTIFACTS["feature_columns"]
    medians = ARTIFACTS["feature_medians"]

    X = build_feature_vector(values_mapped, feature_columns, medians)

    # ======================================================
    # 3️⃣ Stage-1 Gate (CBC relevance)
    # ======================================================
    stage1_model = ARTIFACTS["stage1_model"]
    stage1_threshold = ARTIFACTS.get("stage1_threshold", 0.6)

    try:
        proba = (
            stage1_model.predict_proba(X)[0][1]
            if hasattr(stage1_model, "predict_proba")
            else float(stage1_model.predict(X)[0])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage-1 prediction failed: {e}")

    cbc_related = proba >= stage1_threshold

    stage1_out = {
        "cbc_related_probability": round(float(proba), 3),
        "cbc_related": bool(cbc_related),
        "threshold": float(stage1_threshold),
        "note": "Screening gate only, not diagnostic",
    }

    ontology = ARTIFACTS["medical_ontology"]

    # ======================================================
    # 4️⃣ NON-CBC PATH (Ontology routing)
    # ======================================================
    if not cbc_related:

        non_cbc_rules = ontology.get("non_cbc_related", {})
        matched = []

        # Extract text for matching from context
        diagnosis_hint = ""
        if req.context:
            diagnosis_hint = str(req.context.get("diagnosis_hint", "")).lower()
            # You can also check other context fields as needed
            for key, value in req.context.items():
                if isinstance(value, str):
                    diagnosis_hint += " " + value.lower()

        # Match patterns
        for rule in non_cbc_rules.get("patterns", []):
            keywords = rule.get("keywords", [])
            if any(k.lower() in diagnosis_hint for k in keywords):
                matched.append(rule)

        # If no matches, use default
        if not matched and non_cbc_rules.get("default"):
            matched.append(non_cbc_rules["default"])

        return PredictResponse(
            stage1=stage1_out,
            path="NON_CBC",
            top_predictions=[],
            ontology_support=[],
            urgent_attention=any(r.get("red_flag", False) for r in matched),
            recommended_tests=[
                {
                    "test": t,
                    "reason": rule.get("reason", ""),
                    "priority": rule.get("priority", 2),
                }
                for rule in matched
                for t in rule.get("recommended_tests", [])
            ],
            specialty=list({r.get("specialty") for r in matched if r.get("specialty")}),
            red_flags=[r.get("red_flag_text", "") for r in matched if r.get("red_flag")],
            warnings=warnings,
            disclaimer=ontology.get(
                "global_disclaimer",
                "Clinical decision support only. Not a diagnosis. Always consult a healthcare provider."
            ),
        )

    # ======================================================
    # 5️⃣ CBC PATH — Stage-2 Classification
    # ======================================================
    stage2_model = ARTIFACTS["stage2_model"]
    label_encoder = ARTIFACTS["label_encoder"]

    try:
        probs = stage2_model.predict_proba(X)[0]
        # Get indices of the top-k highest probabilities
        top_indices_local = np.argsort(probs)[::-1][: req.top_k]
        
        # Get the internal class labels (integers) from the model
        model_classes = stage2_model.classes_
        predicted_class_codes = [model_classes[i] for i in top_indices_local]

        # Decode integer labels to string names using the label encoder
        decoded_labels = label_encoder.inverse_transform(predicted_class_codes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage-2 prediction failed: {e}")

    top_preds = [
        {
            "rank": i + 1,
            "condition": str(decoded_labels[i]),
            "probability": float(probs[top_indices_local[i]]),
            "probability_percent": round(float(probs[top_indices_local[i]]) * 100, 2),
        }
        for i in range(len(decoded_labels))
    ]

    # ======================================================
    # 6️⃣ CBC Ontology enrichment
    # ======================================================
    enriched = []
    all_red_flags = []
    all_specialties = set()
    all_tests = []

    for p in top_preds:
        key = _norm(p["condition"])
        info = ontology.get("cbc_conditions", {}).get(key, {})

        # Extract confirmatory tests with priority
        conf_tests = info.get("confirmatory_tests", [])
        if isinstance(conf_tests, list):
            for test in conf_tests:
                if isinstance(test, dict):
                    all_tests.append(test)
                else:
                    # If it's just a string
                    all_tests.append({
                        "test": test,
                        "priority": 2,
                        "reason": f"For evaluating {p['condition']}"
                    })

        # Collect red flags
        red_flags = info.get("red_flags", [])
        all_red_flags.extend(red_flags)

        # Collect specialties
        specialty = info.get("specialty")
        if specialty:
            all_specialties.add(specialty)

        enriched.append({
            **p,
            "likely_causes": info.get("likely_causes", []),
            "confirmatory_tests": conf_tests,
            "specialty": specialty,
            "red_flags": red_flags,
        })

    # Deduplicate and sort tests by priority
    seen_tests = set()
    unique_tests = []
    for test in sorted(all_tests, key=lambda x: x.get("priority", 99)):
        test_name = test.get("test")
        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            unique_tests.append(test)

    urgent = len(all_red_flags) > 0

    # ======================================================
    # 7️⃣ Final Response
    # ======================================================
    return PredictResponse(
        stage1=stage1_out,
        path="CBC",
        top_predictions=enriched,
        ontology_support=[],  # Can be extended with indicator-based support
        urgent_attention=urgent,
        recommended_tests=unique_tests,
        specialty=list(all_specialties),
        red_flags=list(set(all_red_flags)),
        warnings=warnings,
        disclaimer=ontology.get(
            "global_disclaimer",
            "Clinical decision support only. Not a diagnosis. Always consult a healthcare provider."
        ),
    )


# =========================================================
# Optional: Add endpoint to get ontology conditions
# =========================================================
@app.get("/conditions")
def get_conditions():
    """Return list of all CBC conditions in ontology"""
    if not ARTIFACTS or "medical_ontology" not in ARTIFACTS:
        raise HTTPException(status_code=500, detail="Ontology not loaded")
    
    ontology = ARTIFACTS["medical_ontology"]
    conditions = ontology.get("cbc_conditions", {})
    
    return {
        "conditions": list(conditions.keys()),
        "count": len(conditions)
    }