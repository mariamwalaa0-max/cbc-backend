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

MEDICAL_ONTOLOGY_PATH = "./medical_ontology_cbc_only.json"

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
# CBC Ontology scoring helpers
# =========================================================

ONTOLOGY_RANGES = {
    "hemoglobin": (12.0, 16.0),
    "wbc": (4.0, 11.0),
    "platelets": (150.0, 450.0),
    "mcv": (80.0, 100.0),
    "mch": (27.0, 33.0),
    "mchc": (32.0, 36.0),
    "neut_abs": (1.5, 7.5),
    "lymp_abs": (1.0, 4.0),
    "hematocrit": (36.0, 46.0),
    "rbc": (3.8, 5.2),
}

def normalize_ontology_feature(feature: str) -> str:
    f = _norm(feature)
    mapping = {
        "hgb": "hemoglobin",
        "hb": "hemoglobin",
        "wbc": "wbc",
        "plt": "platelets",
        "platelet": "platelets",
        "platelets": "platelets",
        "mcv": "mcv",
        "mch": "mch",
        "mchc": "mchc",
        "hct": "hematocrit",
        "hematocrit": "hematocrit",
        "rbc": "rbc",
        "neutrophils": "neut_abs",
        "neut": "neut_abs",
        "absolute_neutrophils": "neut_abs",
        "anc": "neut_abs",
        "lymphocytes": "lymp_abs",
        "lymph": "lymp_abs",
        "absolute_lymphocytes": "lymp_abs",
        "alc": "lymp_abs",
        "cbc_overall": "cbc_overall",
        "cbc_scope": "cbc_scope",
    }
    return mapping.get(f, f)

def get_flag_from_value(feature: str, value: Optional[float]) -> str:
    if value is None:
        return "UNKNOWN"

    if feature == "cbc_scope":
        return "INSUFFICIENT"
    if feature == "cbc_overall":
        return "NORMAL"

    rng = ONTOLOGY_RANGES.get(feature)
    if not rng:
        return "UNKNOWN"

    low, high = rng
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"

def match_direction(flag: str, direction: str, feature: str, value: Optional[float]) -> bool:
    direction = direction.lower().strip()
    flag = (flag or "UNKNOWN").upper()

    if direction == "low":
        return flag == "LOW"
    if direction == "high":
        return flag == "HIGH"
    if direction == "normal":
        return flag == "NORMAL"
    if direction == "low_or_normal":
        return flag in ("LOW", "NORMAL")
    if direction == "high_or_normal":
        return flag in ("HIGH", "NORMAL")
    if direction == "high_or_low":
        return flag in ("HIGH", "LOW")
    if direction == "very_high_or_very_low":
        if value is None:
            return False
        if feature == "wbc":
            return value < 2.0 or value > 30.0
        return flag in ("HIGH", "LOW")
    if direction == "support_only":
        return True
    return False

def build_flag_map(
    cbc_values: Dict[str, Any],
    cbc_flags: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    flag_map: Dict[str, str] = {}

    if cbc_flags:
        for k, v in cbc_flags.items():
            feat = normalize_ontology_feature(k)
            flag_map[feat] = str(v).upper().strip()

    for k, v in (cbc_values or {}).items():
        feat = normalize_ontology_feature(k)
        try:
            num = float(v)
        except Exception:
            continue

        if feat not in flag_map:
            flag_map[feat] = get_flag_from_value(feat, num)

    if "cbc_scope" not in flag_map:
        flag_map["cbc_scope"] = "INSUFFICIENT"

    major = ["hemoglobin", "wbc", "platelets", "mcv"]
    if all(flag_map.get(f) == "NORMAL" for f in major if f in flag_map):
        flag_map["cbc_overall"] = "NORMAL"

    return flag_map

def score_cbc_ontology(
    ontology: Dict[str, Any],
    cbc_values: Dict[str, Any],
    cbc_flags: Optional[Dict[str, str]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    cbc_section = ontology.get("cbc_related", {})
    flag_map = build_flag_map(cbc_values, cbc_flags)

    numeric_values = {}
    for k, v in (cbc_values or {}).items():
        try:
            numeric_values[normalize_ontology_feature(k)] = float(v)
        except Exception:
            continue

    results = []

    for condition_name, condition_info in cbc_section.items():
        rules = condition_info.get("pattern_rules", [])
        if not rules:
            continue

        matched_rules = []
        total_weight = 0.0
        matched_weight = 0.0

        for rule in rules:
            raw_feature = str(rule.get("feature", ""))
            direction = str(rule.get("direction", ""))
            reason = str(rule.get("reason", ""))
            weight = float(rule.get("weight", 1.0))

            feature = normalize_ontology_feature(raw_feature)
            feature_flag = flag_map.get(feature, "UNKNOWN")
            value = numeric_values.get(feature)

            total_weight += weight

            if match_direction(feature_flag, direction, feature, value):
                matched_weight += weight
                matched_rules.append({
                    "feature": raw_feature,
                    "direction": direction,
                    "reason": reason,
                    "weight": weight,
                })

        if total_weight == 0:
            continue

        score = matched_weight / total_weight
        min_score = float(condition_info.get("min_score", 0.3))

        specialty_info = condition_info.get("suggested_specialty", {})

        result_item = {
            "condition": condition_name,
            "score": round(score, 3),
            "score_percent": round(score * 100, 2),
            "matched_rules": matched_rules,
            "likely_causes": condition_info.get("likely_causes", []),
            "confirmatory_tests": condition_info.get("confirmatory_tests", []),
            "specialty": specialty_info.get("name") if isinstance(specialty_info, dict) else None,
            "red_flags": condition_info.get("red_flags", []),
            "passed_min_score": score >= min_score,
            "min_score": min_score,
        }

        results.append(result_item)

    # sort all candidates by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # strong matches only
    strong_results = [r for r in results if r["passed_min_score"]]

    # if there are strong matches, return them
    if strong_results:
        return strong_results[:top_k]

    # otherwise return best weak matches instead of empty list
    fallback_results = results[:top_k]
    for item in fallback_results:
        item["weak_match"] = True

    return fallback_results
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
        with open("cbc_model_metadata.json") as f:
          metadata = json.load(f)
          ARTIFACTS["stage1_threshold"] = metadata.get("stage1_threshold", 0.6) # Default, override if stored

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
  "https://diagnoaii.netlify.app",
  "https://diagnoaiii.netlify.app",
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
    print("===== PREDICT HIT =====")
    print("Request:", req.dict())
    if not ARTIFACTS:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    warnings = []

    # ======================================================
    # 1️⃣ Validate input
    # ======================================================
    
    # Map to canonical keys first
    values_mapped = map_input_to_model_features(req.cbc_values)
    print("Mapped values:", values_mapped)
    
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
    
    # CBC ontology scoring is always available, even if Stage-1 says NOT CBC related
    cbc_ontology_predictions = score_cbc_ontology(
        ontology=ontology,
        cbc_values=req.cbc_values,
        cbc_flags=req.cbc_flags,
        top_k=req.top_k
    )
    if not cbc_related:
        all_tests = []
        all_red_flags = []
        all_specialties = set()

        for item in cbc_ontology_predictions:
            for test in item.get("confirmatory_tests", []):
                if isinstance(test, dict):
                    all_tests.append(test)

            spec = item.get("specialty")
            if spec:
                all_specialties.add(spec)

            all_red_flags.extend(item.get("red_flags", []))

        seen_tests = set()
        unique_tests = []
        for t in sorted(all_tests, key=lambda x: x.get("priority", 99)):
            name = t.get("test")
            if name and name not in seen_tests:
                seen_tests.add(name)
                unique_tests.append(t)
        if cbc_ontology_predictions and all(item.get("weak_match") for item in cbc_ontology_predictions):
            warnings.append("No strong CBC ontology match found; showing closest weak matches.")

        return PredictResponse(
            stage1=stage1_out,
            path="CBC_ONTOLOGY",
            top_predictions=[],
            ontology_support=cbc_ontology_predictions,
            urgent_attention=len(all_red_flags) > 0,
            recommended_tests=unique_tests,
            specialty=list(all_specialties),
            red_flags=list(set(all_red_flags)),
            warnings=warnings + [
                "Stage-1 marked this case as NOT CBC-related, so CBC ontology scoring was used."
            ],
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

        confidence_threshold = 0.60
        max_prob = float(np.max(probs))
        low_confidence = max_prob < confidence_threshold

        if low_confidence:
            warnings.append(
                f"Low confidence ML prediction (max={round(max_prob*100,2)}%). "
                "Using CBC ontology scoring as fallback."
            )

        top_indices_local = np.argsort(probs)[::-1][: req.top_k]

        model_classes = stage2_model.classes_
        predicted_class_codes = [model_classes[i] for i in top_indices_local]

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

    for p in top_preds:
        p["low_confidence"] = bool(low_confidence)
        p["max_probability"] = round(max_prob, 4)
        p["confidence_threshold"] = confidence_threshold

    # ======================================================
    # 6️⃣ CBC Ontology enrichment
    # ======================================================
    enriched = []
    all_red_flags = []
    all_specialties = set()
    all_tests = []

    for p in top_preds:
        key = _norm(p["condition"])
        info = ontology.get("cbc_related", {}).get(key, {})

        conf_tests = info.get("confirmatory_tests", [])
        if isinstance(conf_tests, list):
            for test in conf_tests:
                if isinstance(test, dict):
                    all_tests.append(test)
                else:
                    all_tests.append({
                        "test": test,
                        "priority": 2,
                        "reason": f"For evaluating {p['condition']}"
                    })

        red_flags = info.get("red_flags", [])
        all_red_flags.extend(red_flags)

        specialty_info = info.get("suggested_specialty", {})
        specialty = specialty_info.get("name") if isinstance(specialty_info, dict) else None
        if specialty:
            all_specialties.add(specialty)

        enriched.append({
            **p,
            "likely_causes": info.get("likely_causes", []),
            "confirmatory_tests": conf_tests,
            "specialty": specialty,
            "red_flags": red_flags,
        })

    seen_tests = set()
    unique_tests = []
    for test in sorted(all_tests, key=lambda x: x.get("priority", 99)):
        test_name = test.get("test")
        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            unique_tests.append(test)

    urgent = len(all_red_flags) > 0

    ontology_support = []
    if low_confidence:
        ontology_support = cbc_ontology_predictions
        if ontology_support and all(item.get("weak_match") for item in ontology_support):
            warnings.append("Ontology fallback found only weak CBC matches.")
        for item in ontology_support:
            for test in item.get("confirmatory_tests", []):
                if isinstance(test, dict):
                    all_tests.append(test)

            spec = item.get("specialty")
            if spec:
                all_specialties.add(spec)

            all_red_flags.extend(item.get("red_flags", []))

        seen_tests = set()
        unique_tests = []
        for test in sorted(all_tests, key=lambda x: x.get("priority", 99)):
            test_name = test.get("test")
            if test_name and test_name not in seen_tests:
                seen_tests.add(test_name)
                unique_tests.append(test)

    # ======================================================
    # 7️⃣ Final Response
    # ======================================================
    return PredictResponse(
        stage1=stage1_out,
        path="CBC",
        top_predictions=enriched,
        ontology_support=ontology_support,
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
    conditions = ontology.get("cbc_related", {})
    
    return {
        "conditions": list(conditions.keys()),
        "count": len(conditions)
    }