# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.json"

DEFAULTS = {
    "default_class_name": "OS_Lab",
    "classes": ["OS_Lab"],
    "default_similarity_threshold": 0.50,
    "default_face_score_threshold": 0.90,
    "capture_interval_sec": 2.0,
    "enroll_samples_target": 10,
    "cooldown_hours": 24.0,
}

def load_config():
    if CFG_PATH.exists():
        try:
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            cfg = DEFAULTS.copy()
            for k, v in DEFAULTS.items():
                cfg[k] = data.get(k, v)
            cls = cfg.get("classes") or []
            cls = [str(x).strip() for x in cls if str(x).strip()]
            if not cls:
                cls = [cfg.get("default_class_name","OS_Lab")]
            cfg["classes"] = list(dict.fromkeys(cls))
            if cfg.get("default_class_name") not in cfg["classes"]:
                cfg["classes"].insert(0, cfg.get("default_class_name","OS_Lab"))
            return cfg
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_config(cfg: dict):
    data = DEFAULTS.copy()
    for k in DEFAULTS.keys():
        if k in cfg:
            data[k] = cfg[k]
    cls = data.get("classes") or []
    cls = [str(x).strip() for x in cls if str(x).strip()]
    cls = list(dict.fromkeys(cls))
    if not cls:
        cls = [data.get("default_class_name","OS_Lab")]
    data["classes"] = cls
    if data.get("default_class_name") not in data["classes"]:
        data["classes"].insert(0, data.get("default_class_name","OS_Lab"))
    CFG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return CFG_PATH
