# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "faces_db"
LOG_DIR = ROOT / "attendance_logs"
DB_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

META_PATH = DB_DIR / "students.json"

# -------- Jalali (Shamsi) date utilities (no extra dependency) --------
def _g2j(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) - 80 + gd + g_d_m[gm - 1]
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + (days % 31)
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + ((days - 186) % 30)
    return jy, jm, jd

def jalali_today_str():
    now = datetime.now()
    jy, jm, jd = _g2j(now.year, now.month, now.day)
    return f"{jy:04d}-{jm:02d}-{jd:02d}"

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}

def save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def student_dir(student_id: str) -> Path:
    d = DB_DIR / str(student_id)
    d.mkdir(exist_ok=True)
    return d

def feat_path(student_id: str) -> Path:
    return student_dir(student_id) / "features.npy"

def load_features(student_id: str) -> np.ndarray:
    p = feat_path(student_id)
    if p.exists():
        return np.load(p)
    return np.empty((0, 128), dtype=np.float32)

def append_feature(student_id: str, feat: np.ndarray):
    feat = feat.astype(np.float32).reshape(1, -1)
    cur = load_features(student_id)
    new = np.concatenate([cur, feat], axis=0) if cur.size else feat
    np.save(feat_path(student_id), new)

def attendance_csv_path(class_name: str) -> Path:
    safe = "".join([c if c.isalnum() or c in "_-" else "_" for c in class_name])
    j = jalali_today_str()
    return LOG_DIR / f"{safe}_{j}.csv"

def append_attendance_row(class_name: str, sid: str, name: str, similarity: float):
    p = attendance_csv_path(class_name)
    row = pd.DataFrame([{
        "timestamp": now_str(),
        "jalali_date": jalali_today_str(),
        "class": class_name,
        "student_id": str(sid),
        "name": name,
        "cosine_similarity": float(similarity),
    }])
    if p.exists():
        row.to_csv(p, mode="a", header=False, index=False)
    else:
        row.to_csv(p, index=False)
    return p

def delete_student(student_id: str, delete_logs: bool = True):
    sid = str(student_id)
    meta = load_meta()
    if sid in meta:
        del meta[sid]
        save_meta(meta)

    d = DB_DIR / sid
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

    if delete_logs:
        for p in LOG_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(p)
                if "student_id" in df.columns:
                    df2 = df[df["student_id"].astype(str) != sid]
                    if len(df2) == 0:
                        p.unlink(missing_ok=True)
                    else:
                        df2.to_csv(p, index=False)
            except Exception:
                pass

def build_last_records(hours: float = 12.0):
    cutoff = datetime.now() - timedelta(hours=float(hours))
    last = {}
    for p in LOG_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            if not {"student_id","timestamp"}.issubset(set(df.columns)):
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df[df["timestamp"] >= cutoff]
            for _, r in df.iterrows():
                sid = str(r["student_id"])
                t = r["timestamp"].to_pydatetime()
                if sid not in last or t > last[sid]:
                    last[sid] = t
        except Exception:
            continue
    return last
