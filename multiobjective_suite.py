#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUITE MULTIOBJETIVO — Framework Multimodal · Gestión de Tráfico  v2        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHIVOS DEL SISTEMA:                                                       ║
║    multiobjective_suite.py   ← este script (todo-en-uno)                    ║
║    multiobjective_db.json    ← BD: configs + runs (texto plano, auto)       ║
║    results_XXXXXXXX.csv      ← generado por diagnostic_validation_full_trace ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PANTALLAS  (Tab para navegar):                                              ║
║  [1] Config      — objetivos/pesos/métodos/rho  [A]Objetivos [B]Métricas    ║
║  [2] BD          — configs (sin CSV) · runs registradas · cargar/borrar     ║
║  [3] Ejecutar    — elegir config BD + CSV → motor interno → registra run     ║
║  [4] Gráficos    — elegir run desde BD → 8 gráficos ASCII                   ║
║  [5] Tablas      — tabla paginada · [Enter] detalle con fórmulas completas  ║
║  [6] Repositorio — campos CSV por fase · % vacíos · sort · paginación       ║
║  [7] Estadística — distribución · normalidad · pruebas param/no-param       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO:                                                                        ║
║    python3 multiobjective_suite.py                                           ║
║    python3 multiobjective_suite.py --input results_XXXXXXXX.csv             ║
║    python3 multiobjective_suite.py --input results.csv --config 3           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse, curses, json, os, sys, threading, traceback, warnings
from copy      import deepcopy
from datetime  import datetime
from pathlib   import Path

# ── TERM fix (RunPod / Docker / tmux) ─────────────────────────────────────────
if os.environ.get("TERM","") not in ("xterm-256color","screen-256color","tmux-256color"):
    os.environ["TERM"] = "xterm-256color"

# ── Dependencias numéricas ────────────────────────────────────────────────────
try:
    import numpy  as np
    import pandas as pd
    _NUM = True
except ImportError:
    _NUM = False

try:
    from scipy.stats import spearmanr as _spr
    def _spearman(a, b): r,_ = _spr(a,b); return float(r)
except ImportError:
    def _spearman(a, b):       # fallback puro Python
        n = len(a)
        if n < 2: return 0.0
        ra = [sorted(range(n), key=lambda i: a[i]).index(i) for i in range(n)]
        rb = [sorted(range(n), key=lambda i: b[i]).index(i) for i in range(n)]
        d2 = sum((ra[i]-rb[i])**2 for i in range(n))
        return 1 - 6*d2/(n*(n*n-1))

# ══════════════════════════════════════════════════════════════════════════════
# § 1  CATÁLOGO DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
CATALOG = {
    "Fase 1 · Detección": [
        {"col":"det_f1",         "lbl":"F1 Score",        "unit":"[0-1]","dir":"max","rec":True },
        {"col":"det_map_50",     "lbl":"mAP@50",          "unit":"[0-1]","dir":"max","rec":True },
        {"col":"det_precision",  "lbl":"Precision",       "unit":"[0-1]","dir":"max","rec":False},
        {"col":"det_recall",     "lbl":"Recall",          "unit":"[0-1]","dir":"max","rec":False},
        {"col":"det_iou",        "lbl":"IoU",             "unit":"[0-1]","dir":"max","rec":False},
        {"col":"det_latency_ms", "lbl":"Latencia Det.",   "unit":"ms",  "dir":"min","rec":False},
        {"col":"det_objects",    "lbl":"Objetos detect.", "unit":"n",   "dir":"max","rec":False},
    ],
    "Fase 2 · Clasificación": [
        {"col":"cls_accuracy",   "lbl":"Accuracy",        "unit":"[0-1]","dir":"max","rec":True },
        {"col":"cls_f1",         "lbl":"F1 Score",        "unit":"[0-1]","dir":"max","rec":True },
        {"col":"cls_precision",  "lbl":"Precision",       "unit":"[0-1]","dir":"max","rec":False},
        {"col":"cls_recall",     "lbl":"Recall",          "unit":"[0-1]","dir":"max","rec":False},
        {"col":"cls_confidence", "lbl":"Confidence",      "unit":"[0-1]","dir":"max","rec":False},
        {"col":"cls_latency_ms", "lbl":"Latencia Cls.",   "unit":"ms",  "dir":"min","rec":False},
    ],
    "Fase 3 · VLM": [
        {"col":"vlm_rougeL_f",   "lbl":"ROUGE-L F",       "unit":"[0-1]","dir":"max","rec":True },
        {"col":"vlm_meteor",     "lbl":"METEOR",          "unit":"[0-1]","dir":"max","rec":True },
        {"col":"vlm_bleu4",      "lbl":"BLEU-4",          "unit":"[0-1]","dir":"max","rec":False},
        {"col":"vlm_bleu1",      "lbl":"BLEU-1",          "unit":"[0-1]","dir":"max","rec":False},
        {"col":"vlm_rouge1_f",   "lbl":"ROUGE-1 F",       "unit":"[0-1]","dir":"max","rec":False},
        {"col":"vlm_clipscore",  "lbl":"CLIPScore",       "unit":"[0-1]","dir":"max","rec":False},
        {"col":"vlm_perplexity", "lbl":"Perplexity",      "unit":"sco", "dir":"min","rec":False},
        {"col":"vlm_latency_ms", "lbl":"Latencia VLM",    "unit":"ms",  "dir":"min","rec":False},
    ],
    "Fase 4 · LLM": [
        {"col":"llm_rougeL_f",   "lbl":"ROUGE-L F",       "unit":"[0-1]","dir":"max","rec":True },
        {"col":"llm_meteor",     "lbl":"METEOR",          "unit":"[0-1]","dir":"max","rec":True },
        {"col":"llm_bleu4",      "lbl":"BLEU-4",          "unit":"[0-1]","dir":"max","rec":False},
        {"col":"llm_rouge1_f",   "lbl":"ROUGE-1 F",       "unit":"[0-1]","dir":"max","rec":False},
        {"col":"llm_perplexity", "lbl":"Perplexity",      "unit":"sco", "dir":"min","rec":False},
        {"col":"llm_latency_ms", "lbl":"Latencia LLM",    "unit":"ms",  "dir":"min","rec":False},
    ],
    "Global · Sistema": [
        {"col":"total_latency_ms","lbl":"Latencia Total", "unit":"ms",  "dir":"min","rec":True },
        {"col":"vram_used_mb",    "lbl":"VRAM usada",     "unit":"MB",  "dir":"min","rec":False},
        {"col":"ram_used_gb",     "lbl":"RAM usada",      "unit":"GB",  "dir":"min","rec":False},
    ],
}
PH_ICON = {"Fase 1 · Detección":"⬡","Fase 2 · Clasificación":"◈",
           "Fase 3 · VLM":"◎","Fase 4 · LLM":"◉","Global · Sistema":"◇"}
METHODS = ["WSum","WProd","Tcheby","ASF"]
M_DESC  = {
    "WSum":  "Suma lineal ponderada      (↑ mayor=mejor)",
    "WProd": "Media geométrica pond.     (↑ mayor=mejor)",
    "Tcheby":"Dist. Chebyshev al ideal   (↓ menor=mejor)",
    "ASF":   "Achievement Scalar. Fn.   (↓ menor=mejor)",
}

# ══════════════════════════════════════════════════════════════════════════════
# § 2  BASE DE DATOS — JSON texto plano  (configs + runs separados)
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = Path("multiobjective_db.json")

def _db_read() -> dict:
    if DB_PATH.exists():
        try: return json.loads(DB_PATH.read_text("utf-8"))
        except Exception: pass
    return {"next_config_id": 1, "next_run_id": 1, "configs": {}, "runs": {}}

def _db_write(db: dict):
    DB_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False), "utf-8")

def _db_migrate(db: dict) -> dict:
    """Migra BD antigua (next_id / sin runs) al nuevo esquema."""
    if "next_id" in db and "next_config_id" not in db:
        db["next_config_id"] = db.pop("next_id")
    if "next_run_id" not in db:
        db["next_run_id"] = 1
    if "runs" not in db:
        db["runs"] = {}
        # Mover last_result de configs → runs si existe
        nrid = 1
        for k, cfg in db["configs"].items():
            lr = cfg.pop("last_result", "")
            ic = cfg.pop("input_csv", "")
            if lr:
                db["runs"][str(nrid)] = {
                    "id": nrid, "config_id": cfg["id"],
                    "config_name": cfg["name"],
                    "input_csv": ic, "out_dir": lr,
                    "ran_at": cfg.get("created_at",""),
                    "n_rows": 0,
                }
                nrid += 1
        db["next_run_id"] = nrid
    return db

def _db_read_m() -> dict:
    db = _db_read()
    return _db_migrate(db)

# ── CONFIGS (sin CSV) ─────────────────────────────────────────────────────────
def db_save(name: str, desc: str, state: dict) -> int:
    """Guarda solo la configuración (pesos, métodos, objetivos, rho) — SIN csv."""
    db  = _db_read_m()
    nid = db["next_config_id"]
    db["configs"][str(nid)] = {
        "id": nid, "name": name, "description": desc,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "objectives": deepcopy(state["objectives"]),
        "methods":    dict(state["methods"]),
        "rho":        state["rho"],
    }
    db["next_config_id"] = nid + 1
    _db_write(db)
    return nid

def db_list() -> list:
    return sorted(_db_read_m()["configs"].values(), key=lambda x: x["id"], reverse=True)

def db_load(config_id: int) -> dict:
    rec = _db_read_m()["configs"].get(str(config_id))
    if not rec: return {}
    loaded = {
        "objectives":  deepcopy(rec["objectives"]),
        "methods":     dict(rec["methods"]),
        "rho":         rec["rho"],
        "input_csv":   "",        # configs no llevan csv
        "last_result": "",
        "_db_id":      rec["id"],
        "_db_name":    rec["name"],
    }
    _sync_rec_to_catalog(loaded["objectives"])
    return loaded

def db_delete(config_id: int):
    db = _db_read_m()
    db["configs"].pop(str(config_id), None)
    _db_write(db)

# ── RUNS (ejecuciones: config + csv + out_dir) ────────────────────────────────
def run_save(config_id: int, config_name: str,
             input_csv: str, out_dir: str, n_rows: int) -> int:
    db  = _db_read_m()
    nid = db["next_run_id"]
    db["runs"][str(nid)] = {
        "id":          nid,
        "config_id":   config_id,
        "config_name": config_name,
        "input_csv":   input_csv,
        "out_dir":     out_dir,
        "ran_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows":      n_rows,
    }
    db["next_run_id"] = nid + 1
    _db_write(db)
    return nid

def run_list() -> list:
    return sorted(_db_read_m()["runs"].values(), key=lambda x: x["id"], reverse=True)

def run_get(run_id: int) -> dict:
    return _db_read_m()["runs"].get(str(run_id), {})

def run_delete(run_id: int):
    db = _db_read_m()
    db["runs"].pop(str(run_id), None)
    _db_write(db)

def _sync_rec_to_catalog(objectives: list):
    """Propaga el campo rec de los objetivos guardados al CATALOG en memoria."""
    rec_map = {o["col"]: o.get("rec", False) for o in objectives}
    for ph, mets in CATALOG.items():
        for m in mets:
            if m["col"] in rec_map:
                m["rec"] = rec_map[m["col"]]

# ══════════════════════════════════════════════════════════════════════════════
# § 3  MOTOR DE CÁLCULO MULTIOBJETIVO (todo interno, sin scripts externos)
# ══════════════════════════════════════════════════════════════════════════════
def _norm_matrix(df, objs):
    """Normaliza hacia maximización → todos los valores en [0,1], 1=mejor."""
    cols = []
    for o in objs:
        col = o["col"]
        v   = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values \
              if col in df.columns else np.zeros(len(df))
        mn, mx = v.min(), v.max()
        r = mx - mn if mx != mn else 1e-9
        norm = (v - mn) / r if o["dir"] == "max" else (mx - v) / r
        cols.append(norm)
    return np.column_stack(cols) if cols else np.zeros((len(df), 1))

def _pareto_mask(F):
    """Máscara bool de no-dominados (maximización)."""
    n = len(F)
    ok = np.ones(n, bool)
    for i in range(n):
        if not ok[i]: continue
        dom = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        dom[i] = False
        ok[dom] = False
        if ok[i] and np.any(np.all(F[ok & ~dom] >= F[i], axis=1) &
                             np.any(F[ok & ~dom] >  F[i], axis=1)):
            ok[i] = False
    return ok

def _pareto_levels(F):
    n = len(F); lv = np.zeros(n, int); rem = np.arange(n); lev = 1
    while len(rem):
        m = _pareto_mask(F[rem]); lv[rem[m]] = lev
        rem = rem[~m]; lev += 1
    return lv

def _hv2d(pts, ref):
    p = pts[np.all(pts < ref, axis=1)]
    if not len(p): return 0.0
    p = p[p[:,0].argsort()]; hv = 0.0; py = ref[1]
    for pt in p[::-1]:
        if pt[1] < py: hv += (ref[0]-pt[0])*(py-pt[1]); py = pt[1]
    return hv

def _hv_mc(pts, ref, n=40_000, seed=42):
    rng = np.random.default_rng(seed)
    p   = pts[np.all(pts <= ref, axis=1)]
    if not len(p): return 0.0
    lo  = p.min(0); vol = float(np.prod(ref - lo))
    s   = rng.uniform(lo, ref, (n, pts.shape[1]))
    cov = np.zeros(n, bool)
    for pt in p: cov |= np.all(s <= pt, axis=1)
    return vol * float(cov.mean())

def _hv_per_row(F, ref):
    n, d = F.shape; hv = np.zeros(n)
    for i in range(n):
        p = F[i:i+1]
        hv[i] = _hv2d(p, ref) if d == 2 else _hv_mc(p, ref, 6_000)
    return hv

def run_analysis(csv_path: str, state: dict, log_cb=None) -> dict:
    """Motor principal. Devuelve {'df', 'summary', 'out_dir'} o {} en error."""
    def L(msg):
        if log_cb: log_cb(msg)

    if not _NUM:
        L("✗ numpy/pandas no instalados: pip install numpy pandas scipy"); return {}

    L(f"📂 Cargando CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        L(f"✗ {e}"); return {}
    L(f"   {len(df)} filas · {len(df.columns)} columnas")

    objs    = [o for o in state["objectives"] if o["enabled"]]
    weights = np.array([o["weight"] for o in objs], float)
    if not len(objs):
        L("✗ Sin objetivos activos"); return {}

    miss = [o["col"] for o in objs if o["col"] not in df.columns]
    if miss: L(f"⚠  Columnas ausentes (se usará 0): {miss}")

    L(f"\n🔢 {len(objs)} objetivos  Sw={weights.sum():.4f}")
    for o in objs:
        L(f"   {'↑' if o['dir']=='max' else '↓'} {o['lbl']:<22} w={o['weight']:.4f}  ({o['col']})")

    # Normalización
    L("\n📐 Normalizando...")
    F = _norm_matrix(df, objs)
    for i, o in enumerate(objs):
        df[f"norm_{o['col']}"] = F[:, i]
    norm_cols = [f"norm_{o['col']}" for o in objs]

    # Escalarización
    active = [m for m, v in state["methods"].items() if v]
    rho    = state["rho"]
    L(f"⚙  Métodos: {active}  ρ={rho}")
    eps = 1e-9
    if "WSum"   in active: df["WSum"]   = F @ weights
    if "WProd"  in active: df["WProd"]  = np.prod(np.clip(F, eps, None) ** weights, axis=1)
    if "Tcheby" in active: df["Tcheby"] = np.max(weights * np.abs(1.0 - F), axis=1)
    if "ASF"    in active:
        w2 = np.maximum(weights, eps)
        T  = (1.0 - F) / w2
        df["ASF"] = T.max(axis=1) + rho * T.sum(axis=1)

    # Rankings
    rank_cols = []
    for m, asc in [("WSum",False),("WProd",False),("Tcheby",True),("ASF",True)]:
        if m in df.columns:
            df[f"R_{m}"] = df[m].rank(ascending=asc, method="min").astype(int)
            rank_cols.append(f"R_{m}")
    if rank_cols:
        df["R_Consensus"] = df[rank_cols].mean(axis=1)

    # Pareto
    L("🔵 Pareto...")
    df["Pareto"]       = _pareto_mask(F)
    df["Pareto_Level"] = _pareto_levels(F)
    n_p = int(df["Pareto"].sum())
    L(f"   Frente 1: {n_p} Pareto-óptimas ({100*n_p/len(df):.1f}%)")

    # Hipervolumen
    L("📐 Hipervolumen...")
    ref = np.full(F.shape[1], 1.2)
    df["Hypervolume"] = _hv_per_row(F, ref)
    L(f"   HV medio={df['Hypervolume'].mean():.6f}  max={df['Hypervolume'].max():.6f}")

    # Guardar CSV resultado (único archivo de salida)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"multiobjective_report_{ts}")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "multiobjective_report.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    L(f"\n💾 CSV resultado: {out_csv}  ({len(df)} filas · {len(df.columns)} col)")

    # Resumen por combinación
    gcols = [c for c in ["combination_id","combination_name","det_model",
                          "cls_model","vlm_model","llm_model"] if c in df.columns]
    if gcols:
        sc = [c for c in ["WSum","WProd","Tcheby","ASF","Hypervolume","R_Consensus"]
              if c in df.columns]
        rows = []
        for keys, g in df.groupby(gcols):
            row = dict(zip(gcols, keys if isinstance(keys, tuple) else [keys]))
            row["n"] = len(g)
            for c in sc: row[c] = float(g[c].mean())
            row["Pareto_n"] = int(g["Pareto"].sum())
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir/"summary_by_combination.csv",
                                  index=False, float_format="%.6f")
        L(f"   summary_by_combination.csv  ({len(rows)} combinaciones)")

    L(f"\n✅ Análisis completado → {out_dir}/")

    # Dict resumen para la UI
    n = len(df)
    summ = {
        "n": n, "n_pareto": n_p,
        "pct_pareto": 100*n_p/n if n else 0,
        "n_dominated": n - n_p,
        "n_fronts": int(df["Pareto_Level"].max()),
        "hv_mean":  float(df["Hypervolume"].mean()),
        "hv_max":   float(df["Hypervolume"].max()),
        "hv_p_mean":float(df.loc[df["Pareto"],"Hypervolume"].mean()) if n_p else 0,
        "methods": {}, "top5": [], "out_dir": str(out_dir),
    }
    for m, asc, col in [("WSum",False,"WSum"),("WProd",False,"WProd"),
                         ("Tcheby",True,"Tcheby"),("ASF",True,"ASF")]:
        if col not in df.columns: continue
        idx = df[col].idxmin() if asc else df[col].idxmax()
        bv  = float(df[col].min()  if asc else df[col].max())
        bn  = str(df.loc[idx,"combination_name"] if "combination_name" in df.columns else idx)
        summ["methods"][m] = {"best": bv, "best_name": bn}
    if "R_Consensus" in df.columns:
        for _, row in df.nsmallest(5,"R_Consensus").iterrows():
            summ["top5"].append({
                "name": str(row.get("combination_name", row.name)),
                "WSum": float(row.get("WSum",0)), "Tcheby": float(row.get("Tcheby",0)),
                "HV":   float(row.get("Hypervolume",0)), "P": bool(row.get("Pareto",False)),
            })
    return {"df": df, "summary": summ, "out_dir": str(out_dir), "n_rows": n}

# ══════════════════════════════════════════════════════════════════════════════
# § 4  HELPERS ESTADO / CURSES
# ══════════════════════════════════════════════════════════════════════════════
def default_objectives():
    objs = []
    for ph, mets in CATALOG.items():
        for m in mets:
            if m["rec"]:
                objs.append({"phase":ph,"col":m["col"],"lbl":m["lbl"],
                             "unit":m["unit"],"dir":m["dir"],"weight":0.0,
                             "enabled":True,"rec":True})
    _redist(objs); return objs

def _redist(objs):
    en = [o for o in objs if o["enabled"]]
    if not en: return
    w = round(1.0/len(en), 4)
    for i,o in enumerate(en):
        o["weight"] = round(1.0 - w*(len(en)-1), 4) if i==len(en)-1 else w

def wsum(objs): return round(sum(o["weight"] for o in objs if o["enabled"]),4)

def tr(s, n):
    s = str(s)
    return s if len(s)<=n else s[:n-1]+"…"

def wbar(w, W=10):
    f = max(0,min(int(round(w*W)),W))
    return "█"*f+"░"*(W-f)

# Colores
cN=1;cH=2;cS=3;cP=4;cMX=5;cMN=6;cW=7;cOK=8;cER=9;cDM=10;cMT=11;cBR=12;cTI=13;cRN=14;cGR=15
cHK=16   # hint key: tecla resaltada en barras de ayuda
_DIM=False

def dim():
    return curses.color_pair(cDM)|(curses.A_DIM if _DIM else 0)

def init_colors():
    global _DIM
    curses.start_color(); curses.use_default_colors()
    dc = 8 if curses.COLORS>=16 else curses.COLOR_WHITE
    _DIM = curses.COLORS < 16
    for pid,fg,bg in [
        (cN, curses.COLOR_WHITE,   -1),
        (cH, curses.COLOR_CYAN,    -1),
        (cS, curses.COLOR_BLACK,   curses.COLOR_CYAN),
        (cP, curses.COLOR_YELLOW,  -1),
        (cMX,curses.COLOR_GREEN,   -1),
        (cMN,curses.COLOR_RED,     -1),
        (cW, curses.COLOR_MAGENTA, -1),
        (cOK,curses.COLOR_GREEN,   -1),
        (cER,curses.COLOR_RED,     -1),
        (cDM,dc,                   -1),
        (cMT,curses.COLOR_YELLOW,  -1),
        (cBR,curses.COLOR_BLUE,    -1),
        (cTI,curses.COLOR_CYAN,    -1),
        (cRN,curses.COLOR_GREEN,   -1),
        (cGR,curses.COLOR_MAGENTA, -1),
        (cHK,curses.COLOR_WHITE,   -1),   # teclas en barras de ayuda
    ]:
        try: curses.init_pair(pid, fg, bg)
        except Exception: pass

def sa(win, y, x, txt, a=0):
    h,w = win.getmaxyx()
    if y<0 or y>=h or x<0 or x>=w: return
    ml = w-x-1
    if ml<=0: return
    try: win.addstr(y, x, str(txt)[:ml], a)
    except curses.error: pass

def hl(win, y, x, w, c=cBR):
    sa(win, y, x, "─"*w, curses.color_pair(c))

def hint_bar(win, y, text, max_w=None):
    """
    Dibuja una barra de ayuda en la fila y con colores diferenciados:
      - texto entre [ ] → blanco brillante (cHK + A_BOLD)
      - texto fuera de [ ] → gris tenue (dim)
    """
    h, w = win.getmaxyx()
    if y < 0 or y >= h: return
    if max_w is None: max_w = w-1
    x = 0
    i = 0
    while i < len(text) and x < max_w:
        ch = text[i]
        if ch == '[':
            # buscar el cierre ]
            j = text.find(']', i+1)
            if j == -1: j = len(text)-1
            segment = text[i:j+1]
            at = curses.color_pair(cHK)|curses.A_BOLD
            chunk = segment[:max_w-x]
            try: win.addstr(y, x, chunk, at)
            except curses.error: pass
            x += len(chunk); i = j+1
        else:
            # acumular texto hasta el próximo [ o fin
            j = text.find('[', i+1)
            if j == -1: j = len(text)
            segment = text[i:j]
            chunk = segment[:max_w-x]
            try: win.addstr(y, x, chunk, dim())
            except curses.error: pass
            x += len(chunk); i = j

TABS = ["[1] Config","[2] BD","[3] Listas","[4] Ejecutar","[5] Analisis","[6] Modelos"]

# ══════════════════════════════════════════════════════════════════════════════
# § 5  CLASE BASE
# ══════════════════════════════════════════════════════════════════════════════
class Screen:
    def __init__(self, scr, st):
        self.scr = scr; self.st = st
        self.h, self.w = scr.getmaxyx()

    def _sz(self): self.h, self.w = self.scr.getmaxyx()

    def title(self, idx, sub=""):
        self._sz()
        banner = " Suite Multiobjetivo · Framework Multimodal de Tráfico "
        sa(self.scr,0,0,"═"*(self.w-1),curses.color_pair(cBR))
        sa(self.scr,0,max(0,(self.w-len(banner))//2),banner,curses.color_pair(cTI)|curses.A_BOLD)
        x=2
        for i,t in enumerate(TABS):
            sa(self.scr,1,x,f" {t} ",curses.color_pair(cS)|curses.A_BOLD if i==idx else dim())
            x+=len(t)+3
        if sub: sa(self.scr,1,self.w-len(sub)-2,sub,curses.color_pair(cP))
        hl(self.scr,2,0,self.w-1)

    def statusbar(self):
        en  = [o for o in self.st["objectives"] if o["enabled"]]
        ws  = wsum(self.st["objectives"])
        ok  = abs(ws-1.0)<0.005
        csv = Path(self.st.get("input_csv","")).name if self.st.get("input_csv") else "—"
        col = curses.color_pair(cOK) if ok else curses.color_pair(cER)
        st  = (f" Obj:{len(en)}  Met:{sum(self.st['methods'].values())}  "
               f"Sw={ws:.4f}  ρ={self.st['rho']}  CSV:{tr(csv,30)}  "
               f"{'✓ Pesos OK' if ok else '✗ Revisar pesos'} ")
        sa(self.scr,self.h-2,0," "*(self.w-1),col)
        sa(self.scr,self.h-2,0,st[:self.w-1],col)

    def navbar(self, hint=""):
        hint_bar(self.scr, self.h-1,
                 f" [Tab]Siguiente  [r]Redistribuir pesos  [q]Salir  {hint}")

    def key(self, k): return None

# ══════════════════════════════════════════════════════════════════════════════
# § 6  PANTALLA 1 — CONFIGURADOR
# ══════════════════════════════════════════════════════════════════════════════
class ConfigScreen(Screen):
    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub=0; self.cur=0; self.scroll=0
        self.phc=0; self.mtc=0; self.panel="phases"
        self.phases=list(CATALOG.keys()); self.mc=0
        # ── modo edición explícito ──────────────────────────────────────────
        # None  → solo navegación (↑↓)
        # "dir" → [d] presionado, ←→ cambia max/min
        # "w"   → [w] presionado, ←→ cambia peso ±0.01
        self.mode = None
        self.notif = ""; self.ntimer = 0   # notificaciones inline

    def _en(self): return [o for o in self.st["objectives"] if o["enabled"]]

    def draw(self):
        self.scr.erase(); self._sz()
        self.title(0,["[A]Objetivos","[B]Métricas","[C]Métodos"][self.sub])
        self.statusbar()
        [self._dobj,self._dmet,self._dmth][self.sub]()
        self.navbar(); self.scr.refresh()

    # ── sub A: Objetivos ──────────────────────────────────────────────────────
    def _dobj(self):
        objs=self._en(); rs=3; vis=self.h-7   # -7 para dejar 2 líneas de hints
        # ── cabecera de tabla ──────────────────────────────────────────────
        sa(self.scr,rs,0,
           f"  {'#':>2}  {'Fase':<14} {'Métrica':<16} {'CSV col':<20}  {'Dir':^6}  {'Peso':^8}  {'Barra':<10}  {'U':<5}  Rec"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr,rs+1,0,self.w-1)
        # ── scroll ────────────────────────────────────────────────────────
        if self.cur>=self.scroll+vis-1: self.scroll=self.cur-vis+2
        if self.cur<self.scroll:        self.scroll=self.cur
        self.scroll=max(0,self.scroll)
        # ── filas ─────────────────────────────────────────────────────────
        for idx,obj in enumerate(objs):
            sr=rs+2+(idx-self.scroll)
            if not (rs+2<=sr<=self.h-6): continue
            sel=idx==self.cur
            ba=curses.color_pair(cS) if sel else curses.color_pair(cN)
            ph=obj["phase"].split("·")[1].strip() if "·" in obj["phase"] else obj["phase"]
            ds="↑ max" if obj["dir"]=="max" else "↓ min"
            rec=obj.get("rec",False)
            ln=f"  {idx+1:>2}  {tr(ph,14):<14} {tr(obj['lbl'],16):<16} {tr(obj['col'],20):<20}  "
            sa(self.scr,sr,0," "*(self.w-1),ba)
            sa(self.scr,sr,0,ln[:self.w-1],ba)
            cx=len(ln)
            # columna Dir — resaltada si modo=="dir" y es la fila activa
            if sel and self.mode=="dir":
                da=curses.color_pair(cP)|curses.A_BOLD|curses.A_REVERSE
            elif not sel:
                da=curses.color_pair(cMX) if obj["dir"]=="max" else curses.color_pair(cMN)
            else:
                da=curses.color_pair(cMX)|curses.A_BOLD if obj["dir"]=="max" else curses.color_pair(cMN)|curses.A_BOLD
            sa(self.scr,sr,cx,f"{ds:^6}",da); cx+=6
            # columna Peso — resaltada si modo=="w" y es la fila activa
            if sel and self.mode=="w":
                wa=curses.color_pair(cMT)|curses.A_BOLD|curses.A_REVERSE
            else:
                wa=ba
            sa(self.scr,sr,cx+2,f"{obj['weight']:.4f}",wa); cx+=10
            sa(self.scr,sr,cx,f"  {wbar(obj['weight'])}",ba); cx+=12
            sa(self.scr,sr,cx,f"  {obj['unit']:<5}",dim() if not sel else ba); cx+=7
            # columna Rec — ★ si recomendada, marcada persistente
            if rec:
                sa(self.scr,sr,cx,"  ★",curses.color_pair(cMX)|curses.A_BOLD)
            else:
                sa(self.scr,sr,cx,"  ·",dim())
        # ── panel de instrucciones contextual (2 líneas) ──────────────────
        if self.mode is None:
            hint_bar(self.scr, self.h-4,
                     " [↑↓]Navegar  [d]Editar dirección  [w]Editar peso  [s]Toggle★Rec  [x]Quitar  [r]Redistrib")
        elif self.mode=="dir":
            sa(self.scr,self.h-4,0,
               " MODO DIRECCIÓN ACTIVO → [←]min [→]max  [d]toggle directo  [Esc]Cancelar"[:self.w-1],
               curses.color_pair(cP)|curses.A_BOLD)
        elif self.mode=="w":
            sa(self.scr,self.h-4,0,
               " MODO PESO ACTIVO → [←]−0.01 [→]+0.01  [0]cero  [=]igualar  [Esc]Cancelar"[:self.w-1],
               curses.color_pair(cMT)|curses.A_BOLD)
        # línea inferior: instrucciones globales + notificación
        if self.notif and self.ntimer>0:
            sa(self.scr,self.h-3,0,f" {self.notif} "[:self.w-1],
               curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer-=1
        else:
            hint_bar(self.scr, self.h-3,
                     " [A/B/C]Sub-tabs  [r]Redistribuir  [s]Toggle★Rec  [Tab]Pantalla  [q]Salir ")

    # ── sub B: Métricas ───────────────────────────────────────────────────────
    def _dmet(self):
        rs=3; mid=self.w//3
        sa(self.scr,rs,2,"FASES",curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr,rs+1,0,mid-2)
        ac={o["col"] for o in self.st["objectives"] if o["enabled"]}
        for i,ph in enumerate(self.phases):
            row=rs+2+i
            if row>=self.h-3: break
            n_a=sum(1 for o in self.st["objectives"] if o["enabled"] and o["phase"]==ph)
            sel=i==self.phc
            at=(curses.color_pair(cS) if sel and self.panel=="phases" else
                curses.color_pair(cP) if sel else curses.color_pair(cN))
            pn=ph.split("·")[1].strip() if "·" in ph else ph
            sa(self.scr,row,0," "*(mid-1),at)
            sa(self.scr,row,0,f" {PH_ICON.get(ph,'·')} {tr(pn,mid-8):<16} [{n_a}] ",at)
        for row in range(rs,self.h-3): sa(self.scr,row,mid,"│",curses.color_pair(cBR))
        curr=self.phases[self.phc]; mets=CATALOG[curr]
        sa(self.scr,rs,mid+2,f"MÉTRICAS — {curr}",curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr,rs+1,mid+1,self.w-mid-2)
        for i,m in enumerate(mets):
            row=rs+2+i
            if row>=self.h-3: break
            act=m["col"] in ac; cm=i==self.mtc and self.panel=="metrics"
            at=(curses.color_pair(cS) if cm else
                curses.color_pair(cOK) if act else curses.color_pair(cN))
            ck="✓" if act else "○"; rc="★" if m["rec"] else " "; ds="↑" if m["dir"]=="max" else "↓"
            sa(self.scr,row,mid+1," "*(self.w-mid-2),at)
            sa(self.scr,row,mid+1,
               f" {ck} {rc} {tr(m['lbl'],22):<22} {ds}  {m['unit']:<6}  {m['col']}"[:self.w-mid-3],at)
        hint_bar(self.scr, self.h-3,
                 " ★=Rec([s]en sub-A) ✓=Activa [←→]Panel [↑↓]Mover [Space/Enter]Toggle [A/B/C] ")

    # ── sub C: Métodos ────────────────────────────────────────────────────────
    def _dmth(self):
        rs=3
        sa(self.scr,rs,2,"Métodos de escalarización",curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr,rs+1,0,self.w-1)
        fml={"WSum":"F = Σ wᵢ·fᵢ_norm","WProd":"F = Π fᵢ_norm^wᵢ",
             "Tcheby":"F = max{ wᵢ·|1−fᵢ_norm| }",
             "ASF":f"F = max{{(1−fᵢ)/wᵢ}}+ρ·Σ{{(1−fᵢ)/wᵢ}}  ρ={self.st['rho']}"}
        for i,m in enumerate(METHODS):
            row=rs+2+i*3; ic=i==self.mc; ia=self.st["methods"].get(m,True)
            at=(curses.color_pair(cS) if ic else curses.color_pair(cOK) if ia else dim())
            sa(self.scr,row,2," "*(self.w-4),at)
            sa(self.scr,row,2,f" {'◉' if ia else '○'}  {m:<10}  {M_DESC[m]}",at)
            sa(self.scr,row+1,6,f"   {fml.get(m,'')}  "[:self.w-8],dim())
        rr=rs+2+len(METHODS)*3+1
        ra=curses.color_pair(cS) if self.mc==len(METHODS) else curses.color_pair(cMT)
        sa(self.scr,rr,2,f" ρ = {self.st['rho']:<8}  [←]−0.001  [→]+0.001  Rango:[0.0001–0.05]",ra)
        hint_bar(self.scr, self.h-3,
                 " [↑↓]Mover [Space/Enter]Toggle [←→]Ajustar ρ [A/B/C] ")

    # ── teclas ────────────────────────────────────────────────────────────────
    def key(self, k):
        if k in (ord('a'),ord('A')): self.sub=0; self.mode=None; return None
        if k in (ord('b'),ord('B')): self.sub=1; self.mode=None; return None
        if k in (ord('c'),ord('C')): self.sub=2; self.mode=None; return None
        return [self._ka,self._kb,self._kc][self.sub](k)

    def _ka(self, k):
        en=self._en(); n=len(en)

        # ── Esc: siempre cancela modo activo ──────────────────────────────
        if k==27:
            if self.mode is not None:
                self.mode=None; return None
            return "quit"

        # ── Navegación vertical — solo cuando no hay modo activo ──────────
        if self.mode is None:
            if   k in (curses.KEY_UP,  ord('k')) and self.cur>0:   self.cur-=1; return None
            elif k in (curses.KEY_DOWN,ord('j')) and self.cur<n-1: self.cur+=1; return None

        # ── Activar / desactivar modo dirección con [d] ───────────────────
        if k==ord('d'):
            if self.mode=="dir":
                # segundo [d] = toggle directo sin flechas
                if n>0:
                    o=en[self.cur]; o["dir"]="min" if o["dir"]=="max" else "max"
                    self.notif=f"  Dir → {'↑ max' if o['dir']=='max' else '↓ min'}  ({o['col']})"; self.ntimer=55
                self.mode=None
            else:
                self.mode="dir"
            return None

        # ── Activar / desactivar modo peso con [w] ────────────────────────
        if k==ord('w'):
            self.mode=None if self.mode=="w" else "w"
            return None

        # ── Acciones dentro de modo dirección ────────────────────────────
        if self.mode=="dir" and n>0:
            o=en[self.cur]
            if k==curses.KEY_LEFT:
                o["dir"]="min"
                self.notif=f"  Dir → ↓ min  ({o['col']})"; self.ntimer=55
            elif k==curses.KEY_RIGHT:
                o["dir"]="max"
                self.notif=f"  Dir → ↑ max  ({o['col']})"; self.ntimer=55
            elif k in (curses.KEY_UP,ord('k')) and self.cur>0:
                self.cur-=1
            elif k in (curses.KEY_DOWN,ord('j')) and self.cur<n-1:
                self.cur+=1
            return None

        # ── Acciones dentro de modo peso ─────────────────────────────────
        if self.mode=="w" and n>0:
            o=en[self.cur]
            if k==curses.KEY_LEFT  or k==ord('-'):
                o["weight"]=round(max(0.0,o["weight"]-0.01),4)
            elif k==curses.KEY_RIGHT or k==ord('+'):
                o["weight"]=round(min(1.0,o["weight"]+0.01),4)
            elif k==ord('0'):
                o["weight"]=0.0
            elif k==ord('='):             # igualar pesos de todos
                _redist(self.st["objectives"])
                self.notif="  Pesos redistribuidos"; self.ntimer=55
            elif k in (curses.KEY_UP,ord('k')) and self.cur>0:
                self.cur-=1
            elif k in (curses.KEY_DOWN,ord('j')) and self.cur<n-1:
                self.cur+=1
            return None

        # ── Acciones globales (sin modo activo) ───────────────────────────
        if k==ord('s') and n>0:
            # Toggle recomendación ★ — persiste en el objetivo y en la BD
            o=en[self.cur]
            o["rec"]=not o.get("rec",False)
            # sincronizar también en CATALOG para que sub-B lo refleje
            for ph,mets in CATALOG.items():
                for m in mets:
                    if m["col"]==o["col"]:
                        m["rec"]=o["rec"]
            estado="★ Marcada como recomendada" if o["rec"] else "· Recomendación quitada"
            self.notif=f"  {estado}: {o['col']}  (se persiste al guardar en BD)"; self.ntimer=80
        elif k in (curses.KEY_DC,ord('x')) and n>0:
            col=en[self.cur]["col"]
            for o in self.st["objectives"]:
                if o["col"]==col: o["enabled"]=False
            _redist(self.st["objectives"])
            self.cur=max(0,min(self.cur,len(self._en())-1))
        elif k==ord('r'):
            _redist(self.st["objectives"])
            self.notif="  Pesos redistribuidos uniformemente"; self.ntimer=55
        elif k==ord('\t'): return "next"
        return None

    def _kb(self, k):
        phs=self.phases; mets=CATALOG[phs[self.phc]]
        ac={o["col"] for o in self.st["objectives"] if o["enabled"]}
        if k in (curses.KEY_LEFT,curses.KEY_RIGHT):
            self.panel="phases" if self.panel=="metrics" else "metrics"
        elif self.panel=="phases":
            if   k in (curses.KEY_UP,ord('k'))   and self.phc>0:           self.phc-=1; self.mtc=0
            elif k in (curses.KEY_DOWN,ord('j')) and self.phc<len(phs)-1:  self.phc+=1; self.mtc=0
            elif k in (ord('\n'),ord('\r'),10,13,ord(' ')): self.panel="metrics"
        else:
            if   k in (curses.KEY_UP,ord('k'))   and self.mtc>0:            self.mtc-=1
            elif k in (curses.KEY_DOWN,ord('j')) and self.mtc<len(mets)-1:  self.mtc+=1
            elif k in (ord('\n'),ord('\r'),10,13,ord(' ')):
                m=mets[self.mtc]; ph=phs[self.phc]
                if m["col"] in ac:
                    for o in self.st["objectives"]:
                        if o["col"]==m["col"]: o["enabled"]=False
                else:
                    ex=next((o for o in self.st["objectives"] if o["col"]==m["col"]),None)
                    if ex: ex["enabled"]=True
                    else:
                        self.st["objectives"].append(
                            {"phase":ph,"col":m["col"],"lbl":m["lbl"],
                             "unit":m["unit"],"dir":m["dir"],"weight":0.0,"enabled":True})
                _redist(self.st["objectives"])
        if k==ord('r'): _redist(self.st["objectives"])
        elif k==ord('\t'): return "next"
        elif k in (ord('q'),27): return "prev"
        return None

    def _kc(self, k):
        n=len(METHODS)
        if   k in (curses.KEY_UP,ord('k'))   and self.mc>0: self.mc-=1
        elif k in (curses.KEY_DOWN,ord('j')) and self.mc<n: self.mc+=1
        elif k in (ord('\n'),ord('\r'),10,13,ord(' ')):
            if self.mc<n:
                m=METHODS[self.mc]; self.st["methods"][m]=not self.st["methods"].get(m,True)
        elif k==curses.KEY_RIGHT and self.mc==n:
            self.st["rho"]=round(min(0.05,self.st["rho"]+0.001),4)
        elif k==curses.KEY_LEFT and self.mc==n:
            self.st["rho"]=round(max(0.0001,self.st["rho"]-0.001),4)
        elif k==ord('r'): _redist(self.st["objectives"])
        elif k==ord('\t'): return "next"
        elif k in (ord('q'),27): return "prev"
        return None

# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# § 7  PANTALLA 2 — BASE DE DATOS  (configs sin CSV + runs registradas)
# ══════════════════════════════════════════════════════════════════════════════
class DBScreen(Screen):
    """
    Sub-tabs:  [C]onfigs  |  [R]uns
    Configs : guardar/cargar/borrar configs (pesos, métodos, objetivos, rho) — SIN csv
    Runs    : lista de ejecuciones registradas (config + csv + out_dir)
    """
    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub     = "configs"   # "configs" | "runs"
        self.cur     = 0
        self.scroll  = 0
        self.cfgs    = []
        self.runs    = []
        self.mode    = "list"      # list|save_name|save_desc|confirm_del|confirm_del_run
        self.ibuf    = ""
        self.sname   = ""
        self.sdesc   = ""
        self.notif   = ""; self.ntimer = 0
        self._reload()

    def _reload(self):
        self.cfgs = db_list()
        self.runs = run_list()
        items = self.cfgs if self.sub == "configs" else self.runs
        self.cur = min(self.cur, max(0, len(items) - 1))

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase(); self._sz()
        sub_hint = "[C]Configs  [R]Runs"
        self.title(1, sub_hint)
        self.statusbar()

        if self.mode == "save_name":   self._dsave(); self.navbar(); self.scr.refresh(); return
        if self.mode == "save_desc":   self._dsave(); self.navbar(); self.scr.refresh(); return
        if self.mode == "confirm_del": self._ddel("config"); self.navbar(); self.scr.refresh(); return
        if self.mode == "confirm_del_run": self._ddel("run"); self.navbar(); self.scr.refresh(); return

        rs = 3; vis = self.h - 7
        # ── sub-tab header ────────────────────────────────────────────────
        cA = curses.color_pair(cS)|curses.A_BOLD if self.sub=="configs" else dim()
        cB = curses.color_pair(cS)|curses.A_BOLD if self.sub=="runs"    else dim()
        sa(self.scr, rs, 2, " [C] CONFIGURACIONES ", cA)
        sa(self.scr, rs, 24," [R] EJECUCIONES (RUNS) ", cB)
        hl(self.scr, rs+1, 0, self.w-1)

        if self.sub == "configs":
            self._dconfigs(rs+2, vis)
            if self.notif and self.ntimer > 0:
                sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1],
                   curses.color_pair(cOK)|curses.A_BOLD)
                self.ntimer -= 1
            else:
                hint_bar(self.scr, self.h-3,
                         " [↑↓]Mover  [s]Guardar cfg  [l]Cargar en P1  [d]Borrar  [r]Recargar  [C]Configs [R]Runs ")
        else:
            self._druns(rs+2, vis)
            if self.notif and self.ntimer > 0:
                sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1],
                   curses.color_pair(cOK)|curses.A_BOLD)
                self.ntimer -= 1
            else:
                hint_bar(self.scr, self.h-3,
                         " [↑↓]Mover  [d]Borrar run  [r]Recargar  [C]Configs [R]Runs ")
        self.navbar(); self.scr.refresh()

    def _dconfigs(self, rs, vis):
        sa(self.scr, rs, 0,
           f"  {'ID':>4}  {'Nombre':<24} {'Fecha':<19} {'Obj':>4}  {'ρ':>6}  {'Métodos':<20}  {'#Runs'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+1, 0, self.w-1)
        if not self.cfgs:
            sa(self.scr, rs+3, 4, "Sin configuraciones. [s] para guardar la configuración actual.", dim())
            return
        if self.cur >= self.scroll+vis-1: self.scroll = self.cur-vis+2
        if self.cur < self.scroll:        self.scroll = self.cur
        self.scroll = max(0, self.scroll)
        # contar runs por config
        runs_by_cfg = {}
        for r in self.runs:
            runs_by_cfg[r["config_id"]] = runs_by_cfg.get(r["config_id"], 0) + 1
        for idx, cfg in enumerate(self.cfgs):
            sr = rs+2+(idx-self.scroll)
            if not (rs+2 <= sr <= self.h-6): continue
            sel = idx == self.cur
            at  = curses.color_pair(cS) if sel else curses.color_pair(cN)
            n_o = len([o for o in cfg.get("objectives",[]) if o.get("enabled")])
            mets= ",".join(k for k,v in cfg.get("methods",{}).items() if v)
            nr  = runs_by_cfg.get(cfg["id"], 0)
            ln  = (f"  {cfg['id']:>4}  {tr(cfg['name'],24):<24} "
                   f"{cfg['created_at'][:19]:<19} {n_o:>4}  ρ={cfg['rho']:.4f}  "
                   f"{tr(mets,20):<20}  {nr}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
        # panel detalle
        cfg = self.cfgs[self.cur]
        dr  = rs+2+min(vis, len(self.cfgs))+1
        if dr < self.h-4:
            hl(self.scr, dr-1, 0, self.w-1)
            sa(self.scr, dr, 2,
               f"Desc: {tr(cfg.get('description','—'),40)}   (sin CSV — CSV se elige en P3)",
               curses.color_pair(cP))

    def _druns(self, rs, vis):
        sa(self.scr, rs, 0,
           f"  {'ID':>4}  {'Config':<20} {'CSV de entrada':<36} {'Fecha':<19}  {'Filas':>6}  {'Out dir'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+1, 0, self.w-1)
        if not self.runs:
            sa(self.scr, rs+3, 4, "Sin ejecuciones. Ejecuta el análisis en P3 → [e].", dim())
            return
        if self.cur >= self.scroll+vis-1: self.scroll = self.cur-vis+2
        if self.cur < self.scroll:        self.scroll = self.cur
        self.scroll = max(0, self.scroll)
        for idx, run in enumerate(self.runs):
            sr = rs+2+(idx-self.scroll)
            if not (rs+2 <= sr <= self.h-6): continue
            sel = idx == self.cur
            at  = curses.color_pair(cS) if sel else curses.color_pair(cN)
            csv_name = Path(run["input_csv"]).name if run["input_csv"] else "—"
            out_name = Path(run["out_dir"]).name   if run["out_dir"]   else "—"
            ok_mark  = "✓" if run["out_dir"] and Path(run["out_dir"]).exists() else "✗"
            ln = (f"  {run['id']:>4}  {tr(run['config_name'],20):<20} "
                  f"{tr(csv_name,36):<36} {run['ran_at'][:19]:<19}  "
                  f"{run['n_rows']:>6}  {ok_mark} {tr(out_name,self.w-110)}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
            if sel:
                # colorear el ✓/✗
                cx = 4+2+20+2+36+2+19+2+7
                col = curses.color_pair(cOK) if ok_mark=="✓" else curses.color_pair(cER)
                sa(self.scr, sr, cx, ok_mark, col)
        # panel detalle
        run = self.runs[self.cur]
        dr  = rs+2+min(vis, len(self.runs))+1
        if dr < self.h-4:
            hl(self.scr, dr-1, 0, self.w-1)
            sa(self.scr, dr, 2,
               f"CSV: {tr(run['input_csv'] or '—', 50)}   Out: {tr(run['out_dir'] or '—', max(10,self.w-70))}",
               curses.color_pair(cP))

    def _dsave(self):
        rs = 5; b = curses.color_pair(cBR)
        sa(self.scr, rs,   4, "╔"+"═"*58+"╗", b)
        sa(self.scr, rs+1, 4, "║  GUARDAR CONFIGURACIÓN (sin CSV)"+" "*25+"║", b)
        sa(self.scr, rs+2, 4, "╠"+"═"*58+"╣", b)
        if self.mode == "save_name":
            sa(self.scr, rs+3, 6, f"Nombre : {self.ibuf}█",  curses.color_pair(cP)|curses.A_BOLD)
            sa(self.scr, rs+4, 6, f"Desc   : {self.sdesc}",  dim())
        else:
            sa(self.scr, rs+3, 6, f"Nombre : {self.sname}",  curses.color_pair(cOK))
            sa(self.scr, rs+4, 6, f"Desc   : {self.ibuf}█",  curses.color_pair(cP)|curses.A_BOLD)
        n_o = len([o for o in self.st["objectives"] if o["enabled"]])
        mets= ",".join(k for k,v in self.st["methods"].items() if v)
        sa(self.scr, rs+5, 6,
           f"Obj:{n_o}  ρ={self.st['rho']}  Sw={wsum(self.st['objectives']):.4f}  Mét:{mets}",dim())
        sa(self.scr, rs+6, 6, "Nota: el CSV se asocia al ejecutar en P3, no aquí.", curses.color_pair(cMT))
        sa(self.scr, rs+7, 4, "║  [Enter] Confirmar   [Esc] Cancelar"+" "*22+"║", dim())
        sa(self.scr, rs+8, 4, "╚"+"═"*58+"╝", b)

    def _ddel(self, kind):
        items = self.cfgs if kind == "config" else self.runs
        if not items: return
        item = items[self.cur]; rs = 6
        label = item.get("name", item.get("config_name","?"))
        sa(self.scr, rs,   4, "╔"+"═"*52+"╗", curses.color_pair(cER))
        sa(self.scr, rs+1, 4, f"║  ¿BORRAR {'Config' if kind=='config' else 'Run'} "
                               f"ID={item['id']}  {tr(label,28)}?  ║",
           curses.color_pair(cER)|curses.A_BOLD)
        sa(self.scr, rs+2, 6, "  [y] Confirmar borrado    [n / Esc] Cancelar",
           curses.color_pair(cP))
        sa(self.scr, rs+3, 4, "╚"+"═"*52+"╝", curses.color_pair(cER))

    # ── teclas ────────────────────────────────────────────────────────────────
    def key(self, k):
        # sub-tab switch
        if k in (ord('c'), ord('C')) and self.mode == "list":
            self.sub = "configs"; self.cur = 0; self.scroll = 0; self._reload(); return None
        if k in (ord('r'), ord('R')) and self.mode == "list":
            self.sub = "runs";    self.cur = 0; self.scroll = 0; self._reload(); return None

        items = self.cfgs if self.sub == "configs" else self.runs

        # ── modo entrada texto (guardar config) ───────────────────────────
        if self.mode in ("save_name", "save_desc"):
            if k in (ord('\n'), ord('\r'), 10, 13):
                if self.mode == "save_name":
                    if self.ibuf.strip():
                        self.sname = self.ibuf.strip(); self.ibuf = ""; self.mode = "save_desc"
                elif self.mode == "save_desc":
                    self.sdesc = self.ibuf.strip()
                    rid = db_save(self.sname, self.sdesc, self.st)
                    self.st["_db_id"] = rid
                    self.notif = f"✓ Config guardada ID={rid}: {self.sname}"; self.ntimer = 80
                    self.ibuf = ""; self.mode = "list"; self._reload()
            elif k == 27: self.mode = "list"; self.ibuf = ""
            elif k in (curses.KEY_BACKSPACE, 127, 8): self.ibuf = self.ibuf[:-1]
            elif 32 <= k <= 126: self.ibuf += chr(k)
            return None

        # ── confirmar borrado config ──────────────────────────────────────
        if self.mode == "confirm_del":
            if k == ord('y') and self.cfgs:
                db_delete(self.cfgs[self.cur]["id"])
                self.notif = "✓ Config borrada"; self.ntimer = 50; self._reload()
            self.mode = "list"; return None

        # ── confirmar borrado run ─────────────────────────────────────────
        if self.mode == "confirm_del_run":
            if k == ord('y') and self.runs:
                run_delete(self.runs[self.cur]["id"])
                self.notif = "✓ Run borrada"; self.ntimer = 50; self._reload()
            self.mode = "list"; return None

        # ── navegación normal ─────────────────────────────────────────────
        n = len(items)
        if   k in (curses.KEY_UP,   ord('k')) and self.cur > 0:   self.cur -= 1
        elif k in (curses.KEY_DOWN, ord('j')) and self.cur < n-1: self.cur += 1
        elif k == ord('s') and self.sub == "configs":
            self.ibuf = ""; self.sname = ""; self.sdesc = ""
            self.mode = "save_name"
        elif k == ord('l') and self.sub == "configs" and n > 0:
            ld = db_load(self.cfgs[self.cur]["id"])
            if ld:
                self.st.update(ld)
                self.notif = f"✓ Config cargada en P1: {ld.get('_db_name','?')}"; self.ntimer = 80
        elif k == ord('d') and n > 0:
            self.mode = "confirm_del" if self.sub == "configs" else "confirm_del_run"
        elif k in (ord('R'), ord('r')) and self.mode != "list":
            pass   # ya manejado arriba
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        return None


def _find_csvs(extra_hint: str = "") -> list:
    """Busca CSVs de diagnóstico en el directorio actual, el del script,
    /workspace y subdirectorios comunes. Soporta los patrones:
      results_YYYYMMDD_HHMMSS.csv     ← diagnostic_validation_full_trace.py
      res_F*_YYYYMMDD_HHMMSS.csv      ← fases_diagnostic_validation_full_trace.py
      fases_results_YYYYMMDD_HHMMSS.csv ← versiones anteriores de fases
    Ordena por fecha de modificación descendente (más reciente primero).
    """
    seen = set(); candidates = []
    search_dirs = [
        Path("."),
        Path(__file__).parent,
        Path("/workspace"),
        Path("/workspace/scripts"),
        Path("/workspace/diagnostic_results"),
        Path("/workspace/outputs"),
    ]
    if extra_hint:
        p = Path(extra_hint)
        search_dirs.insert(0, p.parent if p.is_file() else p)

    _PATTERNS = ("results_*.csv", "res_*.csv", "fases_results_*.csv")

    for d in search_dirs:
        try:
            for pat in _PATTERNS:
                for p in d.glob(pat):
                    rp = str(p.resolve())
                    if rp not in seen:
                        seen.add(rp)
                        try:
                            mtime = p.stat().st_mtime
                        except Exception:
                            mtime = 0.0
                        candidates.append((mtime, p))
        except Exception:
            pass

    # Ordenar por mtime descendente → archivos más recientes primero
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in candidates[:12]]   # máximo 12 sugerencias


# ══════════════════════════════════════════════════════════════════════════════
# § 8  PANTALLA 3 — EJECUCIÓN
#       Selector de config BD + selector de CSV → motor interno → registra run
# ══════════════════════════════════════════════════════════════════════════════
class RunScreen(Screen):
    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.log     = []; self.running = False; self.done = False; self.ecode = None
        self.scroll  = 0;  self.sub    = "log";  self.res  = {};    self._thr  = None
        self.mode    = "normal"   # normal | pick_cfg | pick_csv
        self.cfgs    = []
        self.cfg_cur = 0; self.cfg_scroll = 0
        self.ibuf    = ""         # buffer para editar CSV a mano
        self.notif   = ""; self.ntimer = 0
        self.list_scroll = 0      # scroll para sub="list"
        self._reload_cfgs()

    def _reload_cfgs(self):
        self.cfgs = db_list()
        for i, c in enumerate(self.cfgs):
            if c["id"] == self.st.get("_db_id"):
                self.cfg_cur = i; break

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase(); self._sz()
        self.title(3, "[L]Log  [S]Resumen  [l]Lista  [e]Ejecutar  [c]Config  [f]CSV")
        self.statusbar()

        if self.mode == "pick_cfg":
            self._dpick_cfg(); self.navbar(); self.scr.refresh(); return
        if self.mode == "pick_csv":
            self._dpick_csv(); self.navbar(); self.scr.refresh(); return

        if self.sub == "log":     self._dlog()
        elif self.sub == "summary": self._dsum()
        elif self.sub == "list":  self._dlist_runs()

        # barra estado ejecución
        cfg_name = tr(self.st.get("_db_name","— sin config —"), 28)
        csv_name = tr(Path(self.st.get("input_csv","")).name if self.st.get("input_csv") else "— sin CSV —", 38)
        if self.running:
            sa(self.scr, self.h-4, 0,
               f" ⏳ EJECUTANDO...  Config: {cfg_name}  CSV: {csv_name} "[:self.w-1],
               curses.color_pair(cRN)|curses.A_BOLD)
        elif self.done and self.ecode == 0:
            sa(self.scr, self.h-4, 0,
               f" ✅ OK → {tr(self.res.get('out_dir',''),45)}  [e]Re-ejec  [L/S] "[:self.w-1],
               curses.color_pair(cOK)|curses.A_BOLD)
        elif self.done:
            sa(self.scr, self.h-4, 0, " ✗ ERROR — revisa el log"[:self.w-1],
               curses.color_pair(cER)|curses.A_BOLD)
        else:
            sa(self.scr, self.h-4, 0,
               f" Config: {cfg_name}   CSV: {csv_name} "[:self.w-1],
               curses.color_pair(cP))

        # notificaciones
        if self.notif and self.ntimer > 0:
            sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1],
               curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer -= 1
        else:
            hint_bar(self.scr, self.h-3,
                     " [e]Ejecutar  [c]Elegir Config BD  [f]Elegir CSV  [l]Lista runs  [L]Log  [S]Resumen ")
        self.navbar("[e]Ejecutar"); self.scr.refresh()

    def _dpick_cfg(self):
        rs = 3
        sa(self.scr, rs, 2, "SELECCIONAR CONFIGURACIÓN DE BD", curses.color_pair(cH)|curses.A_BOLD)
        sa(self.scr, rs, self.w-38, "(Enter=cargar  Esc=cancelar)", dim())
        hl(self.scr, rs+1, 0, self.w-1)
        sa(self.scr, rs+2, 0,
           f"  {'ID':>4}  {'Nombre':<24} {'Fecha':<19} {'Obj':>4}  {'ρ':>6}  {'Métodos'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+3, 0, self.w-1)
        vis = self.h - rs - 6
        if not self.cfgs:
            sa(self.scr, rs+5, 4, "Sin configs en BD. Guarda primero en P2 → [s].", dim()); return
        if self.cfg_cur >= self.cfg_scroll+vis-1: self.cfg_scroll = self.cfg_cur-vis+2
        if self.cfg_cur < self.cfg_scroll:        self.cfg_scroll = self.cfg_cur
        self.cfg_scroll = max(0, self.cfg_scroll)
        for idx, cfg in enumerate(self.cfgs):
            sr = rs+4+(idx-self.cfg_scroll)
            if not (rs+4 <= sr <= self.h-5): continue
            sel = idx == self.cfg_cur
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            n_o = len([o for o in cfg.get("objectives",[]) if o.get("enabled")])
            mets= ",".join(k for k,v in cfg.get("methods",{}).items() if v)
            ln  = (f"  {cfg['id']:>4}  {tr(cfg['name'],24):<24} "
                   f"{cfg['created_at'][:19]:<19} {n_o:>4}  ρ={cfg['rho']:.4f}  {tr(mets,28)}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
        hint_bar(self.scr, self.h-3,
                 " [↑↓]Mover  [Enter]Cargar config  [Esc]Cancelar ")

    def _dpick_csv(self):
        rs = 5; b = curses.color_pair(cBR)
        found = _find_csvs(self.ibuf)
        box_h = 6 + min(len(found), 8) + 2
        sa(self.scr, rs,   4, "╔"+"═"*68+"╗", b)
        sa(self.scr, rs+1, 4, "║  CSV DE ENTRADA  (results_YYYYMMDD_HHMMSS.csv | res_F*_YYYYMMDD.csv)"+" "*0+"║", b)
        sa(self.scr, rs+2, 4, "╠"+"═"*68+"╣", b)
        sa(self.scr, rs+3, 6, f"Ruta: {self.ibuf}█", curses.color_pair(cP)|curses.A_BOLD)
        ok = Path(self.ibuf).exists() if self.ibuf else False
        sa(self.scr, rs+4, 6,
           ("✓ Archivo encontrado" if ok else "✗ No encontrado") if self.ibuf
           else "  Escribe la ruta o usa [1-8] para elegir de la lista",
           curses.color_pair(cOK) if ok else curses.color_pair(cER if self.ibuf else cDM))
        if found:
            sa(self.scr, rs+5, 6, f"CSVs encontrados ({len(found)}):", dim())
            for i, p in enumerate(found):
                label = f"{i+1}. {p}"
                at = curses.color_pair(cMX)|curses.A_BOLD if str(p) == self.ibuf \
                     else curses.color_pair(cMT)
                sa(self.scr, rs+6+i, 8, tr(label, self.w-12), at)
        else:
            sa(self.scr, rs+5, 6, "No se encontraron CSVs de diagnóstico en rutas conocidas.", curses.color_pair(cER))
        bot = rs + box_h
        sa(self.scr, bot,   4, "║  [Enter]Aplicar  [Esc]Cancelar  [1-9]Elegir lista" +" "*18+"║", dim())
        sa(self.scr, bot+1, 4, "╚"+"═"*68+"╝", b)
        hint_bar(self.scr, self.h-3,
                 " [1-9]Elegir CSV de lista  [Enter]Aplicar  [Esc]Cancelar  [BackSp]Borrar ")

    def _dlist_runs(self):
        """Lista todas las configs con sus runs ejecutadas agrupadas."""
        rs = 3
        hl(self.scr, rs, 0, self.w-1)
        sa(self.scr, rs, 2, " HISTORIAL: CONFIGS + EJECUCIONES ASOCIADAS ",
           curses.color_pair(cH)|curses.A_BOLD)
        vis = self.h - 7

        cfgs  = db_list()
        runs  = run_list()
        # Agrupar runs por config_id
        runs_by = {}
        for r in runs:
            runs_by.setdefault(r["config_id"], []).append(r)
        # Runs sin config conocida
        orphan = [r for r in runs if r["config_id"] not in {c["id"] for c in cfgs}]

        # Construir filas de visualización plana
        lines = []  # cada entrada: (tipo, data)  tipo= "cfg"|"run"|"orphan_hdr"|"sep"
        for cfg in cfgs:
            n_o   = len([o for o in cfg.get("objectives",[]) if o.get("enabled")])
            mets  = ",".join(k for k,v in cfg.get("methods",{}).items() if v)
            cfg_runs = runs_by.get(cfg["id"], [])
            lines.append(("cfg", cfg, n_o, mets, len(cfg_runs)))
            for r in sorted(cfg_runs, key=lambda x: x["id"], reverse=True):
                lines.append(("run", r))
        if orphan:
            lines.append(("orphan_hdr", None))
            for r in orphan:
                lines.append(("run", r))

        total = len(lines)
        vis_real = vis - 1
        self.list_scroll = max(0, min(self.list_scroll, max(0, total - vis_real)))

        row = rs + 1
        for i in range(vis_real):
            idx = self.list_scroll + i
            if idx >= total: break
            entry = lines[idx]
            if entry[0] == "cfg":
                _, cfg, n_o, mets, nr = entry
                active_mark = "►" if cfg["id"] == self.st.get("_db_id") else " "
                txt = (f" {active_mark} CFG #{cfg['id']:>3}  {tr(cfg['name'],22):<22} "
                       f"{cfg['created_at'][:16]}  Obj:{n_o}  [{tr(mets,18)}]  "
                       f"{nr} run{'s' if nr!=1 else ''}")
                at = curses.color_pair(cP)|curses.A_BOLD if cfg["id"] == self.st.get("_db_id") \
                     else curses.color_pair(cH)|curses.A_BOLD
                sa(self.scr, row, 0, " "*(self.w-1), at)
                sa(self.scr, row, 0, txt[:self.w-1], at)
            elif entry[0] == "run":
                _, r = entry
                csv_n = Path(r["input_csv"]).name if r["input_csv"] else "—"
                ok    = "✓" if r["out_dir"] and Path(r["out_dir"]).exists() else "✗"
                ok_col= curses.color_pair(cOK) if ok=="✓" else curses.color_pair(cER)
                txt   = (f"      run #{r['id']:>3}  {r['ran_at'][:16]}  "
                         f"{tr(csv_n,36):<36}  {r['n_rows']:>5} filas")
                sa(self.scr, row, 0, txt[:self.w-2], curses.color_pair(cN))
                sa(self.scr, row, min(self.w-4, 2), ok, ok_col)
                # out_dir resumido al final
                out_n = Path(r["out_dir"]).name if r["out_dir"] else ""
                if out_n:
                    sa(self.scr, row, min(self.w-2, len(txt)+2),
                       f" → {tr(out_n, max(10,self.w-len(txt)-6))}",
                       curses.color_pair(cMT))
            elif entry[0] == "orphan_hdr":
                sa(self.scr, row, 0,
                   " ── Runs sin config asociada ─────────────────",
                   curses.color_pair(cMN)|curses.A_BOLD)
            row += 1

        if not lines:
            sa(self.scr, rs+3, 4,
               "Sin configs ni runs. Guarda una config en P2→[s] y ejecuta P3→[e].", dim())

        # scrollbar lateral
        if total > vis_real:
            for i in range(vis_real):
                f = (self.list_scroll + i) / max(1, total)
                sa(self.scr, rs+1+i, self.w-2,
                   "█" if f < (self.list_scroll + vis_real/2)/max(1,total) else "░", dim())

    def _dlog(self):
        hl(self.scr, 3, 0, self.w-1)
        sa(self.scr, 3, 2, " LOG DE EJECUCIÓN ", curses.color_pair(cH)|curses.A_BOLD)
        vis = self.h - 7
        if self.running: self.scroll = max(0, len(self.log)-vis)
        else:            self.scroll = max(0, min(self.scroll, max(0, len(self.log)-vis)))
        for i in range(vis):
            idx = self.scroll+i
            if idx >= len(self.log): break
            ln = self.log[idx]
            if   "✓" in ln or "✅" in ln: at = curses.color_pair(cOK)
            elif "✗" in ln or "ERROR" in ln or "⚠" in ln: at = curses.color_pair(cER)
            elif "═" in ln or "─" in ln: at = curses.color_pair(cBR)
            else: at = curses.color_pair(cN)
            sa(self.scr, 4+i, 1, tr(ln, self.w-3), at)

    def _dsum(self):
        s = self.res.get("summary", {})
        if not s:
            sa(self.scr, 5, 4, "Sin resumen — ejecuta [e].", dim()); return
        row = 3
        sa(self.scr, row, 2, "RESUMEN DE RESULTADOS", curses.color_pair(cH)|curses.A_BOLD); row+=1
        hl(self.scr, row, 0, self.w-1); row+=1
        def L(lbl, val, c=cN):
            nonlocal row
            sa(self.scr, row, 4, f"{lbl:<36}: {val}", curses.color_pair(c)); row+=1
        sa(self.scr, row, 2, "── PARETO ──", curses.color_pair(cP)|curses.A_BOLD); row+=1
        L("Pareto-óptimas",  f"{s.get('n_pareto',0)}  ({s.get('pct_pareto',0):.1f}%)", cOK)
        L("Dominadas",       f"{s.get('n_dominated',0)}", cDM)
        L("Frentes Pareto",  s.get('n_fronts','—')); row+=1
        sa(self.scr, row, 2, "── HIPERVOLUMEN ──", curses.color_pair(cP)|curses.A_BOLD); row+=1
        L("HV medio",        f"{s.get('hv_mean',0):.6f}")
        L("HV medio Pareto", f"{s.get('hv_p_mean',0):.6f}", cOK)
        L("HV máximo",       f"{s.get('hv_max',0):.6f}",  cMX); row+=1
        sa(self.scr, row, 2, "── ESCALARIZACIÓN ──", curses.color_pair(cP)|curses.A_BOLD); row+=1
        for m, info in s.get("methods",{}).items():
            L(f"{m} — mejor",  f"{info.get('best',0):.5f}")
            L(f"  └ config",   tr(str(info.get('best_name','—')), self.w-46), cP)
        row+=1
        sa(self.scr, row, 2, "── TOP-5 CONSENSO ──", curses.color_pair(cP)|curses.A_BOLD); row+=1
        for i, t in enumerate(s.get("top5",[]), 1):
            if row >= self.h-5: break
            sa(self.scr, row, 4,
               f"{i}. {tr(t.get('name','?'), self.w-20)}", curses.color_pair(cMX)|curses.A_BOLD); row+=1
            sa(self.scr, row, 6,
               f"WSum={t.get('WSum',0):.4f}  TCH={t.get('Tcheby',0):.4f}  "
               f"HV={t.get('HV',0):.4f}  Pareto={t.get('P',False)}", dim()); row+=1

    def _worker(self):
        self.running = True; self.done = False; self.ecode = None
        self.log = ["── Iniciando análisis multiobjetivo (motor interno) ──", ""]
        csv = self.st.get("input_csv","")
        if not csv or not Path(csv).exists():
            self.log.append(f"✗ CSV no encontrado: '{csv}'")
            self.log.append("  → Pulsa [f] para elegir el CSV de entrada.")
            self.running = False; self.done = True; self.ecode = 1; return
        try:
            self.res = run_analysis(csv, self.st, log_cb=self.log.append)
            self.ecode = 0 if self.res else 1
        except Exception as e:
            self.log.append(f"✗ Excepción: {e}")
            for ln in traceback.format_exc().splitlines(): self.log.append(f"  {ln}")
            self.ecode = 1
        # registrar run en BD
        if self.ecode == 0:
            try:
                n_rows = self.res.get("n_rows", 0)
                rid = run_save(
                    config_id   = self.st.get("_db_id") or 0,
                    config_name = self.st.get("_db_name","(sin config)"),
                    input_csv   = csv,
                    out_dir     = self.res.get("out_dir",""),
                    n_rows      = n_rows,
                )
                self.st["last_result"] = self.res.get("out_dir","")
                self.log.append(f"✓ Run registrada en BD con ID={rid}")
            except Exception as e:
                self.log.append(f"⚠ No se pudo registrar run: {e}")
        self.running = False; self.done = True

    def key(self, k):
        # dentro del picker de config
        if self.mode == "pick_cfg":
            n = len(self.cfgs)
            if   k in (curses.KEY_UP,   ord('k')) and self.cfg_cur > 0:   self.cfg_cur -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and self.cfg_cur < n-1: self.cfg_cur += 1
            elif k in (ord('\n'), ord('\r'), 10, 13) and n > 0:
                ld = db_load(self.cfgs[self.cfg_cur]["id"])
                if ld:
                    self.st.update(ld)
                    self.notif = f"✓ Config cargada: {ld.get('_db_name','?')}"; self.ntimer = 80
                self.mode = "normal"
            elif k in (27, ord('q')): self.mode = "normal"
            return None

        # dentro del picker de CSV
        if self.mode == "pick_csv":
            found = _find_csvs(self.ibuf)
            if ord('1') <= k <= ord('9'):
                idx = k - ord('1')
                if idx < len(found): self.ibuf = str(found[idx].resolve())
            elif k in (ord('\n'), ord('\r'), 10, 13):
                self.st["input_csv"] = self.ibuf.strip()
                self.notif = f"✓ CSV: {tr(self.ibuf, 52)}"; self.ntimer = 70
                self.mode = "normal"
            elif k == 27: self.mode = "normal"; self.ibuf = ""
            elif k in (curses.KEY_BACKSPACE, 127, 8): self.ibuf = self.ibuf[:-1]
            elif 32 <= k <= 126: self.ibuf += chr(k)
            return None

        # modo normal
        if k in (ord('L'),):              self.sub = "log"
        elif k in (ord('S'),):            self.sub = "summary"
        elif k in (ord('l'),):            self.sub = "list"; self.list_scroll = 0
        elif k in (ord('e'), ord('E')) and not self.running:
            self.sub = "log"
            self._thr = threading.Thread(target=self._worker, daemon=True); self._thr.start()
        elif k in (ord('c'), ord('C')) and not self.running:
            self._reload_cfgs(); self.mode = "pick_cfg"
        elif k in (ord('f'), ord('F')) and not self.running:
            self.ibuf = self.st.get("input_csv",""); self.mode = "pick_csv"
        elif k in (curses.KEY_UP,   ord('k')):
            if   self.sub == "log":  self.scroll = max(0, self.scroll-1)
            elif self.sub == "list": self.list_scroll = max(0, self.list_scroll-1)
        elif k in (curses.KEY_DOWN, ord('j')):
            if   self.sub == "log":  self.scroll += 1
            elif self.sub == "list": self.list_scroll += 1
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 9  PANTALLA 4 — GRÁFICOS ASCII
#       Selector de run BD → 8 gráficos
# ══════════════════════════════════════════════════════════════════════════════
class GraphScreen(Screen):
    GRAPHS = [
        ("Distribución WSum",           "wsum"),
        ("Distribución Tcheby",         "tcheby"),
        ("Distribución Hipervolumen",   "hv"),
        ("Scatter Pareto",              "scatter"),
        ("Niveles Pareto",              "levels"),
        ("Top-10 WSum",                 "top10"),
        ("Correlación Rankings",        "corr"),
        ("Radar Top-5 configuraciones", "radar"),
    ]
    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.gi     = 0; self.df = None; self.dfp = ""; self.canvas = []; self.scroll = 0
        self.errmsg = ""
        self.mode   = "normal"   # normal | pick_run
        self.runs   = []; self.run_cur = 0; self.run_scroll = 0
        self.cur_run= {}          # run actualmente seleccionada

    def _reload_runs(self):
        self.runs = run_list()
        # pre-seleccionar la run más reciente del last_result actual
        lr = self.st.get("last_result","")
        for i, r in enumerate(self.runs):
            if r["out_dir"] == lr:
                self.run_cur = i; break

    def _load_from_run(self, run: dict) -> bool:
        out = run.get("out_dir","")
        if not out: self.errmsg="Run sin out_dir"; return False
        cp  = Path(out) / "multiobjective_report.csv"
        if not cp.exists(): self.errmsg = f"CSV no encontrado: {cp}"; return False
        if self.dfp == str(cp): return True
        try:
            self.df = pd.read_csv(cp); self.dfp = str(cp); self.errmsg = ""; return True
        except Exception as e:
            self.errmsg = f"Error CSV: {e}"; return False

    def _load(self):
        # prioridad: run seleccionada manualmente
        if self.cur_run:
            return self._load_from_run(self.cur_run)
        # fallback: last_result del estado
        res = self.st.get("last_result","")
        if not res:
            dirs = sorted(Path(".").glob("multiobjective_report_*"), reverse=True)
            if dirs: res = str(dirs[0])
        if not res: self.errmsg = "Sin resultados. Elige una run con [r] o ejecuta P3 → [e]."; return False
        return self._load_from_run({"out_dir": res})

    def _build(self):
        self.canvas = []
        if not _NUM: self.canvas = ["  ✗ numpy/pandas no disponibles."]; return
        if not self._load():
            self.canvas = ["", f"  ⚠  {self.errmsg}", "",
                           "  Para ver gráficos:",
                           "  1. Ejecuta el análisis en P3 → [e]",
                           "  2. O elige una run con [r] (selector de runs BD)"]; return
        k = self.GRAPHS[self.gi][1]; df = self.df
        try:
            if   k == "wsum":    self.canvas = self._hist(df, "WSum",        "Distribución WSum  (↑ mayor=mejor)")
            elif k == "tcheby":  self.canvas = self._hist(df, "Tcheby",      "Distribución Tcheby  (↓ menor=mejor)")
            elif k == "hv":      self.canvas = self._hist(df, "Hypervolume", "Distribución Hipervolumen")
            elif k == "scatter": self.canvas = self._scatter(df)
            elif k == "levels":  self.canvas = self._levels(df)
            elif k == "top10":   self.canvas = self._top10(df)
            elif k == "corr":    self.canvas = self._corr(df)
            elif k == "radar":   self.canvas = self._radar(df)
        except Exception as e:
            self.canvas = [f"  Error generando gráfico: {e}"]

    # ── gráficos (idénticos a versión anterior) ───────────────────────────────
    def _hist(self, df, col, title, B=24, H=13):
        if col not in df.columns: return [f"  '{col}' no disponible"]
        v = df[col].dropna().values
        if not len(v): return ["  Sin datos"]
        mn,mx = float(v.min()), float(v.max()); rng = mx-mn if mx!=mn else 1e-9
        cts = [0]*B
        for x in v: cts[min(int((x-mn)/rng*B), B-1)] += 1
        mc = max(cts) if max(cts) > 0 else 1
        lines = [f"  {title}",
                 f"  n={len(v)}  min={mn:.4f}  max={mx:.4f}  μ={v.mean():.4f}  σ={v.std():.4f}", ""]
        for ri in range(H, 0, -1):
            thr = ri/H*mc
            lines.append("  │"+"".join("█" if c>=thr else " " for c in cts))
        lines += ["  └"+"─"*B, f"  {mn:.3f}"+" "*(B-12)+f"{mx:.3f}", ""]
        if "Pareto" in df.columns:
            pv = df.loc[df["Pareto"], col].values; nv = df.loc[~df["Pareto"], col].values
            if len(pv): lines.append(f"  ★ Pareto n={len(pv)}: μ={pv.mean():.4f}  max={pv.max():.4f}")
            if len(nv): lines.append(f"  · Domin. n={len(nv)}: μ={nv.mean():.4f}")
        return lines

    def _scatter(self, df, W=64, H=16):
        nc = [c for c in df.columns if c.startswith("norm_")]
        if len(nc) < 2: return ["  Se necesitan ≥2 columnas norm_*"]
        xc,yc = nc[0],nc[1]; xs=df[xc].values; ys=df[yc].values
        par = df["Pareto"].values if "Pareto" in df.columns else [False]*len(xs)
        xmn,xmx=xs.min(),xs.max(); xrng=xmx-xmn if xmx!=xmn else 1e-9
        ymn,ymx=ys.min(),ys.max(); yrng=ymx-ymn if ymx!=ymn else 1e-9
        g = [[" "]*W for _ in range(H)]
        for x,y,p in zip(xs,ys,par):
            gx=min(int((x-xmn)/xrng*(W-1)),W-1); gy=H-1-min(int((y-ymn)/yrng*(H-1)),H-1)
            if g[gy][gx]==" ": g[gy][gx]="★" if p else "·"
            elif p: g[gy][gx]="★"
        lines=[f"  Scatter: {xc.replace('norm_','')} vs {yc.replace('norm_','')}",
               f"  ★=Pareto(n={int(sum(par))})  ·=Dominada","  ┌"+"─"*W+"┐"]
        for r in g: lines.append("  │"+"".join(r)+"│")
        lines+=["  └"+"─"*W+"┘", f"  {xmn:.2f}"+" "*(W-12)+f"{xmx:.2f}  (eje-X)"]
        return lines

    def _levels(self, df, W=52):
        if "Pareto_Level" not in df.columns: return ["  Sin columna Pareto_Level"]
        cts = df["Pareto_Level"].value_counts().sort_index(); mc = cts.max() if len(cts) else 1
        lines = ["  Niveles de Frente de Pareto", ""]
        for lv,cnt in cts.items():
            bl = max(1, int(cnt/mc*W))
            lines.append(f"  Frente {lv:>2}  │{'█'*bl:<{W}}│  n={cnt:>5}  ({100*cnt/len(df):.1f}%)")
        lines += ["", f"  Frentes: {len(cts)}   Frente-1 (Pareto): {cts.get(1,0)} ({100*cts.get(1,0)/len(df):.1f}%)"]
        return lines

    def _top10(self, df, W=40):
        if "WSum" not in df.columns: return ["  Sin columna WSum"]
        sub = df.nlargest(10,"WSum"); mx = sub["WSum"].max() if len(sub) else 1.0
        if mx == 0: mx = 1.0
        lines = ["  Top-10 por WSum  (★=Pareto-óptima)", ""]
        for i,(_,row) in enumerate(sub.iterrows(),1):
            nm  = str(row.get("combination_name", f"idx={row.name}"))
            val = float(row["WSum"]); bl = max(1,int(val/mx*W)); par="★" if row.get("Pareto",False) else " "
            tch = f" TCH={row['Tcheby']:.3f}" if "Tcheby" in row else ""
            hv  = f" HV={row['Hypervolume']:.4f}" if "Hypervolume" in row else ""
            lines.append(f"  {i:>2}{par} {tr(nm,26):<26}  │{'█'*bl:<{W}}│ {val:.4f}{tch}{hv}")
        return lines

    def _corr(self, df):
        rc = [c for c in ["R_WSum","R_WProd","R_Tcheby","R_ASF"] if c in df.columns]
        if len(rc) < 2: return ["  Se necesitan R_WSum, R_WProd, R_Tcheby, R_ASF"]
        lb = [c.replace("R_","") for c in rc]
        lines = ["  Correlación de Spearman entre Rankings",""]
        hdr   = "          "+"".join(f"{l:>10}" for l in lb)
        lines += [hdr, "  "+"─"*len(hdr)]
        for i,(ci,li) in enumerate(zip(rc,lb)):
            row = f"  {li:<8} "
            for j,cj in enumerate(rc):
                if i==j: row += f"{'1.000':>10}"
                else:
                    r   = _spearman(df[ci].values.tolist(), df[cj].values.tolist())
                    sym = "▓" if abs(r)>0.8 else ("▒" if abs(r)>0.5 else "░")
                    row += f"{r:>8.3f}{sym} "
            lines.append(row)
        lines += ["","  ▓=ρ>0.8  ▒=ρ>0.5  ░=ρ<0.5"]
        return lines

    def _radar(self, df, top=5, BW=30):
        nc = [c for c in df.columns if c.startswith("norm_")]
        if not nc: return ["  Sin columnas norm_*"]
        rk = next((c for c in ["R_Consensus","R_WSum"] if c in df.columns), None)
        if rk is None: return ["  Sin columna de ranking"]
        tops = df.nsmallest(min(top,len(df)), rk)
        lines = [f"  Radar Top-{min(top,len(df))} configuraciones","  (barras [0–1], 1=óptimo)",""]
        for _,row in tops.iterrows():
            nm = str(row.get("combination_name", f"idx={row.name}"))
            par = "★" if row.get("Pareto",False) else " "
            lines.append(f"  {par} {tr(nm,56)}")
            for nc_ in nc:
                val = float(row.get(nc_,0)); bf = max(0,min(int(val*BW),BW))
                lbl = nc_.replace("norm_","")[:16]
                lines.append(f"      {lbl:<16} │{'█'*bf+'░'*(BW-bf)}│ {val:.3f}")
            lines.append("")
        return lines

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase(); self._sz()
        self.title(4, "[←→]Gráfico  [r]Elegir Run  [R]Recargar  [1-8]Directo  [SubG]Gráficos Ejecutados")
        self.statusbar()

        if self.mode == "pick_run":
            self._dpick_run(); self.navbar(); self.scr.refresh(); return

        rs = 3; vis = self.h-6; PW = 28
        hl(self.scr, rs, self.w-PW-2, PW+1)
        sa(self.scr, rs, self.w-PW-1, " GRÁFICOS ", curses.color_pair(cH)|curses.A_BOLD)
        for i,(g,_) in enumerate(self.GRAPHS):
            rw = rs+1+i
            if rw >= self.h-4: break
            at = curses.color_pair(cS)|curses.A_BOLD if i==self.gi else dim()
            sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g,PW-4)}", at)
        # run activa
        run_lbl = tr(self.cur_run.get("config_name","(última run)"), 30) if self.cur_run else "(última run)"
        sa(self.scr, rs+10, self.w-PW-1, f" Run: {tr(run_lbl,PW-6)}", curses.color_pair(cMT))

        DW = self.w-PW-3
        hl(self.scr, rs, 0, DW)
        sa(self.scr, rs, 2, f" {self.GRAPHS[self.gi][0]} ", curses.color_pair(cGR)|curses.A_BOLD)
        if not self.canvas: self._build()
        ms = max(0, len(self.canvas)-vis+1)
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(vis-1):
            idx = self.scroll+i
            if idx >= len(self.canvas): break
            ln = self.canvas[idx]
            if   "★" in ln: at = curses.color_pair(cMX)
            elif "│" in ln and ("█" in ln or "░" in ln): at = curses.color_pair(cGR)
            elif any(c in ln for c in "┌└┐┘"): at = curses.color_pair(cBR)
            elif "Pareto" in ln: at = curses.color_pair(cOK)
            else: at = curses.color_pair(cN)
            sa(self.scr, rs+1+i, 1, tr(ln, DW-2), at)
        hint_bar(self.scr, self.h-3,
                 " [←→]Gráfico  [↑↓]Scroll  [r]Elegir run  [R]Recargar  [1-8]Directo ")
        self.navbar(); self.scr.refresh()

    def _dpick_run(self):
        rs = 3
        sa(self.scr, rs, 2, "SELECCIONAR EJECUCIÓN (RUN) PARA VISUALIZAR",
           curses.color_pair(cH)|curses.A_BOLD)
        sa(self.scr, rs, self.w-36, "(Enter=cargar  Esc=cancelar)", dim())
        hl(self.scr, rs+1, 0, self.w-1)
        sa(self.scr, rs+2, 0,
           f"  {'ID':>4}  {'Config':<22} {'CSV':<32} {'Fecha':<19}  {'Filas':>6}  {'Ok'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+3, 0, self.w-1)
        vis = self.h - rs - 6
        if not self.runs:
            sa(self.scr, rs+5, 4, "Sin runs. Ejecuta el análisis en P3 → [e].", dim()); return
        if self.run_cur >= self.run_scroll+vis-1: self.run_scroll = self.run_cur-vis+2
        if self.run_cur < self.run_scroll:        self.run_scroll = self.run_cur
        self.run_scroll = max(0, self.run_scroll)
        for idx, run in enumerate(self.runs):
            sr = rs+4+(idx-self.run_scroll)
            if not (rs+4 <= sr <= self.h-5): continue
            sel = idx == self.run_cur
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            csv_n = Path(run["input_csv"]).name if run["input_csv"] else "—"
            ok    = "✓" if run["out_dir"] and Path(run["out_dir"]).exists() else "✗"
            ln    = (f"  {run['id']:>4}  {tr(run['config_name'],22):<22} "
                     f"{tr(csv_n,32):<32} {run['ran_at'][:19]:<19}  {run['n_rows']:>6}  {ok}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
        hint_bar(self.scr, self.h-3,
                 " [↑↓]Mover  [Enter]Cargar run  [Esc]Cancelar ")

    def key(self, k):
        if self.mode == "pick_run":
            n = len(self.runs)
            if   k in (curses.KEY_UP,   ord('k')) and self.run_cur > 0:   self.run_cur -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and self.run_cur < n-1: self.run_cur += 1
            elif k in (ord('\n'), ord('\r'), 10, 13) and n > 0:
                self.cur_run = self.runs[self.run_cur]
                self.dfp = ""; self.canvas = []; self.scroll = 0
                self.mode = "normal"
            elif k in (27, ord('q')): self.mode = "normal"
            return None
        # modo normal
        if k == curses.KEY_LEFT:
            self.gi = (self.gi-1) % len(self.GRAPHS); self.canvas = []; self.scroll = 0
        elif k == curses.KEY_RIGHT:
            self.gi = (self.gi+1) % len(self.GRAPHS); self.canvas = []; self.scroll = 0
        elif k in (curses.KEY_UP,   ord('k')): self.scroll = max(0, self.scroll-1)
        elif k in (curses.KEY_DOWN, ord('j')): self.scroll += 1
        elif k in (ord('r'), ord('r')):
            # minúscula r = pick_run
            if k == ord('r'):
                self._reload_runs(); self.mode = "pick_run"
            # mayúscula R = recargar
        elif k == ord('R'):
            self.dfp = ""; self.canvas = []; self.scroll = 0
        elif ord('1') <= k <= ord('8'):
            self.gi = k - ord('1'); self.canvas = []; self.scroll = 0
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 10  PANTALLA 5 — TABLAS DE RESULTADOS
#        Selector de run → tabla paginada + sort por cualquier columna
# ══════════════════════════════════════════════════════════════════════════════
class TableScreen(Screen):
    # Columnas de la tabla (etiqueta, nombre columna DataFrame)
    TABLE_COLS = [
        ("Combinación",  "combination_name", 30),
        ("WSum",         "WSum",              9),
        ("WProd",        "WProd",             9),
        ("Tcheby",       "Tcheby",            9),
        ("ASF",          "ASF",               9),
        ("HV",           "Hypervolume",       9),
        ("R_W",          "R_WSum",            6),
        ("R_T",          "R_Tcheby",          6),
        ("R_C",          "R_Consensus",       7),
        ("Pareto",       "Pareto",            7),
        ("Nivel",        "Pareto_Level",      6),
    ]
    PAGE_SIZE = 20   # filas por página

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.mode      = "normal"    # normal | pick_run | detail
        self.runs      = []; self.run_cur = 0; self.run_scroll = 0
        self.cur_run   = {}
        self.df        = None; self.dfp = ""
        self.rows      = []
        self.raw_rows  = []          # filas con valores numéricos originales
        self.sort_col  = 1
        self.sort_asc  = False
        self.page      = 0
        self.cur_row   = 0
        self.header_ci = 0
        self.errmsg    = ""
        self.notif     = ""; self.ntimer = 0
        self.detail_row= None        # fila seleccionada para detalle
        self.detail_scroll = 0
        self._reload_runs()

    def _reload_runs(self):
        self.runs = run_list()
        lr = self.st.get("last_result","")
        for i, r in enumerate(self.runs):
            if r["out_dir"] == lr:
                self.run_cur = i; self.cur_run = r; break

    def _load(self):
        if not self.cur_run: self.errmsg="Elige una run con [r]"; return False
        out = self.cur_run.get("out_dir","")
        cp  = Path(out) / "multiobjective_report.csv"
        if not cp.exists(): self.errmsg=f"CSV no encontrado: {cp}"; return False
        if self.dfp == str(cp): return True
        try:
            self.df = pd.read_csv(cp); self.dfp = str(cp)
            self._rebuild_rows(); self.errmsg = ""; return True
        except Exception as e:
            self.errmsg = f"Error CSV: {e}"; return False

    def _rebuild_rows(self):
        if self.df is None: return
        lbl, col, _ = self.TABLE_COLS[self.sort_col]
        if col in self.df.columns:
            df_s = self.df.sort_values(col, ascending=self.sort_asc, na_position="last")
        else:
            df_s = self.df
        rows = []; raw_rows = []
        for _, row in df_s.iterrows():
            d = {}
            for _, c, _ in self.TABLE_COLS:
                v = row.get(c, "—")
                if isinstance(v, float): d[c] = f"{v:.4f}"
                elif isinstance(v, bool): d[c] = "★" if v else "·"
                else: d[c] = str(v)
            rows.append(d)
            # raw: guardar toda la fila original para el detalle
            raw_rows.append({c: row.get(c) for c in self.df.columns})
        self.rows = rows; self.raw_rows = raw_rows
        total_pages = max(1, (len(rows) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page   = min(self.page, total_pages-1)
        self.cur_row = 0

    def _total_pages(self):
        return max(1, (len(self.rows) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

    def _page_rows(self):
        s = self.page * self.PAGE_SIZE
        return self.rows[s:s+self.PAGE_SIZE]

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase(); self._sz()
        self.title(4, "[r]Run  [←→]Col  [↑↓]Fila  [PgUp/PgDn]Página  [s]Sort  [Enter]Detalle  [SubT]Tablas Ejecutados")
        self.statusbar()

        if self.mode == "pick_run":
            self._dpick_run(); self.navbar(); self.scr.refresh(); return
        if self.mode == "detail":
            self._ddetail(); self.navbar(); self.scr.refresh(); return

        rs = 3
        # intentar cargar si hay run
        if self.cur_run and not self.dfp:
            self._load()

        if not self.rows:
            if not self.cur_run:
                sa(self.scr, rs+2, 4, "Elige una ejecución con [r] para ver los resultados.", dim())
            else:
                if not self._load():
                    sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
            self.navbar(); self.scr.refresh(); return

        run_info = (f"Run #{self.cur_run.get('id','?')}  "
                    f"Config: {tr(self.cur_run.get('config_name','?'),24)}  "
                    f"CSV: {tr(Path(self.cur_run.get('input_csv','')).name,30)}  "
                    f"Filas: {len(self.rows)}")
        sa(self.scr, rs, 0, f" {run_info} "[:self.w-1], curses.color_pair(cMT))

        tp = self._total_pages()
        sort_lbl, sort_col, _ = self.TABLE_COLS[self.sort_col]
        dir_sym = "↑" if self.sort_asc else "↓"
        sa(self.scr, rs+1, 0,
           f" Pág {self.page+1}/{tp}   Sort: {sort_lbl} {dir_sym}   "
           f"[←→]Col-sort  [s]Alternar dir  [PgDn/PgUp/n/p]Página "[:self.w-1],
           curses.color_pair(cH))

        # cabecera tabla
        hdr_row = rs+2
        x = 1
        for ci, (lbl, col, w) in enumerate(self.TABLE_COLS):
            is_sort = ci == self.sort_col
            is_hl   = ci == self.header_ci
            arrow   = dir_sym if is_sort else " "
            cell    = f"{tr(lbl+arrow, w):<{w}}"
            if is_sort and is_hl:
                at = curses.color_pair(cS)|curses.A_BOLD|curses.A_REVERSE
            elif is_sort:
                at = curses.color_pair(cP)|curses.A_BOLD
            elif is_hl:
                at = curses.color_pair(cH)|curses.A_BOLD
            else:
                at = curses.color_pair(cH)
            sa(self.scr, hdr_row, x, cell, at)
            x += w+1
        hl(self.scr, hdr_row+1, 0, self.w-1)

        # filas de datos
        page_rows = self._page_rows()
        for ri, row in enumerate(page_rows):
            sr = hdr_row+2+ri
            if sr >= self.h-4: break
            sel = ri == self.cur_row
            base_at = curses.color_pair(cS) if sel else curses.color_pair(cN)
            sa(self.scr, sr, 0, " "*(self.w-1), base_at)
            x = 1
            for ci, (lbl, col, w) in enumerate(self.TABLE_COLS):
                val = row.get(col, "—")
                # color especial por columna
                if sel:
                    at = curses.color_pair(cS)|curses.A_BOLD if ci==self.header_ci else base_at
                else:
                    if col == "Pareto":
                        at = curses.color_pair(cMX)|curses.A_BOLD if val=="★" else curses.color_pair(cDM)
                    elif col in ("WSum","WProd"):
                        at = curses.color_pair(cOK)
                    elif col in ("Tcheby","ASF"):
                        at = curses.color_pair(cMN)
                    elif col == "Hypervolume":
                        at = curses.color_pair(cMT)
                    elif col.startswith("R_"):
                        at = curses.color_pair(cBR)
                    elif col == "Pareto_Level":
                        at = curses.color_pair(cP)
                    else:
                        at = base_at
                sa(self.scr, sr, x, tr(val, w), at)
                x += w+1

        # barra paginación
        hl(self.scr, self.h-5, 0, self.w-1)
        page_info = (f" Mostrando filas {self.page*self.PAGE_SIZE+1}–"
                     f"{min((self.page+1)*self.PAGE_SIZE, len(self.rows))} "
                     f"de {len(self.rows)} totales   Pág {self.page+1}/{tp} ")
        sa(self.scr, self.h-4, 0, page_info[:self.w-1], curses.color_pair(cH))

        if self.notif and self.ntimer > 0:
            sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1],
               curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer -= 1
        else:
            hint_bar(self.scr, self.h-3,
                     " [r]Elegir run  [←→]Columna sort  [s]Toggle dir  [↑↓]Fila  [n/p]Página  [Enter]Detalle ")
        self.navbar(); self.scr.refresh()

    def _ddetail(self):
        """Detalle completo de una fila: valores, columnas CSV involucradas y fórmulas."""
        rs = 3; raw = self.detail_row
        if raw is None:
            sa(self.scr, rs+2, 4, "Sin fila seleccionada.", dim()); return

        combo = str(raw.get("combination_name", raw.get("combination_id","?")))
        sa(self.scr, rs, 0,
           f" DETALLE: {tr(combo, self.w-12)} "[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+1, 0, self.w-1)

        objs = [o for o in self.st["objectives"] if o["enabled"]]

        lines = []
        def sec(title, color=cP):
            lines.append(("hdr", title, color))
        def row_line(label, val, formula="", color=cN):
            lines.append(("row", label, val, formula, color))
        def blank():
            lines.append(("blank",))

        # 1. IDENTIFICACIÓN
        sec("IDENTIFICACIÓN")
        for ck in ["combination_id","combination_name","det_model","cls_model","vlm_model","llm_model"]:
            v = raw.get(ck)
            if v is not None and str(v) not in ("nan","None",""):
                row_line(ck, str(v), "", cMT)
        blank()

        # 2. COLUMNAS CSV ACTIVAS
        sec("COLUMNAS CSV ACTIVAS  (valores originales y normalizados)")
        for o in objs:
            col  = o["col"]; v = raw.get(col)
            vstr = f"{float(v):.6f}" if v is not None and str(v) not in ("nan","None") else "N/A"
            nv   = raw.get(f"norm_{col}")
            nstr = f"{float(nv):.6f}" if nv is not None and str(nv) not in ("nan","None") else "N/A"
            dir_s = "↑ max" if o["dir"]=="max" else "↓ min"
            row_line(f"  {tr(o['lbl'],22):<22} ({col})",
                     f"raw={vstr}  norm={nstr}",
                     f"w={o['weight']:.4f}  {dir_s}",
                     cOK if o["dir"]=="max" else cMN)
        blank()

        # 3. NORMALIZACIÓN
        sec("NORMALIZACIÓN")
        row_line("  Fórmula (dir=max)", "norm = (x − min) / (max − min)", "", cDM)
        row_line("  Fórmula (dir=min)", "norm = (max − x) / (max − min)", "", cDM)
        row_line("  Rango resultado",   "[0, 1]  donde 1 = valor óptimo", "", cDM)
        blank()

        # 4. ESCALARIZACIÓN
        sec("ESCALARIZACIÓN")
        rho  = self.st.get("rho", 0.001)
        w_s  = " + ".join(f"{o['weight']:.3f}·n_{o['col'][-6:]}" for o in objs[:4])
        if len(objs)>4: w_s += " + …"
        ws = raw.get("WSum"); wp = raw.get("WProd")
        tc = raw.get("Tcheby"); af = raw.get("ASF")
        if ws is not None:
            row_line("WSum  ↑ mayor=mejor", f"{float(ws):.6f}",
                     f"Σ(wᵢ·normᵢ) = {w_s}", cOK)
        if wp is not None:
            row_line("WProd ↑ mayor=mejor", f"{float(wp):.6f}",
                     "Π(normᵢ^wᵢ)  producto de potencias", cOK)
        if tc is not None:
            row_line("Tcheby ↓ menor=mejor", f"{float(tc):.6f}",
                     "max_i( wᵢ·|1−normᵢ| )  distancia Chebyshev", cMN)
        if af is not None:
            row_line("ASF    ↓ menor=mejor", f"{float(af):.6f}",
                     f"max((1−n)/w) + ρ·Σ(1−n)/w   ρ={rho}", cMN)
        blank()

        # 5. HIPERVOLUMEN
        sec("HIPERVOLUMEN")
        hv = raw.get("Hypervolume")
        if hv is not None:
            row_line("HV por fila", f"{float(hv):.6f}",
                     "vol. espacio dominado; punto referencia=[1.2,…,1.2]", cMT)
        blank()

        # 6. RANKINGS
        sec("RANKINGS  (posición en el conjunto — 1 = mejor)")
        for rc, dirstr in [("R_WSum","↓ rank WSum desc"),("R_WProd","↓ rank WProd desc"),
                            ("R_Tcheby","↑ rank Tcheby asc"),("R_ASF","↑ rank ASF asc"),
                            ("R_Consensus","promedio de los 4 rankings")]:
            v = raw.get(rc)
            if v is not None and str(v) not in ("nan","None"):
                row_line(f"  {rc}", f"{v}", dirstr, cBR)
        blank()

        # 7. PARETO
        sec("PARETO")
        pareto = raw.get("Pareto"); plevel = raw.get("Pareto_Level")
        pstr   = "★ Pareto-óptima (no dominada)" if pareto else "· Dominada"
        row_line("Estado Pareto", pstr,
                 "dominada = existe otra config mejor en al menos 1 objetivo sin empeorar otra",
                 cMX if pareto else cDM)
        if plevel is not None and str(plevel) not in ("nan","None"):
            row_line("Nivel frente Pareto", f"{int(float(plevel))}",
                     "1=primer frente (óptimo)  2=segundo frente tras retirar frente-1…", cP)

        # ── render ───────────────────────────────────────────────────────────
        vis = self.h - rs - 5
        self.detail_scroll = max(0, min(self.detail_scroll, max(0, len(lines)-vis)))
        row = rs+2
        for i in range(vis):
            idx = self.detail_scroll + i
            if idx >= len(lines): break
            entry = lines[idx]
            if entry[0] == "hdr":
                hl(self.scr, row, 0, self.w-1)
                sa(self.scr, row, 1, f" {entry[1]} ", curses.color_pair(entry[2])|curses.A_BOLD)
            elif entry[0] == "row":
                _, lbl, val, formula, color = entry
                lw = 36; vw = 16
                sa(self.scr, row, 2,    tr(lbl, lw),  curses.color_pair(color)|curses.A_BOLD)
                sa(self.scr, row, 2+lw, tr(str(val), vw), curses.color_pair(color))
                if formula:
                    sa(self.scr, row, 2+lw+vw+1, tr(formula, self.w-lw-vw-6), dim())
            row += 1
        # scrollbar
        tot = len(lines)
        if tot > vis:
            for i in range(vis):
                f = (self.detail_scroll+i)/max(1,tot)
                sa(self.scr, rs+2+i, self.w-2,
                   "█" if f<(self.detail_scroll+vis/2)/max(1,tot) else "░", dim())
        hint_bar(self.scr, self.h-3,
                 " [↑↓/j/k]Scroll detalle  [Esc/q]Volver a tabla ")

    def _dpick_run(self):
        rs = 3
        sa(self.scr, rs, 2, "SELECCIONAR EJECUCIÓN PARA VER TABLA",
           curses.color_pair(cH)|curses.A_BOLD)
        sa(self.scr, rs, self.w-32, "(Enter=ver  Esc=cancelar)", dim())
        hl(self.scr, rs+1, 0, self.w-1)
        sa(self.scr, rs+2, 0,
           f"  {'ID':>4}  {'Config':<22} {'CSV':<34} {'Fecha':<19}  {'Filas':>6}  {'Ok'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+3, 0, self.w-1)
        vis = self.h - rs - 7
        if not self.runs:
            sa(self.scr, rs+5, 4, "Sin runs. Ejecuta el análisis en P3 → [e].", dim()); return
        if self.run_cur >= self.run_scroll+vis-1: self.run_scroll = self.run_cur-vis+2
        if self.run_cur < self.run_scroll:        self.run_scroll = self.run_cur
        self.run_scroll = max(0, self.run_scroll)
        for idx, run in enumerate(self.runs):
            sr = rs+4+(idx-self.run_scroll)
            if not (rs+4 <= sr <= self.h-5): continue
            sel = idx == self.run_cur
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            csv_n = Path(run["input_csv"]).name if run["input_csv"] else "—"
            ok    = "✓" if run["out_dir"] and Path(run["out_dir"]).exists() else "✗"
            col_ok= curses.color_pair(cOK) if ok=="✓" else curses.color_pair(cER)
            ln    = (f"  {run['id']:>4}  {tr(run['config_name'],22):<22} "
                     f"{tr(csv_n,34):<34} {run['ran_at'][:19]:<19}  {run['n_rows']:>6}  {ok}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
        hint_bar(self.scr, self.h-3,
                 " [↑↓]Mover  [Enter]Seleccionar  [Esc]Cancelar ")

    def key(self, k):
        if self.mode == "pick_run":
            n = len(self.runs)
            if   k in (curses.KEY_UP,   ord('k')) and self.run_cur > 0:   self.run_cur -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and self.run_cur < n-1: self.run_cur += 1
            elif k in (ord('\n'), ord('\r'), 10, 13) and n > 0:
                self.cur_run = self.runs[self.run_cur]
                self.dfp = ""; self.rows = []; self.raw_rows = []
                self.page = 0; self.cur_row = 0
                self._load()
                self.mode = "normal"
            elif k in (27, ord('q')): self.mode = "normal"
            return None

        if self.mode == "detail":
            if   k in (curses.KEY_UP,   ord('k')): self.detail_scroll = max(0, self.detail_scroll-1)
            elif k in (curses.KEY_DOWN, ord('j')): self.detail_scroll += 1
            elif k in (27, ord('q'), ord('\t')): self.mode = "normal"
            elif k == ord('\t'): return "next"
            return None

        # modo normal
        tp = self._total_pages()
        n  = len(self._page_rows())

        if k in (ord('r'), ord('R')):
            self._reload_runs(); self.mode = "pick_run"
        elif k == curses.KEY_LEFT:
            self.header_ci = max(0, self.header_ci-1)
            self.sort_col = self.header_ci; self._rebuild_rows()
        elif k == curses.KEY_RIGHT:
            self.header_ci = min(len(self.TABLE_COLS)-1, self.header_ci+1)
            self.sort_col = self.header_ci; self._rebuild_rows()
        elif k in (ord('s'), ord('S')):
            self.sort_asc = not self.sort_asc; self._rebuild_rows()
            lbl,_,_ = self.TABLE_COLS[self.sort_col]
            self.notif = f"Sort: {lbl} {'↑ ASC' if self.sort_asc else '↓ DESC'}"; self.ntimer = 50
        elif k in (curses.KEY_UP,   ord('k')) and n > 0:
            self.cur_row = max(0, self.cur_row-1)
        elif k in (curses.KEY_DOWN, ord('j')) and n > 0:
            self.cur_row = min(n-1, self.cur_row+1)
        elif k in (curses.KEY_NPAGE, ord('n')):
            if self.page < tp-1: self.page += 1; self.cur_row = 0
        elif k in (curses.KEY_PPAGE, ord('p')):
            if self.page > 0:    self.page -= 1; self.cur_row = 0
        elif k in (curses.KEY_HOME,): self.page = 0; self.cur_row = 0
        elif k in (curses.KEY_END,):  self.page = tp-1; self.cur_row = 0
        elif k in (ord('\n'), ord('\r'), 10, 13):
            # Abrir detalle de la fila actual
            abs_idx = self.page * self.PAGE_SIZE + self.cur_row
            if abs_idx < len(self.raw_rows):
                self.detail_row    = self.raw_rows[abs_idx]
                self.detail_scroll = 0
                self.mode = "detail"
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 11  PANTALLA 6 — REPOSITORIO CSV
#        Campos del CSV por fase · paginación · sort · columnas vacías
# ══════════════════════════════════════════════════════════════════════════════
class RepositoryScreen(Screen):
    PAGE_SIZE = 22

    # Fases conocidas y sus prefijos de columna
    PHASE_PREFIXES = [
        ("Detección",      ["det_"]),
        ("Clasificación",  ["cls_"]),
        ("VLM",            ["vlm_"]),
        ("LLM",            ["llm_"]),
        ("Sistema",        ["total_","pipeline_","system_","latency_","mem_","gpu_"]),
        ("Identificación", ["combination_","det_model","cls_model","vlm_model","llm_model"]),
    ]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.mode       = "normal"   # normal | pick_csv
        self.csv_path   = st.get("input_csv","")
        self.df         = None; self.dfp = ""
        self.all_cols   = []         # todas las columnas del CSV
        self.filter_phase = "Todas" # fase seleccionada
        self.filter_empty = False   # mostrar solo columnas con vacíos
        self.sort_col_idx = 0       # índice de columna de sort de la meta-tabla
        self.sort_asc   = True
        self.page       = 0
        self.cur_row    = 0
        self.ibuf       = ""
        self.errmsg     = ""
        self.notif      = ""; self.ntimer = 0
        self.phase_idx  = 0
        self.phases     = ["Todas"] + [p for p,_ in self.PHASE_PREFIXES] + ["Otras"]
        self.meta_rows  = []         # filas de la meta-tabla: info por columna

    # Columnas de la meta-tabla
    META_COLS = [
        ("Columna CSV",  "col",    28),
        ("Fase",         "phase",  14),
        ("Tipo",         "dtype",   8),
        ("No-nulos",     "notnull", 9),
        ("% Vacíos",     "pct_na", 10),
        ("Min",          "min",    10),
        ("Max",          "max",    10),
        ("Media",        "mean",   10),
        ("Ejemplo",      "sample", 20),
    ]

    def _load(self):
        cp = self.csv_path
        if not cp or not Path(cp).exists():
            self.errmsg = f"CSV no encontrado: {cp or '(ninguno)'}"; return False
        if self.dfp == cp: return True
        try:
            self.df = pd.read_csv(cp, nrows=5000)   # limitar para velocidad
            self.dfp = cp
            self._build_meta()
            self.errmsg = ""; return True
        except Exception as e:
            self.errmsg = f"Error: {e}"; return False

    def _phase_of(self, col):
        for ph, pf in self.PHASE_PREFIXES:
            for p in pf:
                if col.startswith(p) or col == p:
                    return ph
        return "Otras"

    def _build_meta(self):
        if self.df is None: return
        rows = []
        for col in self.df.columns:
            s    = self.df[col]
            nn   = int(s.notna().sum())
            tot  = len(s)
            pna  = round(100*(tot-nn)/tot, 1) if tot else 0
            dt   = str(s.dtype)[:6]
            ph   = self._phase_of(col)
            mn   = mx = mu = "—"
            try:
                nv = pd.to_numeric(s, errors="coerce")
                if nv.notna().any():
                    mn = f"{nv.min():.4f}"; mx = f"{nv.max():.4f}"; mu = f"{nv.mean():.4f}"
            except Exception: pass
            ex = str(s.dropna().iloc[0])[:18] if nn > 0 else "—"
            rows.append({"col":col,"phase":ph,"dtype":dt,"notnull":str(nn),
                         "pct_na":str(pna),"min":mn,"max":mx,"mean":mu,"sample":ex})
        self.meta_rows = rows
        self._apply_filter()

    def _apply_filter(self):
        ph = self.phases[self.phase_idx]
        rows = self.meta_rows
        if ph != "Todas":
            rows = [r for r in rows if r["phase"] == ph]
        if self.filter_empty:
            rows = [r for r in rows if float(r["pct_na"]) > 0]
        # sort
        if self.sort_col_idx < len(self.META_COLS):
            key = self.META_COLS[self.sort_col_idx][1]
            try:
                rows = sorted(rows, key=lambda x: float(x[key]), reverse=not self.sort_asc)
            except Exception:
                rows = sorted(rows, key=lambda x: str(x[key]), reverse=not self.sort_asc)
        self.filt_rows = rows
        tp = max(1,(len(rows)+self.PAGE_SIZE-1)//self.PAGE_SIZE)
        self.page = min(self.page, tp-1); self.cur_row = 0

    def _total_pages(self): return max(1,(len(self.filt_rows)+self.PAGE_SIZE-1)//self.PAGE_SIZE)
    def _page_rows(self):
        s = self.page*self.PAGE_SIZE; return self.filt_rows[s:s+self.PAGE_SIZE]

    def draw(self):
        self.scr.erase(); self._sz()
        self.title(2, "[f]CSV  [←→]Fase  [e]Toggle-vacíos  [↑↓]Fila  [n/p]Pág  [SubL]Listas CSV")
        self.statusbar()
        if self.mode == "pick_csv":
            self._dpick(); self.navbar(); self.scr.refresh(); return

        rs = 3
        if not self.dfp:
            self._load()
        if not self.dfp:
            sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
            hint_bar(self.scr, self.h-3, " [f]Elegir CSV  [Tab]Siguiente ")
            self.navbar(); self.scr.refresh(); return

        # Info cabecera
        n_cols = len(self.meta_rows); n_rows = len(self.df)
        n_na   = sum(1 for r in self.meta_rows if float(r["pct_na"]) > 0)
        ph     = self.phases[self.phase_idx]
        filt_lbl = f"Fase:{ph}  {'[solo vacíos]' if self.filter_empty else ''}"
        sa(self.scr, rs, 0,
           f" CSV: {tr(Path(self.dfp).name,30)}  Filas:{n_rows}  Cols:{n_cols}  "
           f"Con-vacíos:{n_na}  {filt_lbl} "[:self.w-1],
           curses.color_pair(cMT))

        tp = self._total_pages()
        sa(self.scr, rs+1, 0,
           f" Pág {self.page+1}/{tp}  Mostrando {len(self.filt_rows)} cols  "
           f"[←→]Fase  [e]Vacíos  [s]Sort  [n/p]Página "[:self.w-1],
           curses.color_pair(cH))

        # Cabecera meta-tabla
        hr = rs+2; x = 1
        for ci,(lbl,_,w) in enumerate(self.META_COLS):
            is_s = ci==self.sort_col_idx
            arr  = ("↑" if self.sort_asc else "↓") if is_s else " "
            at   = curses.color_pair(cP)|curses.A_BOLD if is_s else curses.color_pair(cH)|curses.A_BOLD
            sa(self.scr, hr, x, tr(lbl+arr,w), at); x += w+1
        hl(self.scr, hr+1, 0, self.w-1)

        # Filas
        for ri, row in enumerate(self._page_rows()):
            sr = hr+2+ri
            if sr >= self.h-4: break
            sel = ri == self.cur_row
            base_at = curses.color_pair(cS) if sel else curses.color_pair(cN)
            sa(self.scr, sr, 0, " "*(self.w-1), base_at)
            x = 1
            for ci,(lbl,key,w) in enumerate(self.META_COLS):
                val = row.get(key,"—")
                if sel:
                    at = base_at
                elif key=="pct_na":
                    pv = float(val) if val!="—" else 0
                    at = curses.color_pair(cER)|curses.A_BOLD if pv>20 \
                         else (curses.color_pair(cMT) if pv>0 else curses.color_pair(cOK))
                elif key=="phase":
                    ph_colors={"Detección":cOK,"Clasificación":cMX,"VLM":cMT,"LLM":cP,
                                "Sistema":cH,"Identificación":cTI,"Otras":cDM}
                    at = curses.color_pair(ph_colors.get(val,cN))
                elif key=="mean":
                    at = curses.color_pair(cH)
                else:
                    at = base_at
                sa(self.scr, sr, x, tr(val,w), at); x += w+1

        hl(self.scr, self.h-5, 0, self.w-1)
        sa(self.scr, self.h-4, 0,
           f" Filas {self.page*self.PAGE_SIZE+1}–{min((self.page+1)*self.PAGE_SIZE,len(self.filt_rows))} "
           f"de {len(self.filt_rows)} cols "[:self.w-1], curses.color_pair(cH))
        if self.notif and self.ntimer>0:
            sa(self.scr,self.h-3,0,f" {self.notif} "[:self.w-1],curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer-=1
        else:
            hint_bar(self.scr, self.h-3,
                     " [f]CSV  [←→]Fase  [e]Toggle-vacíos  [s]Sort  [↑↓]Fila  [n/p]Página ")
        self.navbar(); self.scr.refresh()

    def _dpick(self):
        rs=5; b=curses.color_pair(cBR)
        found = _find_csvs(self.ibuf)
        sa(self.scr,rs,4,"╔"+"═"*68+"╗",b)
        sa(self.scr,rs+1,4,"║  CSV DE ENTRADA (results_YYYYMMDD / res_F*_YYYYMMDD.csv)"+" "*12+"║",b)
        sa(self.scr,rs+2,4,"╠"+"═"*68+"╣",b)
        sa(self.scr,rs+3,6,f"Ruta: {self.ibuf}█",curses.color_pair(cP)|curses.A_BOLD)
        ok=Path(self.ibuf).exists() if self.ibuf else False
        sa(self.scr,rs+4,6,
           ("✓ Encontrado" if ok else "✗ No encontrado") if self.ibuf else "(escribe o elige de la lista)",
           curses.color_pair(cOK) if ok else curses.color_pair(cER if self.ibuf else cDM))
        if found:
            sa(self.scr,rs+5,6,f"CSVs encontrados ({len(found)}):",dim())
            for i,p in enumerate(found):
                at = curses.color_pair(cMX)|curses.A_BOLD if str(p)==self.ibuf else curses.color_pair(cMT)
                sa(self.scr,rs+6+i,8,tr(f"{i+1}. {p}",self.w-12),at)
        bot=rs+6+len(found)+2
        sa(self.scr,bot,4,"║  [Enter]Aplicar  [Esc]Cancelar  [1-9]Elegir lista" +" "*19+"║",dim())
        sa(self.scr,bot+1,4,"╚"+"═"*68+"╝",b)
        hint_bar(self.scr,self.h-3," [1-9]Elegir  [Enter]Aplicar  [Esc]Cancelar ")

    def key(self, k):
        if self.mode=="pick_csv":
            found=_find_csvs(self.ibuf)
            if ord('1')<=k<=ord('9'):
                idx=k-ord('1')
                if idx<len(found): self.ibuf=str(found[idx].resolve())
            elif k in (ord('\n'),ord('\r'),10,13):
                self.csv_path=self.ibuf.strip(); self.st["input_csv"]=self.csv_path
                self.dfp=""; self.meta_rows=[]; self.filt_rows=[]
                self.notif=f"✓ CSV: {tr(self.ibuf,50)}"; self.ntimer=60
                self.mode="normal"; self._load()
            elif k==27: self.mode="normal"; self.ibuf=""
            elif k in (curses.KEY_BACKSPACE,127,8): self.ibuf=self.ibuf[:-1]
            elif 32<=k<=126: self.ibuf+=chr(k)
            return None
        n=len(self._page_rows()) if hasattr(self,"filt_rows") else 0
        tp=self._total_pages() if hasattr(self,"filt_rows") else 1
        if k in (ord('f'),ord('F')): self.ibuf=self.csv_path; self.mode="pick_csv"
        elif k==curses.KEY_LEFT:
            self.phase_idx=max(0,self.phase_idx-1); self._apply_filter()
        elif k==curses.KEY_RIGHT:
            self.phase_idx=min(len(self.phases)-1,self.phase_idx+1); self._apply_filter()
        elif k in (ord('e'),ord('E')):
            self.filter_empty=not self.filter_empty; self._apply_filter()
            self.notif=f"Filtro vacíos: {'ON' if self.filter_empty else 'OFF'}"; self.ntimer=50
        elif k in (ord('s'),ord('S')):
            self.sort_asc=not self.sort_asc; self._apply_filter()
        elif k in (curses.KEY_UP,ord('k')) and n>0:   self.cur_row=max(0,self.cur_row-1)
        elif k in (curses.KEY_DOWN,ord('j')) and n>0: self.cur_row=min(n-1,self.cur_row+1)
        elif k in (curses.KEY_NPAGE,ord('n')): 
            if self.page<tp-1: self.page+=1; self.cur_row=0
        elif k in (curses.KEY_PPAGE,ord('p')):
            if self.page>0: self.page-=1; self.cur_row=0
        elif k==ord('\t'): return "next"
        elif k in (ord('q'),27): return "prev"
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 12  PANTALLA 7 — ESTADÍSTICA CSV
#        Distribución · normalidad · pruebas paramétricas/no paramétricas
# ══════════════════════════════════════════════════════════════════════════════
class StatScreen(Screen):
    """
    Sub-tabs:
      [A] Resumen estadístico  — media, moda, σ, asimetría, curtosis
      [B] Distribución         — histograma ASCII + test normalidad (Shapiro-Wilk)
      [C] Pruebas              — paramétricas vs no paramétricas por columna
      [D] Homogeneidad         — Levene entre fases
    """
    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub       = 0           # 0=A 1=B 2=C 3=D
        self.csv_path  = st.get("input_csv","")
        self.df        = None; self.dfp = ""
        self.num_cols  = []          # columnas numéricas
        self.col_idx   = 0           # columna actualmente seleccionada
        self.col_scroll= 0
        self.scroll    = 0
        self.canvas    = []
        self.errmsg    = ""
        self.mode      = "normal"    # normal | pick_csv
        self.ibuf      = ""
        self.notif     = ""; self.ntimer = 0
        self.phase_idx = 0
        self.phases    = ["Todas"] + [p for p,_ in RepositoryScreen.PHASE_PREFIXES] + ["Otras"]

    def _load(self):
        cp = self.csv_path
        if not cp or not Path(cp).exists():
            self.errmsg = f"CSV no encontrado: {cp or '(ninguno)'}"; return False
        if self.dfp == cp: return True
        try:
            self.df = pd.read_csv(cp, nrows=5000)
            self.dfp = cp
            all_num = [c for c in self.df.columns
                       if pd.api.types.is_numeric_dtype(self.df[c])]
            # Ordenar por fases según nombre del CSV
            csv_name = Path(cp).stem.upper()
            PHASE_ORDER = [
                ("det_", "F1" in csv_name),
                ("cls_", "F2" in csv_name),
                ("vlm_", "F3" in csv_name),
                ("llm_", "F4" in csv_name),
            ]
            ordered = []
            for prefix, active in PHASE_ORDER:
                if active:
                    ordered.extend([c for c in all_num if c.startswith(prefix)
                                    and c not in ordered])
            rest = [c for c in all_num if c not in ordered]
            self.num_cols = ordered + rest
            self.col_idx = 0; self.canvas = []; self.errmsg = ""
            return True
        except Exception as e:
            self.errmsg = f"Error: {e}"; return False

    def _col(self):
        if not self.num_cols: return None, None
        c = self.num_cols[self.col_idx]
        v = pd.to_numeric(self.df[c], errors="coerce").dropna().values
        return c, v

    # ── cálculos estadísticos ─────────────────────────────────────────────────
    def _stats(self, v):
        """Devuelve dict con estadísticos básicos."""
        if not _NUM or v is None or len(v)<2: return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy import stats as sp_stats
            v = np.asarray(v, dtype=float)
            n = len(v)
            mu  = float(np.mean(v))
            med = float(np.median(v))
            rng = float(v.max()-v.min())
            if rng == 0:
                mo = mu; sd = sk = ku = 0.0; mn = mx = q1 = q3 = mu
                return {"n":n,"mean":mu,"median":med,"mode":mo,"std":sd,"skew":sk,"kurt":ku,
                        "min":mn,"max":mx,"q1":q1,"q3":q3,
                        "sw_s":1.0,"sw_p":1.0,"ks_s":None,"ks_p":None,"_constant":True}
            sd = float(np.std(v, ddof=1))
            try:   sk = float(sp_stats.skew(v))
            except Exception: sk = 0.0
            try:   ku = float(sp_stats.kurtosis(v))
            except Exception: ku = 0.0
            mn = float(v.min()); mx = float(v.max())
            q1 = float(np.percentile(v, 25)); q3 = float(np.percentile(v, 75))
            try:   mo = float(sp_stats.mode(v, keepdims=True).mode[0])
            except Exception: mo = mu
            sw_s = sw_p = None
            if 3 <= n <= 5000:
                try:
                    sw_s, sw_p = sp_stats.shapiro(v[:5000])
                    sw_s = float(sw_s); sw_p = float(sw_p)
                except Exception: pass
            ks_s = ks_p = None
            try:
                norm_v = (v-mu)/max(sd,1e-9)
                ks_s, ks_p = sp_stats.kstest(norm_v, "norm")
                ks_s = float(ks_s); ks_p = float(ks_p)
            except Exception: pass
        return {"n":n,"mean":mu,"median":med,"mode":mo,"std":sd,"skew":sk,"kurt":ku,
                "min":mn,"max":mx,"q1":q1,"q3":q3,"sw_s":sw_s,"sw_p":sw_p,
                "ks_s":ks_s,"ks_p":ks_p}

    def _build_canvas(self):
        self.canvas = []
        if not _NUM: self.canvas=["  ✗ numpy/pandas/scipy no disponibles."]; return
        if not self._load(): self.canvas=[f"  ⚠ {self.errmsg}"]; return
        if not self.num_cols: self.canvas=["  Sin columnas numéricas en este CSV."]; return
        c, v = self._col()
        if v is None or len(v)<2: self.canvas=[f"  '{c}' sin datos suficientes."]; return
        s = self._stats(v)
        sub = self.sub

        if sub == 0:   self._canvas_summary(c, v, s)
        elif sub == 1: self._canvas_dist(c, v, s)
        elif sub == 2: self._canvas_tests(c, v, s)
        elif sub == 3: self._canvas_homog()

    def _canvas_summary(self, c, v, s):
        L = self.canvas.append
        L(f"  ══ RESUMEN ESTADÍSTICO: {c} ══")
        L(f"  n = {s['n']}   (columna numérica del CSV)")
        L("")
        L(f"  {'Medida':<28} {'Valor':>14}")
        L("  " + "─"*44)
        for lbl, key, fmt in [
            ("Media (μ)",          "mean",   ".6f"),
            ("Mediana",            "median", ".6f"),
            ("Moda",               "mode",   ".6f"),
            ("Desv. estándar (σ)", "std",    ".6f"),
            ("Mínimo",             "min",    ".6f"),
            ("Máximo",             "max",    ".6f"),
            ("Q1 (25%)",           "q1",     ".6f"),
            ("Q3 (75%)",           "q3",     ".6f"),
            ("IQR (Q3-Q1)",        None,     ".6f"),
            ("Asimetría (skew)",   "skew",   ".4f"),
            ("Curtosis (excess)",  "kurt",   ".4f"),
        ]:
            v2 = s["q3"]-s["q1"] if key is None else s.get(key,0)
            v2_str = format(v2, fmt)
            L(f"  {lbl:<28} {v2_str:>14}")
        L("")
        L("  ── INTERPRETACIÓN ──")
        sk = s["skew"]; ku = s["kurt"]
        sym_s = "→ Simétrica" if abs(sk)<0.5 else ("→ Sesgada derecha (cola larga)" if sk>0 else "→ Sesgada izquierda")
        kur_s = "→ Normal (mesocúrtica)" if abs(ku)<0.5 else ("→ Leptocúrtica (colas pesadas)" if ku>0 else "→ Platicúrtica (colas ligeras)")
        L(f"  Asimetría  sk={sk:.3f}  {sym_s}")
        L(f"  Curtosis   ku={ku:.3f}  {kur_s}")
        L("")
        L("  ── NORMALIDAD ──")
        if s.get("sw_p") is not None:
            sw_res = "✓ Normal (p>0.05)" if s["sw_p"]>0.05 else "✗ No normal (p≤0.05)"
            L(f"  Shapiro-Wilk  W={s['sw_s']:.4f}  p={s['sw_p']:.4f}  {sw_res}")
        if s.get("ks_p") is not None:
            ks_res = "✓ Normal (p>0.05)" if s["ks_p"]>0.05 else "✗ No normal (p≤0.05)"
            L(f"  Kolmogorov-Smirnov  D={s['ks_s']:.4f}  p={s['ks_p']:.4f}  {ks_res}")

    def _canvas_dist(self, c, v, s, B=30, H=12):
        """Histograma ASCII + curva normal superpuesta."""
        L = self.canvas.append
        mn=s["min"]; mx=s["max"]; mu=s["mean"]; sd=s["std"]
        rng=mx-mn if mx!=mn else 1e-9
        cts=[0]*B
        for x in v: cts[min(int((x-mn)/rng*B),B-1)]+=1
        mc=max(cts) if max(cts)>0 else 1
        L(f"  ══ DISTRIBUCIÓN: {c} ══")
        L(f"  n={s['n']}  μ={mu:.4f}  σ={sd:.4f}  [{mn:.4f}, {mx:.4f}]")
        L("")
        for ri in range(H,0,-1):
            thr=ri/H*mc
            row="  │"+"".join("█" if c2>=thr else " " for c2 in cts)+"│"
            L(row)
        L("  └"+"─"*B+"┘")
        L(f"  {mn:.3f}"+" "*(B-14)+f"{mx:.3f}")
        L("")
        # Estado normalidad
        if s.get("sw_p") is not None:
            normal = s["sw_p"] > 0.05
            L(f"  Normalidad (Shapiro-Wilk p={s['sw_p']:.4f}): "
              f"{'✓ Distribución NORMAL' if normal else '✗ Distribución NO NORMAL'}")
            L(f"  → Usar: {'Pruebas PARAMÉTRICAS (t-test, ANOVA, Pearson)' if normal else 'Pruebas NO PARAMÉTRICAS (Mann-Whitney, Kruskal-Wallis, Spearman)'}")
        L("")
        L("  ── PERCENTILES ──")
        for pct in [5,10,25,50,75,90,95]:
            pv = float(np.percentile(v,pct))
            bar = "█"*int(pct/5)
            L(f"  P{pct:>2}  {pv:>10.4f}  {bar}")

    def _canvas_tests(self, c, v, s):
        """Tabla de pruebas paramétricas y no paramétricas."""
        L = self.canvas.append
        sw_p = s.get("sw_p") or 0.0
        sw_s = s.get("sw_s") or 0.0
        ks_p = s.get("ks_p") or 0.0
        ks_s = s.get("ks_s") or 0.0
        normal = s["sw_p"] > 0.05 if s.get("sw_p") is not None else None
        L(f"  ══ PRUEBAS ESTADÍSTICAS RECOMENDADAS: {c} ══")
        L(f"  Normalidad: {'✓ Normal' if normal else ('✗ No normal' if normal is not None else '? No determinada')}  "
          f"(Shapiro-Wilk W={sw_s:.4f} p={sw_p:.4f})")
        L("")
        # Pruebas paramétricas
        L("  ── PRUEBAS PARAMÉTRICAS  (requieren normalidad) ──────────────────────")
        pruebas_p = [
            ("t de Student",  "2 grupos independientes",   "scipy.stats.ttest_ind(a,b)"),
            ("t pareada",     "antes/después mismo grupo", "scipy.stats.ttest_rel(a,b)"),
            ("ANOVA",         "3+ grupos independientes",  "scipy.stats.f_oneway(*grupos)"),
            ("MANOVA",        "multivariado de varianza",  "statsmodels MANOVA"),
            ("Pearson r",     "correlación lineal",        "scipy.stats.pearsonr(x,y)"),
            ("Levene",        "homogeneidad de varianzas", "scipy.stats.levene(*grupos)"),
        ]
        for nm, uso, fn in pruebas_p:
            mark = "✓ APLICA" if normal else "✗ no aplica"
            at_s = f"  {'●' if normal else '○'}  {nm:<18} {uso:<34} → {mark}"
            L(at_s)
        L("")
        # Pruebas no paramétricas
        L("  ── PRUEBAS NO PARAMÉTRICAS  (no requieren normalidad) ────────────────")
        pruebas_np = [
            ("Mann-Whitney U", "2 grupos independientes",   "scipy.stats.mannwhitneyu(a,b)"),
            ("Wilcoxon",       "antes/después mismo grupo", "scipy.stats.wilcoxon(d)"),
            ("Kruskal-Wallis", "3+ grupos independientes",  "scipy.stats.kruskal(*grupos)"),
            ("Friedman",       "medidas repetidas ANOVA",   "scipy.stats.friedmanchisquare()"),
            ("Spearman ρ",     "correlación no lineal",     "scipy.stats.spearmanr(x,y)"),
            ("Shapiro-Wilk",   "verificar normalidad",      f"p={sw_p:.4f}"),
        ]
        for nm, uso, fn in pruebas_np:
            mark = "✓ APLICA" if not normal else "○ (normal → usar paramétrica)"
            at_s = f"  {'●' if not normal else '○'}  {nm:<18} {uso:<34} → {mark}"
            L(at_s)
        L("")
        # Pruebas de supuestos
        L("  ── PRUEBAS DE SUPUESTOS ───────────────────────────────────────────────")
        suposiciones = [
            ("Shapiro-Wilk",   "normalidad de datos",         f"p={sw_p:.4f}"),
            ("K-S test",       "normalidad (n>50)",            f"p={ks_p:.4f}"),
            ("Levene",         "homogeneidad de varianzas",    "comparar entre modelos"),
            ("Mauchly",        "esfericidad medidas repetidas","statsmodels"),
            ("Post-hoc Tukey", "comparaciones múltiples",     "statsmodels pairwise_tukeyhsd"),
            ("Dunn / Nemenyi", "post-hoc no paramétrico",     "scikit-posthocs"),
        ]
        for nm, uso, fn in suposiciones:
            L(f"  ►  {nm:<18} {uso:<34}  {fn}")

    def _canvas_homog(self):
        """Test de homogeneidad (Levene) entre fases."""
        from scipy import stats as sp_stats
        L = self.canvas.append
        L("  ══ HOMOGENEIDAD ENTRE FASES (Levene) ══")
        L("  H₀: varianzas iguales entre fases  p>0.05 → homogéneas")
        L("")
        # Agrupar columnas numéricas por fase
        phase_data = {}
        for col in self.num_cols:
            ph = RepositoryScreen(None,{})._phase_of(col) if False else "?"
            # Usar la lógica de fase directamente
            ph = "Otras"
            for phname, prefixes in RepositoryScreen.PHASE_PREFIXES:
                for p in prefixes:
                    if col.startswith(p) or col==p:
                        ph = phname; break
                if ph != "Otras": break
            v2 = pd.to_numeric(self.df[col],errors="coerce").dropna().values
            if len(v2) >= 3:
                phase_data.setdefault(ph,[]).extend(list(v2))

        groups = [(ph, np.array(vals)) for ph,vals in phase_data.items() if len(vals)>=3]
        if len(groups) < 2:
            L("  ⚠ Se necesitan al menos 2 fases con datos numéricos.")
            return
        L(f"  {'Fase A':<18} {'Fase B':<18} {'Stat':>8} {'p-valor':>10} {'Resultado':<30}")
        L("  " + "─"*80)
        from itertools import combinations
        for (phA,vA),(phB,vB) in combinations(groups, 2):
            try:
                stat,pv = sp_stats.levene(vA,vB)
                res = "✓ Homogéneas (p>0.05)" if pv>0.05 else "✗ Heterogéneas (p≤0.05)"
                L(f"  {phA:<18} {phB:<18} {stat:>8.3f} {pv:>10.4f} {res}")
            except Exception as e:
                L(f"  {phA:<18} {phB:<18} Error: {e}")
        L("")
        L("  ── INTERPRETACIÓN ──")
        L("  p > 0.05 → varianzas homogéneas → usar ANOVA / t-test")
        L("  p ≤ 0.05 → varianzas heterogéneas → usar Welch-ANOVA / Mann-Whitney")

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw(self):
        self.scr.erase(); self._sz()
        subtabs = ["[A]Resumen","[B]Distribución","[C]Pruebas","[D]Homogeneidad"]
        self.title(2, "  ".join(subtabs))
        self.statusbar()

        if self.mode=="pick_csv":
            self._dpick(); self.navbar(); self.scr.refresh(); return

        rs = 3
        if not self.dfp: self._load()
        if not self.dfp:
            sa(self.scr,rs+2,4,f"⚠ {self.errmsg}",curses.color_pair(cER))
            hint_bar(self.scr,self.h-3," [f]Elegir CSV ")
            self.navbar(); self.scr.refresh(); return

        # Panel lateral: lista de columnas numéricas
        PW = 26
        hl(self.scr,rs,self.w-PW-2,PW+1)
        sa(self.scr,rs,self.w-PW-1," COLUMNAS NUM ",curses.color_pair(cH)|curses.A_BOLD)
        vis_p = self.h-rs-4
        # sub-tabs display
        for i,t in enumerate(subtabs):
            sa(self.scr,rs,2+i*16,f" {t} ",
               curses.color_pair(cS)|curses.A_BOLD if i==self.sub else dim())

        nc = self.num_cols
        if self.col_idx>=self.col_scroll+vis_p-1: self.col_scroll=self.col_idx-vis_p+2
        if self.col_idx<self.col_scroll: self.col_scroll=self.col_idx
        self.col_scroll=max(0,self.col_scroll)
        for i,col in enumerate(nc):
            sr=rs+1+(i-self.col_scroll)
            if not (rs+1<=sr<=self.h-5): continue
            sel=i==self.col_idx
            at=curses.color_pair(cS)|curses.A_BOLD if sel else dim()
            sa(self.scr,sr,self.w-PW-1,f" {tr(col,PW-2)} ",at)

        # Área principal
        DW = self.w-PW-3
        hl(self.scr,rs+1,0,DW)
        if not self.canvas: self._build_canvas()
        ms=max(0,len(self.canvas)-(self.h-rs-5))
        self.scroll=max(0,min(self.scroll,ms))
        for i in range(self.h-rs-6):
            idx=self.scroll+i
            if idx>=len(self.canvas): break
            ln=self.canvas[idx]
            if   "✓" in ln or "NORMAL" in ln: at=curses.color_pair(cOK)
            elif "✗" in ln or "NO NORMAL" in ln or "✗" in ln: at=curses.color_pair(cER)
            elif "══" in ln or "──" in ln: at=curses.color_pair(cP)|curses.A_BOLD
            elif "●" in ln: at=curses.color_pair(cMX)
            elif "○" in ln: at=curses.color_pair(cDM)
            elif "►" in ln: at=curses.color_pair(cMT)
            elif "│" in ln and "█" in ln: at=curses.color_pair(cGR)
            else: at=curses.color_pair(cN)
            sa(self.scr,rs+2+i,1,tr(ln,DW-2),at)

        if self.notif and self.ntimer>0:
            sa(self.scr,self.h-3,0,f" {self.notif} "[:self.w-1],curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer-=1
        else:
            hint_bar(self.scr,self.h-3,
                     " [A/B/C/D]Sub-tab  [↑↓]Col-lista  [j/k]Scroll  [f]CSV  [R]Recargar ")
        self.navbar(); self.scr.refresh()

    def _dpick(self):
        rs=5; b=curses.color_pair(cBR)
        found=_find_csvs(self.ibuf)
        sa(self.scr,rs,4,"╔"+"═"*68+"╗",b)
        sa(self.scr,rs+1,4,"║  CSV DE ENTRADA (results_YYYYMMDD / res_F*_YYYYMMDD.csv)"+" "*11+"║",b)
        sa(self.scr,rs+2,4,"╠"+"═"*68+"╣",b)
        sa(self.scr,rs+3,6,f"Ruta: {self.ibuf}█",curses.color_pair(cP)|curses.A_BOLD)
        ok=Path(self.ibuf).exists() if self.ibuf else False
        sa(self.scr,rs+4,6,
           ("✓ Encontrado" if ok else "✗ No encontrado") if self.ibuf else "(escribe o elige de la lista)",
           curses.color_pair(cOK) if ok else curses.color_pair(cER if self.ibuf else cDM))
        if found:
            sa(self.scr,rs+5,6,f"CSVs ({len(found)}):",dim())
            for i,p in enumerate(found):
                sa(self.scr,rs+6+i,8,tr(f"{i+1}. {p}",self.w-12),curses.color_pair(cMT))
        bot=rs+6+len(found)+2
        sa(self.scr,bot,4,"║  [Enter]Aplicar  [Esc]Cancelar  [1-9]Elegir lista" +" "*19+"║",dim())
        sa(self.scr,bot+1,4,"╚"+"═"*68+"╝",b)
        hint_bar(self.scr,self.h-3," [1-9]Elegir  [Enter]Aplicar  [Esc]Cancelar ")

    def key(self, k):
        if self.mode=="pick_csv":
            found=_find_csvs(self.ibuf)
            if ord('1')<=k<=ord('9'):
                idx=k-ord('1')
                if idx<len(found): self.ibuf=str(found[idx].resolve())
            elif k in (ord('\n'),ord('\r'),10,13):
                self.csv_path=self.ibuf.strip(); self.st["input_csv"]=self.csv_path
                self.dfp=""; self.num_cols=[]; self.canvas=[]
                self.notif=f"✓ CSV: {tr(self.ibuf,50)}"; self.ntimer=60
                self.mode="normal"; self._load()
            elif k==27: self.mode="normal"; self.ibuf=""
            elif k in (curses.KEY_BACKSPACE,127,8): self.ibuf=self.ibuf[:-1]
            elif 32<=k<=126: self.ibuf+=chr(k)
            return None
        # Sub-tabs
        if   k in (ord('a'),ord('A')): self.sub=0; self.canvas=[]; self.scroll=0
        elif k in (ord('b'),ord('B')): self.sub=1; self.canvas=[]; self.scroll=0
        elif k in (ord('c'),ord('C')): self.sub=2; self.canvas=[]; self.scroll=0
        elif k in (ord('d'),ord('D')): self.sub=3; self.canvas=[]; self.scroll=0
        elif k in (ord('f'),ord('F')): self.ibuf=self.csv_path; self.mode="pick_csv"
        elif k in (ord('r'),ord('R')): self.dfp=""; self.canvas=[]; self.scroll=0
        # Navegación columnas (panel lateral)
        elif k in (curses.KEY_UP,):
            self.col_idx=max(0,self.col_idx-1); self.canvas=[]; self.scroll=0
        elif k in (curses.KEY_DOWN,):
            self.col_idx=min(max(0,len(self.num_cols)-1),self.col_idx+1); self.canvas=[]; self.scroll=0
        # Scroll canvas (j/k)
        elif k==ord('k'): self.scroll=max(0,self.scroll-1)
        elif k==ord('j'): self.scroll+=1
        elif k==ord('\t'): return "next"
        elif k in (ord('q'),27): return "prev"
        return None



# ══════════════════════════════════════════════════════════════════════════════
# § 13  PANTALLA 3 — LISTAS  (ex-Repositorio + sub: Gráficos CSV + Estadística CSV)
#        Sub-tabs: [L]Lista-columnas  [G]Gráficos-CSV  [E]Estadística-CSV
# ══════════════════════════════════════════════════════════════════════════════
class ListasScreen(Screen):
    """
    Wrapper que integra en una sola pestaña [3] Listas:
      sub=0  [L]ista columnas  → RepositoryScreen  (evalúa CSV sin ejecución)
      sub=1  [G]ráficos CSV    → GraphCSVScreen     (gráficos sobre CSV directo)
      sub=2  [E]stadística CSV → StatScreen          (estadística sobre CSV)
    Ningún sub-tab requiere haber ejecutado un run.
    """
    SUBTABS = ["[L]Lista-Col", "[G]Gráficos-CSV", "[E]Estadística-CSV"]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub     = 0
        self._repo   = RepositoryScreen(scr, st)
        self._stat   = StatScreen(scr, st)
        # GraphCSV: usa GraphScreen pero forzando carga desde csv_path en lugar de run
        self._gcsv   = _GraphCSVScreen(scr, st)

    def _active(self):
        return [self._repo, self._gcsv, self._stat][self.sub]

    def draw(self):
        self.scr.erase(); self._sz()
        hint = "  ".join(
            f" {t} " if i == self.sub else t
            for i, t in enumerate(self.SUBTABS)
        )
        self.title(2, hint)
        # Delegar completamente al sub-screen activo
        act = self._active()
        act.h = self.h; act.w = self.w
        # Redibujar sin erase (title ya está puesto)
        if self.sub == 0:
            self._repo._draw_body()
        elif self.sub == 1:
            self._gcsv._draw_body()
        else:
            self._stat._draw_body()
        self.navbar(f"  [L/G/E]Sub-tab")

    def key(self, k):
        # Cambio de sub-tab
        if   k in (ord('l'), ord('L')): self.sub = 0; return None
        elif k in (ord('g'), ord('G')): self.sub = 1; return None
        elif k in (ord('e'), ord('E')) and self.sub != 0: self.sub = 2; return None
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        # Delegar al sub-screen
        res = self._active().key(k)
        if res in ("next", "prev", "quit"): return res
        return None


class _GraphCSVScreen(Screen):
    """
    Gráficos estadísticos sobre el CSV directo (sin ejecución de run).
    Gráficos disponibles:
      1. Histograma         — distribución de la columna seleccionada
      2. Boxplot ASCII      — Q1/mediana/Q3/bigotes
      3. Serie temporal     — valores ordenados (para detectar tendencias)
      4. Top-N valores      — barra horizontal de los N mayores/menores
      5. Correlación col.   — correlación de la columna vs todas las demás numéricas
      6. Densidad empírica  — histograma acumulado normalizado (CDF)
      7. Scatter dos cols   — dispersión entre col seleccionada y la siguiente
      8. Resumen por grupo  — media por valor de columna categórica
    """
    GRAPHS_CSV = [
        ("Histograma",           "hist"),
        ("Boxplot ASCII",        "box"),
        ("Serie / Tendencia",    "serie"),
        ("Top-N valores",        "topn"),
        ("Correlación columnas", "corr"),
        ("CDF empírica",         "cdf"),
        ("Scatter dos columnas", "scatter2"),
        ("Resumen por grupo",    "bygroup"),
    ]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.gi         = 0
        self.df         = None; self.dfp = ""
        self.canvas     = []; self.scroll = 0
        self.errmsg     = ""
        self.num_cols   = []
        self.col_idx    = 0
        self.col_scroll = 0
        self.cat_cols   = []
        self.mode4      = True             # True=grilla 2×2, False=1 gráfico
        self.q_canvases = [[], [], [], []] # canvas por cuadrante en modo4

    def _load(self):
        cp = self.st.get("input_csv", "")
        if not cp or not Path(cp).exists():
            self.errmsg = f"CSV no encontrado: {cp or '(ninguno) — elige CSV en [L]Lista-Col'}"; return False
        if self.dfp == cp: return True
        try:
            self.df = pd.read_csv(cp, nrows=5000); self.dfp = cp
            all_num = [c for c in self.df.columns
                       if pd.api.types.is_numeric_dtype(self.df[c])
                       and str(self.df[c].dtype) != "bool"]
            # Ordenar columnas según fases presentes en el nombre del CSV
            # res_F1_... → det primero; res_F1F2_... → det,cls; etc.
            csv_name = Path(cp).stem.upper()  # ej: RES_F1F2F3F4_20260314
            PHASE_ORDER = [
                ("det_",  "F1" in csv_name),
                ("cls_",  "F2" in csv_name),
                ("vlm_",  "F3" in csv_name),
                ("llm_",  "F4" in csv_name),
            ]
            ordered = []
            for prefix, active in PHASE_ORDER:
                if active:
                    ordered.extend([c for c in all_num if c.startswith(prefix)
                                    and c not in ordered])
            # Resto de columnas no cubiertas (total_, system_, etc.)
            rest = [c for c in all_num if c not in ordered]
            self.num_cols = ordered + rest
            self.cat_cols = [c for c in self.df.columns
                             if str(self.df[c].dtype) in ("object","category")]
            self.col_idx = 0; self.col_scroll = 0
            self.canvas = []; self.errmsg = ""; return True
        except Exception as e:
            self.errmsg = f"Error CSV: {e}"; return False

    def _col(self):
        if not self.num_cols: return None, None
        c = self.num_cols[self.col_idx % len(self.num_cols)]
        v = pd.to_numeric(self.df[c], errors="coerce").dropna().values.astype(float)
        return c, v

    def _next_col(self):
        if len(self.num_cols) < 2: return None, None
        c = self.num_cols[(self.col_idx + 1) % len(self.num_cols)]
        v = pd.to_numeric(self.df[c], errors="coerce").dropna().values.astype(float)
        return c, v

    # ── generadores de gráficos ───────────────────────────────────────────────
    def _build(self):
        self.canvas = []
        if not _NUM: self.canvas = ["  ✗ numpy/pandas no disponibles."]; return
        if not self._load():
            self.canvas = ["", f"  ⚠  {self.errmsg}", "",
                           "  Para ver gráficos CSV:",
                           "  1. Ve a [L]Lista-Col y selecciona un CSV con [f]",
                           "  2. Regresa aquí con [G]"]; return
        if not self.num_cols: self.canvas = ["  Sin columnas numéricas en el CSV."]; return
        c, v = self._col()
        if v is None or len(v) < 2: self.canvas = [f"  '{c}' sin datos suficientes."]; return
        k = self.GRAPHS_CSV[self.gi][1]
        try:
            if   k == "hist":    self._g_hist(c, v)
            elif k == "box":     self._g_box(c, v)
            elif k == "serie":   self._g_serie(c, v)
            elif k == "topn":    self._g_topn(c, v)
            elif k == "corr":    self._g_corr(c, v)
            elif k == "cdf":     self._g_cdf(c, v)
            elif k == "scatter2":self._g_scatter2(c, v)
            elif k == "bygroup": self._g_bygroup(c, v)
        except Exception as e:
            self.canvas = [f"  Error generando gráfico: {e}"]

    def _g_hist(self, c, v, B=40, H=14):
        L = self.canvas.append
        mn=v.min(); mx=v.max(); mu=float(np.mean(v)); sd=float(np.std(v,ddof=1))
        rng = mx-mn if mx != mn else 1e-9
        cts = [0]*B
        for x in v: cts[min(int((x-mn)/rng*B), B-1)] += 1
        mc = max(cts) if max(cts) > 0 else 1
        L(f"  ══ HISTOGRAMA: {c} ══")
        L(f"  n={len(v)}  μ={mu:.4f}  σ={sd:.4f}  [{mn:.4f}, {mx:.4f}]")
        L("")
        for ri in range(H, 0, -1):
            thr = ri/H*mc
            freq = "".join("█" if ct >= thr else " " for ct in cts)
            L(f"  {ri*mc//H:>4} │{freq}│")
        L(f"       └{'─'*B}┘")
        step = rng/4
        L(f"       {mn:<10.3f}{mn+step:<10.3f}{mn+2*step:<10.3f}{mn+3*step:<10.3f}{mx:.3f}")
        L("")
        L(f"  Moda aproximada: bin [{mn+cts.index(max(cts))*rng/B:.3f}, {mn+(cts.index(max(cts))+1)*rng/B:.3f}]")
        # Shapiro-Wilk
        if _NUM and len(v) >= 3:
            try:
                from scipy import stats as sp
                _,sw_p = sp.shapiro(v[:5000])
                L(f"  Shapiro-Wilk p={sw_p:.4f} → {'✓ Normal' if sw_p>0.05 else '✗ No normal'}")
            except Exception: pass

    def _g_box(self, c, v):
        L = self.canvas.append
        mn=float(v.min()); mx=float(v.max())
        q1=float(np.percentile(v,25)); med=float(np.median(v)); q3=float(np.percentile(v,75))
        iqr=q3-q1; lo=max(mn, q1-1.5*iqr); hi=min(mx, q3+1.5*iqr)
        W=60; rng=mx-mn if mx!=mn else 1e-9
        def pos(x): return int((x-mn)/rng*(W-1))
        L(f"  ══ BOXPLOT: {c} ══")
        L(f"  n={len(v)}  Q1={q1:.4f}  Med={med:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
        L("")
        row = [" "]*W
        # bigotes
        for i in range(pos(lo), pos(q1)):  row[i]="─"
        for i in range(pos(q3), pos(hi)+1):row[i]="─"
        # caja
        for i in range(pos(q1), pos(q3)+1):row[i]="░"
        row[pos(med)]="█"
        row[pos(lo)]="├"; row[pos(hi)]="┤"
        row[pos(q1)]="["; row[pos(q3)]="]"
        L("  " + "".join(row))
        L(f"  {mn:<12.4f}{q1:<12.4f}{med:<12.4f}{q3:<12.4f}{mx:.4f}")
        L(f"  mín         Q1          Mediana     Q3          máx")
        L("")
        # outliers
        outs = v[(v < lo) | (v > hi)]
        L(f"  Outliers: {len(outs)}  ({100*len(outs)/len(v):.1f}%)")
        if len(outs) > 0 and len(outs) <= 10:
            L(f"  Valores: {[round(x,4) for x in sorted(outs)]}")

    def _g_serie(self, c, v, H=12, W=70):
        L = self.canvas.append
        mn=float(v.min()); mx=float(v.max()); rng=mx-mn if mx!=mn else 1e-9
        # Submuestrear si es muy largo
        if len(v) > W:
            idx = np.linspace(0, len(v)-1, W, dtype=int)
            vs = v[idx]
        else:
            vs = v
        n = len(vs)
        L(f"  ══ SERIE / TENDENCIA: {c} ══")
        L(f"  n={len(v)}  [{mn:.4f}, {mx:.4f}]  (mostrando {n} puntos)")
        L("")
        grid = [[" "]*n for _ in range(H)]
        for xi, val in enumerate(vs):
            yi = int((val-mn)/rng*(H-1))
            yi = max(0, min(H-1, yi))
            grid[H-1-yi][xi] = "▪"
        L(f"  {mx:.4f} │")
        for row in grid:
            L("         │" + "".join(row))
        L(f"  {mn:.4f} └" + "─"*n)
        L(f"          {'inicio':<{n//2}}{'fin':>{n//2}}")
        # tendencia lineal
        if _NUM:
            try:
                from scipy import stats as sp
                xs = np.arange(len(v), dtype=float)
                slope,intercept,r,p,_ = sp.linregress(xs, v)
                dir_s = "↑ creciente" if slope>0 else "↓ decreciente"
                L(f"  Tendencia: slope={slope:.6f}  r²={r**2:.4f}  p={p:.4f}  {dir_s if abs(r)>0.1 else '→ sin tendencia clara'}")
            except Exception: pass

    def _g_topn(self, c, v, N=20):
        L = self.canvas.append
        L(f"  ══ TOP-{N} VALORES: {c} ══")
        L(f"  n={len(v)}")
        L("")
        sorted_v = np.sort(v)[::-1][:N]
        mx = sorted_v[0] if sorted_v[0] != 0 else 1
        BW = 40
        L(f"  {'Rk':>3}  {'Valor':>12}  {'Barra'}")
        L("  " + "─"*60)
        for i, val in enumerate(sorted_v):
            bar = "█"*max(1,int(val/mx*BW)) if mx>0 else ""
            L(f"  {i+1:>3}  {val:>12.6f}  {bar}")
        L("")
        L(f"  Media top-{N}: {float(np.mean(sorted_v)):.6f}")
        L(f"  Media total:  {float(np.mean(v)):.6f}")
        L(f"  Ratio:        {float(np.mean(sorted_v)/max(np.mean(v),1e-9)):.3f}x")

    def _g_corr(self, c, v):
        L = self.canvas.append
        L(f"  ══ CORRELACIÓN: {c} vs otras columnas ══")
        L(f"  n={len(v)}")
        L("")
        if not _NUM:
            L("  numpy no disponible"); return
        corrs = []
        for c2 in self.num_cols:
            if c2 == c: continue
            v2 = pd.to_numeric(self.df[c2], errors="coerce").dropna().values.astype(float)
            try:
                common = min(len(v), len(v2))
                if common < 3: continue
                # Omitir columnas constantes (spearmanr lanza ConstantInputWarning)
                if v[:common].std() == 0 or v2[:common].std() == 0: continue
                from scipy import stats as sp
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = sp.spearmanr(v[:common], v2[:common])
                if r != r: continue  # NaN check
                corrs.append((float(r), float(p), c2))
            except Exception: pass
        corrs.sort(key=lambda x: abs(x[0]), reverse=True)
        L(f"  {'Columna':<28} {'Spearman ρ':>12} {'p-valor':>10} {'Fuerza'}")
        L("  " + "─"*65)
        BW = 20
        for r,p,c2 in corrs[:20]:
            bar = "█"*max(1,int(abs(r)*BW))
            sign = "+" if r>=0 else "-"
            mag = "fuerte" if abs(r)>0.7 else ("moderada" if abs(r)>0.4 else "débil")
            sig = "*" if p<0.05 else " "
            L(f"  {tr(c2,27):<28} {r:>12.4f} {p:>10.4f} {sign}{bar} {mag}{sig}")
        if not corrs:
            L("  No se encontraron otras columnas numéricas para correlacionar.")

    def _g_cdf(self, c, v, W=60):
        L = self.canvas.append
        mn=float(v.min()); mx=float(v.max()); H=12
        sv = np.sort(v)
        L(f"  ══ CDF EMPÍRICA: {c} ══")
        L(f"  n={len(v)}  [{mn:.4f}, {mx:.4f}]")
        L("")
        # Submuestrear a W puntos
        idx = np.linspace(0, len(sv)-1, W, dtype=int)
        xs  = sv[idx]
        ps  = (idx+1)/len(sv)
        grid = [[" "]*W for _ in range(H)]
        for xi, (x,p) in enumerate(zip(xs,ps)):
            yi = int(p*(H-1)); yi = max(0,min(H-1,yi))
            grid[H-1-yi][xi]="▪"
        L("  1.00 │")
        for row in grid:
            L("       │" + "".join(row))
        L("  0.00 └" + "─"*W)
        L(f"       {mn:.4f}" + " "*(W-14) + f"{mx:.4f}")
        L("")
        for pct in [25,50,75,90,95]:
            pv = float(np.percentile(v, pct))
            L(f"  P{pct:>2} = {pv:.6f}")

    def _g_scatter2(self, c, v):
        L = self.canvas.append
        c2, v2 = self._next_col()
        if c2 is None: L("  Solo hay una columna numérica."); return
        common = min(len(v), len(v2))
        vx = v[:common]; vy = v2[:common]
        W=50; H=12
        mnx=vx.min(); mxx=vx.max(); mny=vy.min(); mxy=vy.max()
        rngx=mxx-mnx if mxx!=mnx else 1e-9; rngy=mxy-mny if mxy!=mny else 1e-9
        grid = [[" "]*W for _ in range(H)]
        for xi,yi2 in zip(vx,vy):
            gx=int((xi-mnx)/rngx*(W-1)); gy=int((yi2-mny)/rngy*(H-1))
            gx=max(0,min(W-1,gx)); gy=max(0,min(H-1,gy))
            grid[H-1-gy][gx]="▪"
        L(f"  ══ SCATTER: {tr(c,20)} vs {tr(c2,20)} ══")
        L(f"  n={common}")
        L("")
        L(f"  {tr(c2,8):<8} │")
        for row in grid:
            L("          │"+"".join(row))
        L(f"          └{'─'*W}")
        L(f"           {tr(c,W)}")
        if _NUM:
            try:
                from scipy import stats as sp
                r,p = sp.spearmanr(vx,vy)
                L(f"  Spearman ρ={r:.4f}  p={p:.4f}  {'correlación significativa' if p<0.05 else 'no significativa'}")
            except Exception: pass

    def _g_bygroup(self, c, v):
        L = self.canvas.append
        L(f"  ══ RESUMEN POR GRUPO: {c} ══")
        if not self.cat_cols:
            L("  Sin columnas categóricas en el CSV."); return
        gcol = self.cat_cols[0]
        L(f"  Agrupando por: {gcol}  (columna categórica)")
        L("")
        groups = {}
        for g in self.df[gcol].dropna().unique():
            vals = pd.to_numeric(self.df.loc[self.df[gcol]==g, c], errors="coerce").dropna().values.astype(float)
            if len(vals) >= 1:
                groups[str(g)] = vals
        if not groups: L("  Sin grupos con datos."); return
        max_mu = max(float(np.mean(v2)) for v2 in groups.values()); BW=30
        L(f"  {'Grupo':<20} {'n':>5} {'Media':>10} {'Std':>8} {'Barra'}")
        L("  "+"─"*70)
        for g, vals in sorted(groups.items(), key=lambda x: np.mean(x[1]), reverse=True):
            mu2=float(np.mean(vals)); sd2=float(np.std(vals,ddof=1)) if len(vals)>1 else 0
            bar="█"*max(1,int(mu2/max(max_mu,1e-9)*BW))
            L(f"  {tr(g,19):<20} {len(vals):>5} {mu2:>10.4f} {sd2:>8.4f} {bar}")

    def _build_quad(self):
        """Construye los 4 canvas de la grilla 2×2 para la columna seleccionada."""
        self.q_canvases = [[], [], [], []]
        if not self._load() or not self.num_cols: return
        c, v = self._col()
        if v is None or len(v) < 2: return
        # Cuadrante 0: Histograma  (arriba-izq)
        tmp = self.canvas; self.canvas = self.q_canvases[0]
        self._g_hist(c, v, B=22, H=6)
        self.q_canvases[0] = self.canvas
        # Cuadrante 1: Boxplot     (arriba-der)
        self.canvas = self.q_canvases[1]
        self._g_box_compact(c, v)
        self.q_canvases[1] = self.canvas
        # Cuadrante 2: Serie       (abajo-izq)
        self.canvas = self.q_canvases[2]
        self._g_serie(c, v, H=6, W=38)
        self.q_canvases[2] = self.canvas
        # Cuadrante 3: CDF         (abajo-der)
        self.canvas = self.q_canvases[3]
        self._g_cdf(c, v, W=38)
        self.q_canvases[3] = self.canvas
        self.canvas = tmp

    def _g_box_compact(self, c, v):
        """Boxplot compacto para grilla 2×2."""
        L = self.canvas.append
        mn=float(v.min()); mx=float(v.max())
        q1=float(np.percentile(v,25)); med=float(np.median(v)); q3=float(np.percentile(v,75))
        iqr=q3-q1; lo=max(mn,q1-1.5*iqr); hi=min(mx,q3+1.5*iqr)
        W=38; rng=mx-mn if mx!=mn else 1e-9
        def pos(x): return int((x-mn)/rng*(W-1))
        L(f"  ══ BOXPLOT: {tr(c,20)} ══")
        L(f"  n={len(v)}  Q1={q1:.3f}  Med={med:.3f}  Q3={q3:.3f}")
        L("")
        row=[" "]*W
        for i in range(pos(lo),pos(q1)):    row[i]="─"
        for i in range(pos(q3),min(pos(hi)+1,W)): row[i]="─"
        for i in range(pos(q1),min(pos(q3)+1,W)): row[i]="░"
        row[min(pos(med),W-1)]="█"
        for p,ch in [(pos(lo),"├"),(pos(hi),"┤"),(pos(q1),"["),(pos(q3),"]")]:
            if 0<=p<W: row[p]=ch
        L("  "+"".join(row))
        L(f"  {mn:.3f}" + " "*(W//2-8) + f"med={med:.3f}" + " "*(W//2-8) + f"{mx:.3f}")
        L("")
        outs=v[(v<lo)|(v>hi)]
        L(f"  IQR={iqr:.4f}  Outliers={len(outs)}")

    # ── draw body (llamado desde ListasScreen) ────────────────────────────────
    def _draw_body(self):
        rs = 3; PW = 26
        self.statusbar()

        # ── Panel derecho: columnas + modo indicator ──────────────────────────
        hl(self.scr, rs, self.w-PW-2, PW+1)
        if self.mode4:
            sa(self.scr, rs, self.w-PW-1, " 4 GRÁFICOS [M] ", curses.color_pair(cH)|curses.A_BOLD)
        else:
            sa(self.scr, rs, self.w-PW-1, " GRÁFICOS CSV   ", curses.color_pair(cH)|curses.A_BOLD)
            for i, (g, _) in enumerate(self.GRAPHS_CSV):
                rw = rs+1+i
                if rw >= self.h-4: break
                at = curses.color_pair(cS)|curses.A_BOLD if i == self.gi else dim()
                sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g, PW-4)}", at)

        col_list_start = rs+1 if self.mode4 else rs+len(self.GRAPHS_CSV)+2
        if not self.mode4:
            sa(self.scr, col_list_start-1, self.w-PW-1, f" {'─'*(PW-1)}", dim())
        list_vis = max(1, self.h - col_list_start - 3)
        if self.num_cols:
            if self.col_idx >= self.col_scroll + list_vis:
                self.col_scroll = self.col_idx - list_vis + 1
            if self.col_idx < self.col_scroll:
                self.col_scroll = self.col_idx
            self.col_scroll = max(0, self.col_scroll)
            for li in range(list_vis):
                ci = self.col_scroll + li
                if ci >= len(self.num_cols): break
                sr = col_list_start + li
                if sr >= self.h - 2: break
                sel = ci == self.col_idx
                at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cMT)
                sa(self.scr, sr, self.w-PW-1, f" {tr(self.num_cols[ci], PW-2)}", at)
        csv_lbl = tr(Path(self.st.get("input_csv","—")).name, PW-2)
        sa(self.scr, self.h-3, self.w-PW-1, f" {csv_lbl}", dim())

        DW = self.w - PW - 3

        # ── Modo 4 gráficos: grilla 2×2 ──────────────────────────────────────
        if self.mode4:
            if not self._load():
                hl(self.scr, rs, 0, DW)
                sa(self.scr, rs+2, 2, f"⚠  {self.errmsg}", curses.color_pair(cER))
                hint_bar(self.scr, self.h-3,
                         " [↑↓]Columna  [M]Modo1-gráfico  [R]Recargar  [f]CSV ")
                return
            if not any(self.q_canvases):
                self._build_quad()

            HH = max(4, (self.h - rs - 4) // 2)
            HW = DW // 2
            labels = ["Histograma", "Boxplot", "Serie/Tendencia", "CDF empírica"]
            cn = self.num_cols[self.col_idx % len(self.num_cols)] if self.num_cols else "—"

            for qi in range(4):
                row_off = rs + (qi // 2) * (HH + 1)
                col_off = (qi % 2) * HW
                hl(self.scr, row_off, col_off, HW)
                lbl = f" {labels[qi]}: {tr(cn, max(1,HW-len(labels[qi])-4))} "
                sa(self.scr, row_off, col_off+1, lbl[:HW-1],
                   curses.color_pair(cGR)|curses.A_BOLD)
                qc = self.q_canvases[qi]
                for li in range(HH-1):
                    if li >= len(qc): break
                    ln = qc[li]
                    if   "✓" in ln:                            at = curses.color_pair(cOK)
                    elif "✗" in ln:                            at = curses.color_pair(cER)
                    elif "══" in ln or "──" in ln:             at = curses.color_pair(cP)|curses.A_BOLD
                    elif "│" in ln and ("█" in ln or "░" in ln or "▪" in ln):
                                                               at = curses.color_pair(cGR)
                    else:                                      at = curses.color_pair(cN)
                    sa(self.scr, row_off+1+li, col_off+1, tr(ln, HW-2), at)
            for r in range(rs, self.h-4):
                sa(self.scr, r, HW, "│", curses.color_pair(cBR))
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Columna  [M]1-gráfico  [R]Recargar  [f]CSV ")
            return

        # ── Modo 1 gráfico ────────────────────────────────────────────────────
        hl(self.scr, rs, 0, DW)
        sa(self.scr, rs, 2, f" {self.GRAPHS_CSV[self.gi][0]} ",
           curses.color_pair(cGR)|curses.A_BOLD)
        if not self.canvas:
            self._build()
        vis = self.h - 6
        ms = max(0, len(self.canvas)-vis+1)
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(vis-1):
            idx = self.scroll+i
            if idx >= len(self.canvas): break
            ln = self.canvas[idx]
            if   "✓" in ln: at = curses.color_pair(cOK)
            elif "✗" in ln: at = curses.color_pair(cER)
            elif "══" in ln or "──" in ln: at = curses.color_pair(cP)|curses.A_BOLD
            elif "│" in ln and ("█" in ln or "░" in ln or "▪" in ln):
                                           at = curses.color_pair(cGR)
            elif "★" in ln: at = curses.color_pair(cMX)
            else: at = curses.color_pair(cN)
            sa(self.scr, rs+1+i, 1, tr(ln, DW-2), at)
        hint_bar(self.scr, self.h-3,
                 " [←→]Gráfico  [↑↓]Columna  [j/k]Scroll  [M]4-gráficos  [R]Recargar  [1-9]Directo ")

    def key(self, k):
        if k == curses.KEY_LEFT:
            self.gi = (self.gi-1) % len(self.GRAPHS_CSV); self.canvas=[]; self.scroll=0
        elif k == curses.KEY_RIGHT:
            self.gi = (self.gi+1) % len(self.GRAPHS_CSV); self.canvas=[]; self.scroll=0
        elif k == curses.KEY_UP:
            if self.num_cols:
                self.col_idx = (self.col_idx-1) % len(self.num_cols)
                self.canvas=[]; self.scroll=0; self.q_canvases=[[], [], [], []]
        elif k == curses.KEY_DOWN:
            if self.num_cols:
                self.col_idx = (self.col_idx+1) % len(self.num_cols)
                self.canvas=[]; self.scroll=0; self.q_canvases=[[], [], [], []]
        elif k == ord('j'): self.scroll += 1
        elif k == ord('k'): self.scroll = max(0, self.scroll-1)
        elif k == ord('m') or k == ord('M'):
            self.mode4 = not self.mode4
            self.canvas=[]; self.scroll=0; self.q_canvases=[[], [], [], []]
        elif k == ord('R'):
            self.dfp=""; self.col_scroll=0; self.canvas=[]
            self.scroll=0; self.q_canvases=[[], [], [], []]
        elif ord('1') <= k <= ord('9'):
            idx = k-ord('1')
            if idx < len(self.GRAPHS_CSV):
                if self.mode4: self.mode4 = False  # cambiar a modo 1 al elegir gráfico
                self.gi = idx; self.canvas=[]; self.scroll=0
        return None


# Métodos auxiliares de draw para RepositoryScreen y StatScreen
# (body sin erase/title para ser llamados desde ListasScreen)
def _repo_draw_body(self):
    rs = 3
    self.title(2, "  ".join(ListasScreen.SUBTABS))   # ya dibujado, but needed for hl
    self.statusbar()
    if self.mode == "pick_csv":
        self._dpick(); self.navbar(); self.scr.refresh(); return
    if not self.dfp: self._load()
    if not self.dfp:
        sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
        hint_bar(self.scr, self.h-3, " [f]Elegir CSV ")
        return
    PW = 28
    hl(self.scr, rs, self.w-PW-2, PW+1)
    sa(self.scr, rs, self.w-PW-1, " FASES ", curses.color_pair(cH)|curses.A_BOLD)
    for i, ph in enumerate(self.phases):
        rw = rs+1+i
        if rw >= self.h-4: break
        at = curses.color_pair(cS)|curses.A_BOLD if i == self.phase_idx else dim()
        sa(self.scr, rw, self.w-PW-1, f" {tr(ph, PW-2)} ", at)
    DW = self.w-PW-3
    hl(self.scr, rs, 0, DW)
    fr = self.filt_rows if hasattr(self, "filt_rows") else []
    pg = self._page_rows() if hasattr(self, "_page_rows") else []
    tp = self._total_pages() if hasattr(self, "_total_pages") else 1
    sa(self.scr, rs, 2, f" COLUMNAS CSV [{len(fr)} cols · pág {self.page+1}/{tp}] ",
       curses.color_pair(cGR)|curses.A_BOLD)
    # cabecera
    hdr_y = rs+1
    x = 1
    for lbl, _, w2 in RepositoryScreen.META_COLS:
        sa(self.scr, hdr_y, x, tr(lbl, w2), curses.color_pair(cH)|curses.A_BOLD)
        x += w2+1
    hl(self.scr, hdr_y+1, 0, DW)
    for i, row in enumerate(pg):
        ry = hdr_y+2+i
        if ry >= self.h-4: break
        sel = i == self.cur_row
        at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
        sa(self.scr, ry, 0, " "*(DW), at)
        x = 1
        for _, key2, w2 in RepositoryScreen.META_COLS:
            sa(self.scr, ry, x, tr(str(row.get(key2,"—")), w2), at)
            x += w2+1
    hint_bar(self.scr, self.h-3,
             " [f]CSV  [←→]Fase  [↑↓]Fila  [n/p]Pág  [e]Vacíos ")

RepositoryScreen._draw_body = _repo_draw_body


def _stat_draw_body(self):
    rs = 3
    subtabs = ["[A]Resumen","[B]Distribución","[C]Pruebas","[D]Homogeneidad"]
    self.statusbar()
    if self.mode == "pick_csv":
        self._dpick(); self.navbar(); self.scr.refresh(); return
    if not self.dfp: self._load()
    if not self.dfp:
        sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
        hint_bar(self.scr, self.h-3, " [f]Elegir CSV ")
        return
    PW = 26
    hl(self.scr, rs, self.w-PW-2, PW+1)
    sa(self.scr, rs, self.w-PW-1, " COLUMNAS NUM ", curses.color_pair(cH)|curses.A_BOLD)
    vis_p = self.h-rs-4
    for i, t in enumerate(subtabs):
        sa(self.scr, rs, 2+i*16, f" {t} ",
           curses.color_pair(cS)|curses.A_BOLD if i == self.sub else dim())
    nc = self.num_cols
    if self.col_idx >= self.col_scroll+vis_p-1: self.col_scroll = self.col_idx-vis_p+2
    if self.col_idx < self.col_scroll:          self.col_scroll = self.col_idx
    self.col_scroll = max(0, self.col_scroll)
    for i, col in enumerate(nc):
        sr = rs+1+(i-self.col_scroll)
        if not (rs+1 <= sr <= self.h-5): continue
        sel = i == self.col_idx
        at  = curses.color_pair(cS)|curses.A_BOLD if sel else dim()
        sa(self.scr, sr, self.w-PW-1, f" {tr(col, PW-2)} ", at)
    DW = self.w-PW-3
    hl(self.scr, rs+1, 0, DW)
    if not self.canvas: self._build_canvas()
    ms = max(0, len(self.canvas)-(self.h-rs-5))
    self.scroll = max(0, min(self.scroll, ms))
    for i in range(self.h-rs-6):
        idx = self.scroll+i
        if idx >= len(self.canvas): break
        ln = self.canvas[idx]
        if   "✓" in ln or "NORMAL" in ln: at = curses.color_pair(cOK)
        elif "✗" in ln or "NO NORMAL" in ln: at = curses.color_pair(cER)
        elif "══" in ln or "──" in ln: at = curses.color_pair(cP)|curses.A_BOLD
        elif "●" in ln: at = curses.color_pair(cMX)
        elif "○" in ln: at = curses.color_pair(cDM)
        elif "►" in ln: at = curses.color_pair(cMT)
        elif "│" in ln and "█" in ln: at = curses.color_pair(cGR)
        else: at = curses.color_pair(cN)
        sa(self.scr, rs+2+i, 1, tr(ln, DW-2), at)
    if self.notif and self.ntimer > 0:
        sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1], curses.color_pair(cOK)|curses.A_BOLD)
        self.ntimer -= 1
    else:
        hint_bar(self.scr, self.h-3,
                 " [A/B/C/D]Sub-tab  [↑↓]Col  [j/k]Scroll  [f]CSV  [R]Recargar ")

StatScreen._draw_body = _stat_draw_body


# ══════════════════════════════════════════════════════════════════════════════
# § 14  PANTALLA 5 — ANALISIS  (ex-Tablas + sub: Gráficos ejecutados + Estadística ejecutados)
#        Sub-tabs: [T]Tablas  [G]Gráficos  [E]Estadística-5pasos
#        Todo evaluado sobre runs ejecutados (BD), no CSV directo
# ══════════════════════════════════════════════════════════════════════════════
class AnálisisScreen(Screen):
    """
    Pestaña [5] Analisis con tres sub-tabs:
      sub=0  [T]ablas      → TableScreen    (tablas de resultados de runs)
      sub=1  [G]ráficos    → GraphScreen    (gráficos de runs ejecutados)
      sub=2  [E]stadística → StatRunScreen  (5 pasos estadísticos sobre runs)
    """
    SUBTABS = ["[T]Tablas", "[G]Gráficos", "[E]Estadística"]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub    = 0
        self._tbl   = TableScreen(scr, st)
        self._grph  = GraphScreen(scr, st)
        self._stat  = StatRunScreen(scr, st)

    def _active(self):
        return [self._tbl, self._grph, self._stat][self.sub]

    def draw(self):
        self.scr.erase(); self._sz()
        hint = "  ".join(
            f" {t} " if i == self.sub else t
            for i, t in enumerate(self.SUBTABS)
        )
        self.title(4, hint)
        act = self._active()
        act.h = self.h; act.w = self.w
        if   self.sub == 0: self._tbl._draw_body()
        elif self.sub == 1: self._grph._draw_body_analisis()
        else:               self._stat._draw_body()
        self.navbar(f"  [T/G/E]Sub-tab")

    def key(self, k):
        if   k in (ord('t'), ord('T')): self.sub = 0; return None
        elif k in (ord('g'), ord('G')): self.sub = 1; return None
        elif k in (ord('e'), ord('E')): self.sub = 2; return None
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        res = self._active().key(k)
        if res in ("next", "prev", "quit"): return res
        return None


# _draw_body para TableScreen (body sin erase para AnálisisScreen)
def _table_draw_body(self):
    rs = 3
    self.statusbar()
    self._reload_runs()
    if self.mode == "pick_run":
        self._dpick_run(); return
    if self.mode == "detail":
        self._draw_detail(); return
    if not self.cur_run:
        sa(self.scr, rs+2, 4, "Sin run seleccionada. Presiona [r] para elegir.",
           curses.color_pair(cER))
        hint_bar(self.scr, self.h-3, " [r]Elegir run ")
        return
    if not self.dfp: self._load()
    if self.errmsg:
        sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
        hint_bar(self.scr, self.h-3, " [r]Elegir run "); return
    # Encabezado
    run_lbl = tr(self.cur_run.get("config_name","?"), 40)
    tp = self._total_pages(); pg = self._page_rows()
    sa(self.scr, rs, 2,
       f" Run: {run_lbl}  Pág {self.page+1}/{tp}  [{len(self.rows)} filas] ",
       curses.color_pair(cGR)|curses.A_BOLD)
    hl(self.scr, rs+1, 0, self.w-1)
    # Cabecera columnas
    x = 1
    for ci, (lbl, _, w2) in enumerate(self.TABLE_COLS):
        at = curses.color_pair(cS)|curses.A_BOLD if ci == self.header_ci else curses.color_pair(cH)|curses.A_BOLD
        sa(self.scr, rs+2, x, tr(lbl, w2), at); x += w2+1
    hl(self.scr, rs+3, 0, self.w-1)
    # Filas
    for ri, row in enumerate(pg):
        ry = rs+4+ri
        if ry >= self.h-4: break
        sel = ri == self.cur_row
        at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
        sa(self.scr, ry, 0, " "*(self.w-1), at)
        x = 1
        for _, col2, w2 in self.TABLE_COLS:
            sa(self.scr, ry, x, tr(str(row.get(col2,"—")), w2), at); x += w2+1
    hint_bar(self.scr, self.h-3,
             " [r]Run  [←→]Col  [↑↓]Fila  [n/p]Pág  [s]Sort  [Enter]Detalle ")

TableScreen._draw_body = _table_draw_body


# _draw_body_analisis para GraphScreen (dentro de Analisis, sin erase/title)
def _graph_draw_body_analisis(self):
    rs = 3; vis = self.h-6; PW = 28
    self.statusbar()
    if self.mode == "pick_run":
        self._dpick_run(); return
    hl(self.scr, rs, self.w-PW-2, PW+1)
    sa(self.scr, rs, self.w-PW-1, " GRÁFICOS RUNS ", curses.color_pair(cH)|curses.A_BOLD)
    for i, (g, _) in enumerate(self.GRAPHS):
        rw = rs+1+i
        if rw >= self.h-4: break
        at = curses.color_pair(cS)|curses.A_BOLD if i == self.gi else dim()
        sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g, PW-4)}", at)
    run_lbl = tr(self.cur_run.get("config_name","(última run)"), PW-6) if self.cur_run else "(última run)"
    sa(self.scr, rs+10, self.w-PW-1, f" Run: {run_lbl}", curses.color_pair(cMT))
    DW = self.w-PW-3
    hl(self.scr, rs, 0, DW)
    sa(self.scr, rs, 2, f" {self.GRAPHS[self.gi][0]} ", curses.color_pair(cGR)|curses.A_BOLD)
    if not self.canvas: self._build()
    ms = max(0, len(self.canvas)-vis+1)
    self.scroll = max(0, min(self.scroll, ms))
    for i in range(vis-1):
        idx = self.scroll+i
        if idx >= len(self.canvas): break
        ln = self.canvas[idx]
        if   "★" in ln: at = curses.color_pair(cMX)
        elif "│" in ln and ("█" in ln or "░" in ln): at = curses.color_pair(cGR)
        elif any(c in ln for c in "┌└┐┘"): at = curses.color_pair(cBR)
        elif "Pareto" in ln: at = curses.color_pair(cOK)
        else: at = curses.color_pair(cN)
        sa(self.scr, rs+1+i, 1, tr(ln, DW-2), at)
    hint_bar(self.scr, self.h-3,
             " [←→]Gráfico  [↑↓]Scroll  [r]Elegir run  [R]Recargar  [1-8]Directo ")

GraphScreen._draw_body_analisis = _graph_draw_body_analisis


# ══════════════════════════════════════════════════════════════════════════════
# § 15  STAT-RUN SCREEN — sub-tab E de Analisis
#        Estadística completa en 5 pasos sobre runs ejecutados
# ══════════════════════════════════════════════════════════════════════════════
class StatRunScreen(Screen):
    """
    Estadística sobre runs ejecutados (multiobjective_report.csv).
    5 pasos: Supuestos → Elegir prueba → Ejecutar → Interpretar → Post-hoc.
    Sub-tabs: [A]Resumen  [B]Distribución  [C]Pruebas  [D]Homogeneidad  [Z]5-Pasos
    """
    SUBTABS_SR = ["[A]Resumen","[B]Distribución","[C]Pruebas","[D]Homogeneidad","[Z]5-Pasos"]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub       = 0
        self.runs      = []; self.run_cur = 0; self.run_scroll = 0
        self.cur_run   = {}
        self.df        = None; self.dfp = ""
        self.num_cols  = []
        self.col_idx   = 0; self.col_scroll = 0
        self.scroll    = 0
        self.canvas    = []
        self.errmsg    = ""
        self.mode      = "normal"   # normal | pick_run | pick_fields
        self.notif     = ""; self.ntimer = 0
        self.phase_idx = 0
        self.phases    = ["Todas"] + [p for p,_ in RepositoryScreen.PHASE_PREFIXES] + ["Otras"]
        # Para 5-Pasos: grupo categórico y selección de campos
        self.grp_cols    = []   # columnas categóricas detectadas
        self.grp_idx     = 0    # índice del grupo seleccionado
        # Selector de campos para 5-Pasos
        self.pf_met_idx  = 0    # índice columna métrica seleccionada en pick_fields
        self.pf_grp_idx  = 0    # índice columna grupo seleccionada en pick_fields
        self.pf_cursor   = 0    # 0=sobre lista métrica, 1=sobre lista grupo
        self.pf_confirmed= False  # se ejecuta 5-pasos tras confirmar
        self._reload_runs()

    def _reload_runs(self):
        self.runs = run_list()
        lr = self.st.get("last_result","")
        for i, r in enumerate(self.runs):
            if r["out_dir"] == lr:
                self.run_cur = i; self.cur_run = r; break

    def _load(self):
        if not self.cur_run: self.errmsg="Sin run. Presiona [r]."; return False
        out = self.cur_run.get("out_dir","")
        cp  = Path(out)/"multiobjective_report.csv"
        if not cp.exists(): self.errmsg=f"CSV no encontrado: {cp}"; return False
        if self.dfp == str(cp): return True
        try:
            self.df = pd.read_csv(cp, nrows=5000); self.dfp = str(cp)
            # Excluir bool (causa TypeError en percentile) y priorizar escalarización
            SCALAR_PRIO = ["WSum","WProd","Tcheby","ASF","Hypervolume",
                           "R_WSum","R_Tcheby","R_Consensus","Pareto_Level"]
            all_num = [c for c in self.df.columns
                       if pd.api.types.is_numeric_dtype(self.df[c])
                       and str(self.df[c].dtype) != "bool"]
            prio  = [c for c in SCALAR_PRIO if c in all_num]
            resto = [c for c in all_num if c not in prio]
            self.num_cols = prio + resto
            cat_types = ["object","category"]
            self.grp_cols = [c for c in self.df.columns
                             if str(self.df[c].dtype) in cat_types]
            self.col_idx=0; self.canvas=[]; self.errmsg=""; return True
        except Exception as e:
            self.errmsg=f"Error: {e}"; return False

    def _col(self):
        if not self.num_cols: return None, None
        c = self.num_cols[self.col_idx]
        v = pd.to_numeric(self.df[c], errors="coerce").dropna().values
        return c, v

    def _stats(self, v):
        if not _NUM or v is None or len(v)<2: return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # silenciar RuntimeWarning/UserWarning de scipy
            from scipy import stats as sp
            v = np.asarray(v, dtype=float)
            n = len(v)
            mu  = float(np.mean(v))
            med = float(np.median(v))
            rng = float(v.max()-v.min())
            # Datos constantes → std/skew/kurt no definidos
            if rng == 0:
                mo = mu
                sd = sk = ku = 0.0
                mn = mx = mu
                q1 = q3 = mu
                return {"n":n,"mean":mu,"median":med,"mode":mo,"std":sd,"skew":sk,"kurt":ku,
                        "min":mn,"max":mx,"q1":q1,"q3":q3,
                        "sw_s":1.0,"sw_p":1.0,"ks_s":None,"ks_p":None,
                        "_constant":True}
            sd  = float(np.std(v, ddof=1))
            try:   sk = float(sp.skew(v))
            except Exception: sk = 0.0
            try:   ku = float(sp.kurtosis(v))
            except Exception: ku = 0.0
            mn = float(v.min()); mx = float(v.max())
            q1 = float(np.percentile(v, 25)); q3 = float(np.percentile(v, 75))
            try:   mo = float(sp.mode(v, keepdims=True).mode[0])
            except Exception: mo = mu
            # Shapiro-Wilk
            sw_s = sw_p = None
            if 3 <= n <= 5000:
                try:
                    sw_s, sw_p = sp.shapiro(v[:5000])
                    sw_s = float(sw_s); sw_p = float(sw_p)
                except Exception: pass
            # Kolmogorov-Smirnov
            ks_s = ks_p = None
            try:
                norm_v = (v-mu)/max(sd,1e-9)
                ks_s, ks_p = sp.kstest(norm_v, "norm")
                ks_s = float(ks_s); ks_p = float(ks_p)
            except Exception: pass
        return {"n":n,"mean":mu,"median":med,"mode":mo,"std":sd,"skew":sk,"kurt":ku,
                "min":mn,"max":mx,"q1":q1,"q3":q3,"sw_s":sw_s,"sw_p":sw_p,
                "ks_s":ks_s,"ks_p":ks_p}

    def _build_canvas(self):
        self.canvas=[]
        if not _NUM: self.canvas=["  ✗ numpy/pandas/scipy no disponibles."]; return
        if not self._load(): self.canvas=[f"  ⚠ {self.errmsg}"]; return
        if not self.num_cols: self.canvas=["  Sin columnas numéricas en el reporte."]; return
        c,v=self._col()
        if v is None or len(v)<2: self.canvas=[f"  '{c}' sin datos suficientes."]; return
        v = np.asarray(v, dtype=float)   # garantizar float64
        s=self._stats(v)
        if not s: self.canvas=["  No se pudo calcular estadísticos."]; return
        if   self.sub==0: self._canvas_summary(c,v,s)
        elif self.sub==1: self._canvas_dist(c,v,s)
        elif self.sub==2: self._canvas_tests(c,v,s)
        elif self.sub==3: self._canvas_homog()
        elif self.sub==4:
            try:
                self._canvas_5pasos(c,v,s)
            except Exception as e:
                self.canvas=[f"  Error en 5-Pasos: {e}",
                             "","  Revisa que el run tenga columnas categóricas (vlm_model, cls_model, etc.)"]

    # ── reutilizar métodos de StatScreen ──────────────────────────────────────
    _canvas_summary = StatScreen._canvas_summary
    _canvas_dist    = StatScreen._canvas_dist
    _canvas_tests   = StatScreen._canvas_tests
    _canvas_homog   = StatScreen._canvas_homog

    def _canvas_5pasos(self, c, v, s):
        """5 pasos estadísticos completos sobre la columna seleccionada."""
        from scipy import stats as sp
        L = self.canvas.append
        alpha = 0.05
        sw_p  = s.get("sw_p")
        ks_p  = s.get("ks_p")
        normal = (sw_p is not None and sw_p > alpha)
        n = s.get("n", 0)
        # Respetar selección del usuario desde pick_fields
        force_no_grp = getattr(self, "_force_no_group", False)
        grp_cols_eff = [] if force_no_grp else self.grp_cols

        L(f"  ══ ANÁLISIS ESTADÍSTICO EN 5 PASOS: {c} ══")
        L(f"  n={n}   α={alpha}   Run: {tr(self.cur_run.get('config_name','?'),40)}")
        if s.get("_constant"):
            L("  ⚠ Datos constantes (rango=0) — resultados estadísticos triviales")
        L("")

        # ── PASO 1 ─ Supuestos ───────────────────────────────────────────────
        L("  ┌─ PASO 1 ─ VERIFICAR SUPUESTOS ──────────────────────────────────┐")
        # Shapiro-Wilk
        if sw_p is not None:
            ok_sw = sw_p > alpha
            if s.get("_constant"):
                L(f"  │  Shapiro-Wilk  W={s.get('sw_s',1):.4f}  p={sw_p:.4f}  (datos constantes → p=1.0)")
            else:
                L(f"  │  Shapiro-Wilk  W={s.get('sw_s') or 0:.4f}  p={sw_p:.4f}")
            L(f"  │  → {'✓ p>0.05 Datos NORMALES' if ok_sw else '✗ p≤0.05 Datos NO NORMALES'}")
        else:
            L("  │  Shapiro-Wilk: no calculable (n<3 o n>5000)")
            ok_sw = False
        # Levene
        lev_p = None
        if n >= 6 and grp_cols_eff:
            gcol = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grupos_lev = [pd.to_numeric(self.df.loc[self.df[gcol]==g, c],errors="coerce").dropna().values.astype(float)
                              for g in self.df[gcol].dropna().unique()]
                grupos_lev = [g for g in grupos_lev if len(g)>=3 and g.std()>0]
                if len(grupos_lev)>=2:
                    try:
                        _stat_lv, lev_p = sp.levene(*grupos_lev); lev_p=float(lev_p)
                    except Exception: pass
        if lev_p is not None:
            ok_lev = lev_p > alpha
            L(f"  │  Levene (grupos '{grp_cols_eff[self.grp_idx % len(grp_cols_eff)]}')  p={lev_p:.4f}")
            L(f"  │  → {'✓ p>0.05 Varianzas HOMOGÉNEAS' if ok_lev else '✗ p≤0.05 Varianzas HETEROGÉNEAS'}")
        else:
            ok_lev = True
            gcol_nm = grp_cols_eff[self.grp_idx % len(grp_cols_eff)] if grp_cols_eff else "—"
            L(f"  │  Levene: {'grupos constantes o insuficientes' if grp_cols_eff else 'sin grupo seleccionado'} → asumir homogéneo")
        L("  └────────────────────────────────────────────────────────────────┘")
        L("")

        # ── PASO 2 ─ Elegir prueba ───────────────────────────────────────────
        L("  ┌─ PASO 2 ─ ELEGIR PRUEBA ────────────────────────────────────────┐")
        n_grp = 0
        if self.grp_cols:
            gcol = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
            n_grp = self.df[gcol].dropna().nunique()
        if normal and ok_lev:
            if n_grp == 2:   prueba="t de Student (ttest_ind)"
            elif n_grp > 2:  prueba="ANOVA de un factor (f_oneway)"
            else:            prueba="t de Student (una muestra, vs media)"
            family = "PARAMÉTRICA"
        else:
            if n_grp == 2:   prueba="Mann-Whitney U (mannwhitneyu)"
            elif n_grp > 2:  prueba="Kruskal-Wallis (kruskal)"
            else:            prueba="Wilcoxon signed-rank (wilcoxon)"
            family = "NO PARAMÉTRICA"
        L(f"  │  Normalidad: {'✓' if normal else '✗'}   Homogeneidad: {'✓' if ok_lev else '✗'}   Grupos: {n_grp if n_grp else '1'}")
        L(f"  │  → Familia: {family}")
        L(f"  │  → Prueba:  {prueba}")
        L("  └────────────────────────────────────────────────────────────────┘")
        L("")

        # ── PASO 3 ─ Ejecutar ────────────────────────────────────────────────
        L("  ┌─ PASO 3 ─ EJECUTAR PRUEBA ──────────────────────────────────────┐")
        test_p = None; test_stat = None; test_label = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if grp_cols_eff and n_grp >= 2:
                    gcol = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
                    grupos = [pd.to_numeric(self.df.loc[self.df[gcol]==g, c],errors="coerce").dropna().values.astype(float)
                              for g in self.df[gcol].dropna().unique()]
                    grupos = [g for g in grupos if len(g)>=2]
                    if len(grupos) >= 2:
                        if normal and ok_lev:
                            if len(grupos)==2:
                                res_t = sp.ttest_ind(grupos[0], grupos[1], equal_var=True)
                                test_stat=float(res_t.statistic); test_p=float(res_t.pvalue)
                                test_label=f"t={test_stat:.4f}"
                            else:
                                res_f = sp.f_oneway(*grupos)
                                test_stat=float(res_f.statistic); test_p=float(res_f.pvalue)
                                test_label=f"F={test_stat:.4f}"
                        else:
                            if len(grupos)==2:
                                res_mw = sp.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
                                test_stat=float(res_mw.statistic); test_p=float(res_mw.pvalue)
                                test_label=f"U={test_stat:.1f}"
                            else:
                                res_kw = sp.kruskal(*grupos)
                                test_stat=float(res_kw.statistic); test_p=float(res_kw.pvalue)
                                test_label=f"H={test_stat:.4f}"
                if test_p is None:
                    if s.get("_constant"):
                        L("  │  ⚠ Datos constantes — prueba no aplicable (varianza=0)")
                    elif normal:
                        res_t1 = sp.ttest_1samp(v, 0.5)
                        test_stat=float(res_t1.statistic); test_p=float(res_t1.pvalue)
                        # Manejar inf
                        if not np.isfinite(test_stat): test_stat=0.0; test_p=1.0
                        test_label=f"t={test_stat:.4f} (vs μ₀=0.5)"
                    else:
                        dif = v-0.5
                        if np.any(dif!=0):
                            res_wx = sp.wilcoxon(dif)
                            test_stat=float(res_wx.statistic); test_p=float(res_wx.pvalue)
                            test_label=f"W={test_stat:.1f} (vs 0.5)"
                        else:
                            L("  │  ⚠ Todos los valores iguales a 0.5 — prueba no aplicable")
            except Exception as e:
                L(f"  │  Error ejecutando prueba: {e}")
                test_p = None
        if test_p is not None:
            L(f"  │  Prueba: {prueba}")
            L(f"  │  Estadístico: {test_label}")
            L(f"  │  p-valor = {test_p:.4f}")
        L("  └────────────────────────────────────────────────────────────────┘")
        L("")

        # ── PASO 4 ─ Interpretar ─────────────────────────────────────────────
        L("  ┌─ PASO 4 ─ INTERPRETAR RESULTADO ────────────────────────────────┐")
        if test_p is not None:
            sig = test_p < alpha
            L(f"  │  p-valor ({test_p:.4f}) {'<' if sig else '≥'} α ({alpha})")
            if sig:
                L(f"  │  → ✓ DIFERENCIA SIGNIFICATIVA  (rechazar H₀)")
                L(f"  │  → La variable '{tr(c,30)}' difiere entre grupos")
                if grp_cols_eff and n_grp == 2:
                    try:
                        gcol2 = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
                        g1,g2 = [pd.to_numeric(self.df.loc[self.df[gcol2]==g,c],errors="coerce").dropna().values.astype(float)
                                 for g in list(self.df[gcol2].dropna().unique())[:2]]
                        s1=np.std(g1,ddof=1); s2=np.std(g2,ddof=1)
                        pooled_sd = np.sqrt((s1**2+s2**2)/2)
                        if pooled_sd > 0:
                            d = abs(np.mean(g1)-np.mean(g2))/pooled_sd
                            mag = "pequeño" if d<0.5 else ("mediano" if d<0.8 else "grande")
                            L(f"  │  → Tamaño efecto Cohen d={d:.3f} ({mag})")
                    except Exception: pass
            else:
                L(f"  │  → ✗ SIN DIFERENCIA SIGNIFICATIVA  (no rechazar H₀)")
                L(f"  │  → No hay evidencia de diferencia entre grupos")
        else:
            L("  │  No se pudo calcular p-valor.")
        L("  └────────────────────────────────────────────────────────────────┘")
        L("")

        # ── PASO 5 ─ Post-hoc ────────────────────────────────────────────────
        L("  ┌─ PASO 5 ─ POST-HOC ─────────────────────────────────────────────┐")
        if test_p is not None and test_p < alpha and n_grp > 2:
            L("  │  Comparaciones múltiples necesarias (grupos > 2):")
            if normal and ok_lev:
                L("  │  → Tukey HSD  (statsmodels.stats.multicomp.pairwise_tukeyhsd)")
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    gcol3 = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
                    df_ph = self.df[[gcol3,c]].dropna().copy()
                    df_ph[c] = pd.to_numeric(df_ph[c],errors="coerce"); df_ph=df_ph.dropna()
                    res_tk = pairwise_tukeyhsd(df_ph[c].values, df_ph[gcol3].values, alpha=alpha)
                    for row_tk in res_tk.summary().data[1:]:
                        g1t,g2t,_,lb,ub,rej = row_tk
                        marca = "✓ sig" if rej else "○ n.s."
                        L(f"  │    {str(g1t):<15} vs {str(g2t):<15}  {marca}")
                except ImportError:
                    L("  │    statsmodels no instalado → pip install statsmodels")
                except Exception as e:
                    L(f"  │    Tukey error: {e}")
            else:
                L("  │  → Dunn / Nemenyi  (scikit-posthocs)")
                try:
                    import scikit_posthocs as sp_ph
                    gcol3 = grp_cols_eff[self.grp_idx % len(grp_cols_eff)]
                    df_ph = self.df[[gcol3,c]].dropna().copy()
                    df_ph[c] = pd.to_numeric(df_ph[c],errors="coerce"); df_ph=df_ph.dropna()
                    res_d = sp_ph.posthoc_dunn(df_ph, val_col=c, group_col=gcol3, p_adjust="bonferroni")
                    L("  │  Dunn (Bonferroni):")
                    for row_d in res_d.index:
                        for col_d in res_d.columns:
                            if row_d < col_d:
                                pv_d = res_d.loc[row_d,col_d]
                                m = "✓ sig" if pv_d<alpha else "○ n.s."
                                L(f"  │    {str(row_d):<15} vs {str(col_d):<15}  p={pv_d:.4f} {m}")
                except ImportError:
                    L("  │    scikit-posthocs no instalado → pip install scikit-posthocs")
                except Exception as e:
                    L(f"  │    Dunn error: {e}")
        elif test_p is not None and test_p < alpha and n_grp == 2:
            L("  │  Solo 2 grupos → post-hoc NO aplica")
            L("  │  La prueba principal ya identifica el par significativo")
        elif test_p is not None:
            L("  │  No significativo → post-hoc NO aplica")
        else:
            L("  │  Sin resultado de prueba → post-hoc NO aplica")
        L("  └────────────────────────────────────────────────────────────────┘")
        L("")
        # Resumen ejecutivo
        L("  ── RESUMEN EJECUTIVO ──────────────────────────────────────────────")
        L(f"  Columna  : {c}")
        L(f"  n        : {n}")
        L(f"  Familia  : {family}")
        L(f"  Prueba   : {prueba}")
        if test_p is not None:
            L(f"  p-valor  : {test_p:.4f}  ({'SIGNIFICATIVO' if test_p<alpha else 'NO significativo'})")
        L(f"  Post-hoc : {'Tukey/Dunn ejecutado' if (test_p is not None and test_p<alpha and n_grp>2) else 'No aplica'}")

    # ── draw_body (para AnálisisScreen) ──────────────────────────────────────
    def _draw_body(self):
        rs = 3
        self.statusbar()
        if self.mode == "pick_run":
            self._dpick_run(); return
        if self.mode == "pick_fields":
            self._dpick_fields(); return
        if not self.dfp: self._load()
        if not self.dfp:
            sa(self.scr, rs+2, 4, f"⚠ {self.errmsg}", curses.color_pair(cER))
            hint_bar(self.scr, self.h-3, " [r]Elegir run "); return

        PW = 26
        hl(self.scr, rs, self.w-PW-2, PW+1)
        sa(self.scr, rs, self.w-PW-1, " COLUMNAS NUM ", curses.color_pair(cH)|curses.A_BOLD)
        vis_p = self.h-rs-4
        for i, t in enumerate(self.SUBTABS_SR):
            sa(self.scr, rs, 2+i*14, f" {t} ",
               curses.color_pair(cS)|curses.A_BOLD if i==self.sub else dim())
        nc = self.num_cols
        if self.col_idx >= self.col_scroll+vis_p-1: self.col_scroll = self.col_idx-vis_p+2
        if self.col_idx < self.col_scroll:          self.col_scroll = self.col_idx
        self.col_scroll = max(0, self.col_scroll)
        for i, col in enumerate(nc):
            sr = rs+1+(i-self.col_scroll)
            if not (rs+1 <= sr <= self.h-5): continue
            sel = i == self.col_idx
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else dim()
            sa(self.scr, sr, self.w-PW-1, f" {tr(col, PW-2)} ", at)
        DW = self.w-PW-3
        hl(self.scr, rs+1, 0, DW)
        if not self.canvas: self._build_canvas()
        ms = max(0, len(self.canvas)-(self.h-rs-5))
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(self.h-rs-6):
            idx = self.scroll+i
            if idx >= len(self.canvas): break
            ln = self.canvas[idx]
            if   "✓" in ln or "SIGNIFICATIVO" in ln: at = curses.color_pair(cOK)
            elif "✗" in ln or "NO NORMAL" in ln: at = curses.color_pair(cER)
            elif "══" in ln or "──" in ln or "┌" in ln or "└" in ln or "│" in ln: at = curses.color_pair(cP)|curses.A_BOLD
            elif "●" in ln: at = curses.color_pair(cMX)
            elif "○" in ln: at = curses.color_pair(cDM)
            elif "►" in ln: at = curses.color_pair(cMT)
            else: at = curses.color_pair(cN)
            sa(self.scr, rs+2+i, 1, tr(ln, DW-2), at)
        if self.notif and self.ntimer > 0:
            sa(self.scr, self.h-3, 0, f" {self.notif} "[:self.w-1], curses.color_pair(cOK)|curses.A_BOLD)
            self.ntimer -= 1
        else:
            if self.sub == 4:
                hint_bar(self.scr, self.h-3,
                         " [Z]Reconfigurar campos  [↑↓]Col  [j/k]Scroll  [r]Run  [R]Recargar ")
            else:
                hint_bar(self.scr, self.h-3,
                         " [A-D]Sub-tab  [Z]5-Pasos  [↑↓]Col  [j/k]Scroll  [r]Run  [R]Recargar  [g]Grupo ")

    def _dpick_fields(self):
        """Pantalla de selección de campos para 5-Pasos."""
        rs = 3; b = curses.color_pair(cBR)
        self._sz()
        H = self.h; W = self.w
        # Marco
        sa(self.scr, rs,   4, "╔" + "═"*70 + "╗", b)
        sa(self.scr, rs+1, 4, "║  CONFIGURAR 5-PASOS — Seleccionar columnas" + " "*27 + "║", b)
        sa(self.scr, rs+2, 4, "╠" + "═"*70 + "╣", b)
        # Panel izquierdo: columnas métricas
        MX = 36   # ancho panel izquierdo
        sa(self.scr, rs+3, 6,  " COLUMNA MÉTRICA (a analizar) ", curses.color_pair(cH)|curses.A_BOLD)
        sa(self.scr, rs+3, 6+MX+2, " COLUMNA DE GRUPO (categórica) ", curses.color_pair(cH)|curses.A_BOLD)
        # Lista métricas
        vis = min(H - rs - 8, 15)
        met_scroll = max(0, self.pf_met_idx - vis + 2)
        for i, col in enumerate(self.num_cols):
            si = rs+4+(i-met_scroll)
            if not (rs+4 <= si <= rs+4+vis): continue
            sel = (i == self.pf_met_idx) and (self.pf_cursor == 0)
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            mk  = "► " if sel else "  "
            sa(self.scr, si, 6, f"{mk}{tr(col, MX-3)}", at)
        # Lista grupos
        grp_opts = ["(sin grupo — 1 muestra)"] + self.grp_cols
        grp_scroll = max(0, self.pf_grp_idx - vis + 2)
        for i, col in enumerate(grp_opts):
            si = rs+4+(i-grp_scroll)
            if not (rs+4 <= si <= rs+4+vis): continue
            sel = (i == self.pf_grp_idx) and (self.pf_cursor == 1)
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            mk  = "► " if sel else "  "
            sa(self.scr, si, 6+MX+2, f"{mk}{tr(col, MX-3)}", at)
        # Selección actual
        met_sel = self.num_cols[self.pf_met_idx] if self.num_cols else "—"
        grp_sel = grp_opts[self.pf_grp_idx] if grp_opts else "—"
        bot = rs+4+vis+1
        sa(self.scr, bot,   4, "╠" + "═"*70 + "╣", b)
        sa(self.scr, bot+1, 6, f"Métrica seleccionada : {tr(met_sel, 44)}", curses.color_pair(cMX)|curses.A_BOLD)
        sa(self.scr, bot+2, 6, f"Grupo    seleccionado: {tr(grp_sel, 44)}", curses.color_pair(cMT)|curses.A_BOLD)
        sa(self.scr, bot+3, 4, "╠" + "═"*70 + "╣", b)
        sa(self.scr, bot+4, 6, "[←→] Cambiar panel   [↑↓] Mover   [Enter] Confirmar y ejecutar   [Esc] Cancelar", dim())
        sa(self.scr, bot+5, 4, "╚" + "═"*70 + "╝", b)
        hint_bar(self.scr, self.h-3, " [←→]Panel  [↑↓]Mover  [Enter]Ejecutar 5-Pasos  [Esc]Cancelar ")

    def _dpick_run(self):
        rs = 3
        sa(self.scr, rs, 2, "SELECCIONAR RUN PARA ESTADÍSTICA",
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+1, 0, self.w-1)
        sa(self.scr, rs+2, 0,
           f"  {'ID':>4}  {'Config':<22} {'CSV':<32} {'Fecha':<19}  {'Filas':>6}  {'Ok'}"[:self.w-1],
           curses.color_pair(cH)|curses.A_BOLD)
        hl(self.scr, rs+3, 0, self.w-1)
        vis = self.h-rs-6
        if not self.runs:
            sa(self.scr, rs+5, 4, "Sin runs. Ejecuta el análisis en P4 → [e].", dim()); return
        if self.run_cur >= self.run_scroll+vis-1: self.run_scroll = self.run_cur-vis+2
        if self.run_cur < self.run_scroll:        self.run_scroll = self.run_cur
        self.run_scroll = max(0, self.run_scroll)
        for idx, run in enumerate(self.runs):
            sr = rs+4+(idx-self.run_scroll)
            if not (rs+4 <= sr <= self.h-5): continue
            sel = idx == self.run_cur
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else curses.color_pair(cN)
            csv_n = Path(run["input_csv"]).name if run["input_csv"] else "—"
            ok    = "✓" if run["out_dir"] and Path(run["out_dir"]).exists() else "✗"
            ln    = (f"  {run['id']:>4}  {tr(run['config_name'],22):<22} "
                     f"{tr(csv_n,32):<32} {run['ran_at'][:19]:<19}  {run['n_rows']:>6}  {ok}")
            sa(self.scr, sr, 0, " "*(self.w-1), at)
            sa(self.scr, sr, 0, ln[:self.w-1], at)
        hint_bar(self.scr, self.h-3, " [↑↓]Mover  [Enter]Cargar  [Esc]Cancelar ")

    def key(self, k):
        if self.mode == "pick_run":
            n = len(self.runs)
            if   k in (curses.KEY_UP,   ord('k')) and self.run_cur > 0:   self.run_cur -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and self.run_cur < n-1: self.run_cur += 1
            elif k in (ord('\n'), ord('\r'), 10, 13) and n > 0:
                self.cur_run=self.runs[self.run_cur]; self.dfp=""; self.canvas=[]
                self.mode="normal"; self._load()
            elif k in (27, ord('q')): self.mode="normal"
            return None

        if self.mode == "pick_fields":
            grp_opts = ["(sin grupo — 1 muestra)"] + self.grp_cols
            if k == curses.KEY_LEFT:
                self.pf_cursor = 0
            elif k == curses.KEY_RIGHT:
                self.pf_cursor = 1
            elif k == curses.KEY_UP:
                if self.pf_cursor == 0:
                    self.pf_met_idx = max(0, self.pf_met_idx-1)
                else:
                    self.pf_grp_idx = max(0, self.pf_grp_idx-1)
            elif k == curses.KEY_DOWN:
                if self.pf_cursor == 0:
                    self.pf_met_idx = min(len(self.num_cols)-1, self.pf_met_idx+1)
                else:
                    self.pf_grp_idx = min(len(grp_opts)-1, self.pf_grp_idx+1)
            elif k in (ord('\n'), ord('\r'), 10, 13):
                # Confirmar selección → ejecutar 5-pasos con los campos elegidos
                self.col_idx = self.pf_met_idx
                # Ajustar grp_idx: 0 = sin grupo → grp_idx inválido; ≥1 → índice en grp_cols
                if self.pf_grp_idx == 0:
                    self.grp_idx = 0
                    self._force_no_group = True
                else:
                    self.grp_idx = self.pf_grp_idx - 1
                    self._force_no_group = False
                self.sub = 4; self.canvas = []; self.scroll = 0
                self.mode = "normal"
                met_nm = self.num_cols[self.col_idx] if self.num_cols else "?"
                self.notif = f"5-Pasos → {tr(met_nm,30)}"; self.ntimer = 80
            elif k in (27, ord('q')):
                self.mode = "normal"
            return None

        # Modo normal
        if   k in (ord('a'),ord('A')): self.sub=0; self.canvas=[]; self.scroll=0
        elif k in (ord('b'),ord('B')): self.sub=1; self.canvas=[]; self.scroll=0
        elif k in (ord('c'),ord('C')): self.sub=2; self.canvas=[]; self.scroll=0
        elif k in (ord('d'),ord('D')): self.sub=3; self.canvas=[]; self.scroll=0
        elif k in (ord('z'),ord('Z')):
            if self.sub == 4:
                # Ya en 5-Pasos → reabrir selector de campos
                self.pf_cursor = 0; self.mode = "pick_fields"
            else:
                # Primera vez → abrir selector si hay datos, si no ejecutar directo
                if self.num_cols:
                    self.pf_met_idx = self.col_idx
                    self.pf_grp_idx = 0; self.pf_cursor = 0
                    self.mode = "pick_fields"
                else:
                    self.sub=4; self.canvas=[]; self.scroll=0
        elif k in (ord('r'),) : self._reload_runs(); self.mode="pick_run"
        elif k == ord('R'): self.dfp=""; self.canvas=[]; self.scroll=0
        elif k in (ord('g'),ord('G')):
            if self.grp_cols: self.grp_idx=(self.grp_idx+1)%len(self.grp_cols); self.canvas=[]; self.scroll=0
        elif k==curses.KEY_UP:   self.col_idx=max(0,self.col_idx-1); self.canvas=[]; self.scroll=0
        elif k==curses.KEY_DOWN: self.col_idx=min(max(0,len(self.num_cols)-1),self.col_idx+1); self.canvas=[]; self.scroll=0
        elif k==ord('k'): self.scroll=max(0,self.scroll-1)
        elif k==ord('j'): self.scroll+=1
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 11b  PANTALLA 6 — MODELOS
#   Sub-menús: [D]etección  [C]lasificación  [V]LM  [L]LM
#   Detección → sub-tabs: [1]Detalle  [2]Métricas  [3]Gráficos
# ══════════════════════════════════════════════════════════════════════════════

# ── Rutas base ────────────────────────────────────────────────────────────────
_DET_BASE   = Path("/workspace/models/phase1_detection/production_det")
_DET_COMBOS = [_DET_BASE / f"combo_00{i}" for i in range(1, 5)]

def _load_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return None


# ── Catálogo de métricas por fase ────────────────────────────────────────────
# Todas las métricas posibles clasificadas por origen y tipo.
_METRICS_CATALOG = {
    # ── F1 Detección ──────────────────────────────────────────────────────────
    "det": {
        "rendimiento": [
            ("mAP50",       "mAP@50",            "max", "Precisión media a IoU=0.50"),
            ("mAP50-95",    "mAP@50-95",          "max", "Precisión media IoU=0.50:0.95 (COCO)"),
            ("precision",   "Precision",          "max", "TP/(TP+FP)"),
            ("recall",      "Recall",             "max", "TP/(TP+FN)"),
            ("box_loss",    "Box Loss",           "min", "Pérdida de localización bounding box"),
            ("cls_loss",    "Cls Loss",           "min", "Pérdida de clasificación"),
            ("dfl_loss",    "DFL Loss",           "min", "Distribution Focal Loss"),
        ],
        "recursos": [
            ("training_time_seconds", "Tiempo train (s)",   "min", "Segundos de entrenamiento"),
            ("training_time_hours",   "Tiempo train (h)",   "min", "Horas de entrenamiento"),
        ],
        "recursos_sistema": [
            ("gpu_memory_used_mb",   "VRAM usada (MB)",     "min", "Memoria GPU utilizada"),
            ("gpu_memory_percent",   "VRAM %",              "min", "% VRAM utilizada"),
            ("ram_used_gb",          "RAM usada (GB)",      "min", "Memoria RAM utilizada"),
            ("ram_percent",          "RAM %",               "min", "% RAM utilizada"),
            ("cpu_percent",          "CPU %",               "min", "Utilización CPU"),
            ("gpu_temperature_c",    "GPU Temp (°C)",       "min", "Temperatura GPU"),
        ],
    },
    # ── F2 Clasificación ─────────────────────────────────────────────────────
    "cls": {
        "rendimiento": [
            ("best_val_accuracy",    "Best Val Acc",        "max", "Mejor accuracy de validación"),
            ("final_val_acc",        "Val Acc final",       "max", "Accuracy validación época final"),
            ("final_train_acc",      "Train Acc final",     "max", "Accuracy entrenamiento época final"),
            ("final_val_loss",       "Val Loss final",      "min", "Pérdida validación época final"),
            ("final_train_loss",     "Train Loss final",    "min", "Pérdida entrenamiento época final"),
            ("top1_accuracy",        "Top-1 Accuracy",      "max", "Accuracy top-1 (YOLO-cls)"),
            ("top5_accuracy",        "Top-5 Accuracy",      "max", "Accuracy top-5 (YOLO-cls)"),
        ],
        "recursos": [
            ("training_time_seconds","Tiempo train (s)",    "min", "Segundos totales de entrenamiento"),
            ("training_time_hours",  "Tiempo train (h)",    "min", "Horas totales de entrenamiento"),
        ],
    },
    # ── F3 VLM (fine-tuning LoRA) ────────────────────────────────────────────
    "vlm": {
        "rendimiento": [
            ("eval_loss",            "Eval Loss",           "min", "Pérdida en evaluación"),
            ("train_loss",           "Train Loss",          "min", "Pérdida en entrenamiento"),
            ("eval_rouge1",          "ROUGE-1",             "max", "ROUGE-1 F (unigrama)"),
            ("eval_rouge2",          "ROUGE-2",             "max", "ROUGE-2 F (bigrama)"),
            ("eval_rougeL",          "ROUGE-L",             "max", "ROUGE-L F (secuencia más larga)"),
            ("eval_rougeLsum",       "ROUGE-Lsum",          "max", "ROUGE-L sobre todo el resumen"),
            ("eval_bleu",            "BLEU",                "max", "BLEU score"),
            ("eval_meteor",          "METEOR",              "max", "METEOR score"),
            ("eval_bertscore",       "BERTScore",           "max", "BERTScore F1 semántico"),
            ("eval_perplexity",      "Perplexity",          "min", "Perplejidad del modelo"),
            ("eval_cider",           "CIDEr",               "max", "CIDEr (image captioning)"),
        ],
        "entrenamiento": [
            ("epoch",                "Épocas totales",      None,  "Épocas completadas"),
            ("global_step",          "Pasos totales",       None,  "Pasos de entrenamiento"),
            ("best_metric",          "Mejor métrica",       "max", "Mejor valor obtenido"),
            ("total_flos",           "FLOPs totales",       None,  "Operaciones de punto flotante"),
            ("train_runtime",        "Runtime train (s)",   "min", "Tiempo total entrenamiento"),
            ("train_samples_per_second", "Samples/s",       "max", "Throughput de entrenamiento"),
            ("train_steps_per_second",   "Steps/s",         "max", "Pasos por segundo"),
        ],
    },
    # ── F4 LLM (fine-tuning LoRA) ────────────────────────────────────────────
    "llm": {
        "rendimiento": [
            ("eval_loss",            "Eval Loss",           "min", "Pérdida en evaluación"),
            ("train_loss",           "Train Loss",          "min", "Pérdida en entrenamiento"),
            ("eval_rouge1",          "ROUGE-1",             "max", "ROUGE-1 F"),
            ("eval_rouge2",          "ROUGE-2",             "max", "ROUGE-2 F"),
            ("eval_rougeL",          "ROUGE-L",             "max", "ROUGE-L F"),
            ("eval_bleu",            "BLEU",                "max", "BLEU score"),
            ("eval_meteor",          "METEOR",              "max", "METEOR score"),
            ("eval_bertscore",       "BERTScore",           "max", "BERTScore F1"),
            ("eval_perplexity",      "Perplexity",          "min", "Perplejidad"),
            ("eval_mmlu",            "MMLU",                "max", "Massive Multitask LU"),
            ("eval_mauve",           "MAUVE",               "max", "MAUVE score"),
        ],
        "entrenamiento": [
            ("epoch",                "Épocas totales",      None,  "Épocas completadas"),
            ("global_step",          "Pasos totales",       None,  "Pasos de entrenamiento"),
            ("best_metric",          "Mejor métrica",       "max", "Mejor valor obtenido"),
            ("total_flos",           "FLOPs totales",       None,  "FLOPs"),
            ("train_runtime",        "Runtime train (s)",   "min", "Tiempo entrenamiento"),
            ("train_samples_per_second", "Samples/s",       "max", "Throughput"),
            ("train_steps_per_second",   "Steps/s",         "max", "Pasos/s"),
        ],
    },
}


def _scan_all_metrics(base_path: Path) -> dict:
    """Escanea recursivamente /workspace/models/<base_path> buscando
    todos los valores numéricos en cualquier .json que no sea un peso
    del modelo. Devuelve {campo: valor} con todo lo encontrado."""
    result = {}
    if not base_path.exists():
        return result
    skip_names = {"model", "tokenizer", "vocab", "merges", "added_tokens",
                  "special_tokens", "preprocessor", "processor",
                  "generation_config", "chat_template", "adapter_model"}
    try:
        for jf in base_path.rglob("*.json"):
            # Excluir archivos de pesos del modelo
            if any(s in jf.stem.lower() for s in skip_names):
                continue
            if "safetensors" in jf.name or "pytorch_model" in jf.name:
                continue
            data = _load_json(jf)
            if not isinstance(data, dict):
                # Puede ser lista (ej: all_results.json)
                if isinstance(data, list) and data:
                    data = data[0]  # tomar primer elemento
                else:
                    continue
            # Aplanar el dict hasta profundidad 3
            def _flatten(d, prefix="", depth=0):
                if depth > 3 or not isinstance(d, dict):
                    return
                for k, v in d.items():
                    full_k = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        result[full_k] = v
                    elif isinstance(v, dict):
                        _flatten(v, full_k, depth+1)
                    elif isinstance(v, str):
                        result[full_k] = v
            _flatten(data)
    except Exception:
        pass
    return result


def _render_extended_metrics(L, phase_key: str, raw_metrics: dict,
                              base_path: Path, BW: int = 40):
    """Renderiza en canvas la sección extendida de métricas para una fase.
    phase_key: 'det' | 'cls' | 'vlm' | 'llm'
    raw_metrics: dict plano con todos los valores encontrados en JSONs.
    """
    catalog = _METRICS_CATALOG.get(phase_key, {})

    # ── Métricas del catálogo por grupo ──────────────────────────────────────
    for group_name, entries in catalog.items():
        # Filtrar solo las que existen en los datos
        found = [(lbl, desc, direc, raw_metrics.get(key))
                 for key, lbl, direc, desc in entries
                 if raw_metrics.get(key) is not None]
        # También buscar variantes con prefijo (ej: metrics.mAP50)
        if not found:
            for key, lbl, direc, desc in entries:
                for rk, rv in raw_metrics.items():
                    if rk.endswith(f".{key}") or rk.endswith(f"/{key}"):
                        found.append((lbl, desc, direc, rv))
                        break
        if not found:
            continue

        group_lbl = {"rendimiento": "RENDIMIENTO DEL MODELO",
                     "recursos":    "RECURSOS DE CÓMPUTO",
                     "recursos_sistema": "SISTEMA (GPU/CPU/RAM)",
                     "entrenamiento": "PARÁMETROS DE ENTRENAMIENTO"}.get(group_name, group_name.upper())
        L(f"  ── {group_lbl} {'─'*(50-len(group_lbl))}")
        L(f"  {'Métrica':<28} {'Valor':>14}  {'Dir':>4}  Descripción")
        L("  " + "─"*80)
        for lbl, desc, direc, v in found:
            if isinstance(v, float):
                # Barra solo si v entre 0 y 1
                bar = ""
                if 0.0 <= v <= 1.0:
                    bar = "█" * max(1, int(v * BW // 4))
                dir_str = f"[{direc}]" if direc else "     "
                L(f"  {lbl:<28} {v:>14.6f}  {dir_str:<5}  {desc}  {bar}")
            elif isinstance(v, int):
                L(f"  {lbl:<28} {v:>14}  {'':5}  {desc}")
            elif isinstance(v, str):
                L(f"  {lbl:<28} {str(v)[:14]:>14}  {'':5}  {desc}")
        L("")

    # ── Campos adicionales descubiertos (no en catálogo) ─────────────────────
    catalogued_keys = {key for entries in catalog.values() for key, *_ in entries}
    extra = {k: v for k, v in raw_metrics.items()
             if k not in catalogued_keys
             and not any(k.endswith(f".{ck}") or k.endswith(f"/{ck}")
                         for ck in catalogued_keys)
             and isinstance(v, (int, float))
             and not isinstance(v, bool)}
    if extra:
        L(f"  ── CAMPOS ADICIONALES ENCONTRADOS EN /workspace/models/ ──────────")
        L(f"  {'Campo JSON':<38} {'Valor':>12}")
        L("  " + "─"*55)
        for k in sorted(extra.keys()):
            v = extra[k]
            if isinstance(v, float):
                L(f"  {k:<38} {v:>12.6f}")
            else:
                L(f"  {k:<38} {v:>12}")
        L("")


def _load_det_data():
    """Carga todos los JSONs y CSVs de detección. Devuelve dict con todo."""
    d = {}
    d["best"]    = _load_json(_DET_BASE / "best_combination.json") or {}
    d["grid"]    = _load_json(_DET_BASE / "grid_search_results.json") or []
    d["summary"] = _load_json(_DET_BASE / "summary.json") or {}
    d["val"]     = _load_json(_DET_BASE / "validation_metrics.json") or {}
    d["csvs"]    = {}
    for p in _DET_COMBOS:
        cp = p / "results.csv"
        if cp.exists():
            try:
                d["csvs"][p.name] = pd.read_csv(cp)
            except Exception:
                pass
    return d


class _DeteccionScreen(Screen):
    """
    Sub-pantalla de Detección dentro de ModelosScreen.
    Tres sub-tabs: [1]Detalle  [2]Métricas  [3]Gráficos
    """
    SUBTABS = ["[1]Detalle", "[2]Metricas", "[3]Graficos"]
    GRAPH_TYPES = [
        "Train Loss (box/cls/dfl)",
        "Val Loss (box/cls/dfl)",
        "mAP@50 — todos los combos",
        "mAP@50-95 — todos los combos",
        "Precision & Recall",
        "Learning Rate",
        "Comparativa final (barras)",
    ]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.sub      = 0       # 0=Detalle 1=Métricas 2=Gráficos
        self.scroll   = 0
        self.canvas   = []
        self.gi       = 0       # gráfico seleccionado
        self.combo_sel= 0       # combo seleccionado en gráficos (0=todos)
        self._data    = None
        self._built   = False

    def _data_ok(self):
        if self._data is None:
            self._data = _load_det_data()
        return self._data

    # ── builders de canvas ────────────────────────────────────────────────────
    def _build(self):
        self.canvas = []
        d = self._data_ok()
        if   self.sub == 0: self._build_detalle(d)
        elif self.sub == 1: self._build_metricas(d)
        elif self.sub == 2: self._build_graficos(d)
        self._built = True

    def _build_detalle(self, d):
        L = self.canvas.append
        best = d.get("best", {})
        summ = d.get("summary", {})
        val  = d.get("val", {})
        grid = d.get("grid", [])

        L("  ══════════════════════════════════════════════════════════════════")
        L("    DETALLE DE ENTRENAMIENTO — F1 DETECCIÓN  (YOLOv11s)")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")
        L(f"  Total combinaciones entrenadas : {summ.get('total_combinations', '—')}")
        L(f"  Exitosas / Fallidas            : {summ.get('successful','—')} / {summ.get('failed','—')}")
        L(f"  Mejor combinación              : combo_{best.get('combination_id','?'):03d}")
        L("")

        # ── Tabla resumen por combo ───────────────────────────────────────────
        L("  ── RESUMEN POR COMBINACIÓN ──────────────────────────────────────")
        L(f"  {'Combo':<10} {'Batch':>6} {'lr0':>8} {'Epochs':>7} "
          f"{'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8} "
          f"{'Time(h)':>9}")
        L("  " + "─"*80)
        for g in grid:
            hp  = g.get("hyperparameters", {})
            m   = g.get("metrics", {})
            cid = g.get("combination_id", "?")
            best_mark = " ★" if cid == best.get("combination_id") else "  "
            L(f"  combo_{cid:03d}{best_mark}  "
              f"{hp.get('batch','-'):>6}  "
              f"{hp.get('lr0',0):.4f}  "
              f"{hp.get('epochs','-'):>6}  "
              f"{m.get('mAP50',0):.5f}  "
              f"{m.get('mAP50-95',0):.5f}    "
              f"{m.get('precision',0):.5f}  "
              f"{m.get('recall',0):.5f}  "
              f"{g.get('training_time_hours',0):.4f}h")
        L("")

        # ── Hiperparámetros de TODOS los experimentos ─────────────────────────
        best_id = best.get("combination_id", 1)

        # Parámetros que varían entre combos (los mostramos primero destacados)
        # y los que son iguales en todos (se muestran como "comunes")
        HP_KEYS = [
            ("Model",           None,              "YOLOv11s"),
            ("imgsz",           "imgsz",           None),
            ("batch",           "batch",           None),
            ("epochs",          "epochs",          None),
            ("optimizer",       "optimizer",       None),
            ("lr0",             "lr0",             None),
            ("lrf",             "lrf",             None),
            ("momentum",        "momentum",        None),
            ("weight_decay",    "weight_decay",    None),
            ("warmup_epochs",   "warmup_epochs",   None),
            ("warmup_momentum", "warmup_momentum", None),
            ("warmup_bias_lr",  "warmup_bias_lr",  None),
            ("box loss w.",     "box",             None),
            ("cls loss w.",     "cls",             None),
            ("dfl loss w.",     "dfl",             None),
            ("close_mosaic",    "close_mosaic",    None),
            ("mosaic",          "mosaic",          None),
            ("amp",             "amp",             None),
            ("device",          "device",          None),
            ("workers",         "workers",         None),
            ("cache",           "cache",           None),
            ("patience",        "patience",        None),
            ("fraction",        "fraction",        None),
        ]

        # Determinar cuáles varían entre combos
        def hp_val(g, key, fixed):
            if fixed is not None: return fixed
            v = g.get("hyperparameters", {}).get(key, "—")
            return str(v) if not isinstance(v, list) else "/".join(str(x) for x in v)

        def varies(key, fixed):
            if fixed is not None: return False
            vals = {hp_val(g, key, fixed) for g in grid}
            return len(vals) > 1

        CW = 14   # ancho columna combo
        N  = len(grid)

        L("  ── HIPERPARÁMETROS — TODOS LOS EXPERIMENTOS ───────────────────")
        L(f"  {'Parámetro':<20}" +
          "".join(f" {'combo_'+str(g['combination_id'])+'★' if g['combination_id']==best_id else 'combo_'+str(g['combination_id']):<{CW}}"
                  for g in grid))
        L("  " + "─" * (20 + (CW+1)*N))

        # Primero los que varían (destacados), luego los iguales
        varying  = [(lbl,k,f) for lbl,k,f in HP_KEYS if varies(k,f)]
        constant = [(lbl,k,f) for lbl,k,f in HP_KEYS if not varies(k,f)]

        if varying:
            L("  ·· VARÍAN entre experimentos ·····························")
            for lbl, k, f in varying:
                row = f"  {lbl:<20}"
                for g in grid:
                    v = hp_val(g, k, f)
                    row += f" {v:<{CW}}"
                L(row)
            L("")

        L("  ·· COMUNES a todos los experimentos ························")
        for lbl, k, f in constant:
            row = f"  {lbl:<20}"
            for g in grid:
                v = hp_val(g, k, f)
                row += f" {v:<{CW}}"
            L(row)
        L("")

        # ── Recursos de cómputo ───────────────────────────────────────────────
        L("  ── RECURSOS DE CÓMPUTO ─────────────────────────────────────────")
        res = best.get("resources", {})
        L(f"  GPU           : {res.get('gpu_name','—')}")
        L(f"  VRAM usada    : {res.get('gpu_memory_used_mb',0):.0f} MB / "
          f"{res.get('gpu_memory_total_mb',0):.0f} MB  "
          f"({res.get('gpu_memory_percent',0):.2f}%)")
        L(f"  RAM usada     : {res.get('ram_used_gb',0):.2f} GB / "
          f"{res.get('ram_available_gb',0):.2f} GB  "
          f"({res.get('ram_percent',0):.1f}%)")
        L(f"  CPU cores     : {res.get('cpu_count','—')}  "
          f"CPU util: {res.get('cpu_percent',0):.1f}%")
        L(f"  GPU temp      : {res.get('gpu_temperature_c','—')} °C")
        L(f"  Tiempo train  : {best.get('training_time_seconds',0):.1f}s  "
          f"({best.get('training_time_hours',0):.4f}h)")
        L(f"  Timestamp     : {res.get('timestamp','—')}")
        L("")

        # ── Tiempos por combo ─────────────────────────────────────────────────
        L("  ── TIEMPOS DE ENTRENAMIENTO POR COMBO ───────────────────────────")
        L(f"  {'Combo':<12} {'Segundos':>12} {'Horas':>10} {'Batch':>8}")
        L("  " + "─"*46)
        for g in grid:
            hp2 = g.get("hyperparameters", {})
            L(f"  combo_{g['combination_id']:03d}    "
              f"{g.get('training_time_seconds',0):>12.2f}  "
              f"{g.get('training_time_hours',0):>10.4f}  "
              f"{hp2.get('batch','-'):>8}")
        L("")

        # ── Validación en test set ────────────────────────────────────────────
        vm = val.get("metrics", {})
        L("  ── VALIDACIÓN EN TEST SET (combo_001 — mejor modelo) ────────────")
        L(f"  mAP@50       : {vm.get('mAP50',0):.6f}")
        L(f"  mAP@50-95    : {vm.get('mAP50-95',0):.6f}")
        L(f"  Precision    : {vm.get('precision',0):.6f}")
        L(f"  Recall       : {vm.get('recall',0):.6f}")
        L(f"  Tiempo val   : {val.get('validation_time_seconds',0):.2f}s")
        vr = val.get("resources", {})
        L(f"  GPU mem val  : {vr.get('gpu_memory_used_mb',0):.0f} MB  "
          f"({vr.get('gpu_memory_percent',0):.2f}%)")

    def _build_metricas(self, d):
        L = self.canvas.append
        grid = d.get("grid", [])
        val  = d.get("val", {})
        vm   = val.get("metrics", {})
        best_id = d.get("best", {}).get("combination_id", 1)

        L("  ══════════════════════════════════════════════════════════════════")
        L("    COMPARATIVA DE MÉTRICAS — 4 COMBINACIONES  (F1 Detección)")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")

        # ── Tabla comparativa ─────────────────────────────────────────────────
        MKEYS  = ["mAP50","mAP50-95","precision","recall","box_loss","cls_loss","dfl_loss"]
        MLBLS  = ["mAP@50","mAP@50-95","Precision","Recall","box_loss","cls_loss","dfl_loss"]
        W = 12

        header = f"  {'Métrica':<14}" + "".join(f" {'combo_'+str(g['combination_id']):>{W}}" for g in grid)
        L(header)
        L("  " + "─" * (14 + (W+1)*len(grid)))

        for mk, ml in zip(MKEYS, MLBLS):
            vals = [g.get("metrics", {}).get(mk, 0) for g in grid]
            is_loss = "loss" in mk
            best_v = min(vals) if is_loss else max(vals)
            row = f"  {ml:<14}"
            for v in vals:
                mark = " ★" if abs(v - best_v) < 1e-9 else "  "
                row += f" {v:>{W-2}.5f}{mark}"
            L(row)
        L("")
        L("  ★ = mejor valor en esa métrica")
        L("")

        # ── Barras visuales mAP50 ────────────────────────────────────────────
        BW = 40
        L("  ── mAP@50 (barra visual) ─────────────────────────────────────────")
        max_m = max(g["metrics"]["mAP50"] for g in grid)
        for g in grid:
            v   = g["metrics"]["mAP50"]
            bar = "█" * max(1, int(v / max_m * BW))
            mk  = " ★" if g["combination_id"] == best_id else "  "
            L(f"  combo_{g['combination_id']:03d}{mk} │{bar:<{BW}}│ {v:.5f}")
        L("")

        # ── Barras visuales mAP50-95 ─────────────────────────────────────────
        L("  ── mAP@50-95 (barra visual) ──────────────────────────────────────")
        max_m95 = max(g["metrics"]["mAP50-95"] for g in grid)
        for g in grid:
            v   = g["metrics"]["mAP50-95"]
            bar = "█" * max(1, int(v / max_m95 * BW))
            mk  = " ★" if g["combination_id"] == best_id else "  "
            L(f"  combo_{g['combination_id']:03d}{mk} │{bar:<{BW}}│ {v:.5f}")
        L("")

        # ── Diferencias respecto al mejor ────────────────────────────────────
        L("  ── DIFERENCIAS vs MEJOR (combo_001) ──────────────────────────────")
        best_m = {mk: d["best"].get("metrics",{}).get(mk,0) for mk in MKEYS}
        L(f"  {'Métrica':<14}" + "".join(f"  {'Δcombo_'+str(g['combination_id']):>12}" for g in grid[1:]))
        L("  " + "─" * (14 + 14*(len(grid)-1)))
        for mk, ml in zip(MKEYS, MLBLS):
            row = f"  {ml:<14}"
            for g in grid[1:]:
                delta = g["metrics"].get(mk, 0) - best_m[mk]
                sign  = "+" if delta >= 0 else ""
                row  += f"  {sign}{delta:>10.5f}  "
            L(row)
        L("")

        # ── Validación test set ───────────────────────────────────────────────
        L("  ── VALIDACIÓN TEST SET (mejor modelo — combo_001) ──────────────")
        L(f"  mAP@50       : {vm.get('mAP50',0):.6f}   "
          f"(train: {d['best'].get('metrics',{}).get('mAP50',0):.6f}  "
          f"Δ={vm.get('mAP50',0)-d['best'].get('metrics',{}).get('mAP50',0):+.6f})")
        L(f"  mAP@50-95    : {vm.get('mAP50-95',0):.6f}   "
          f"(train: {d['best'].get('metrics',{}).get('mAP50-95',0):.6f}  "
          f"Δ={vm.get('mAP50-95',0)-d['best'].get('metrics',{}).get('mAP50-95',0):+.6f})")
        L(f"  Precision    : {vm.get('precision',0):.6f}   "
          f"(train: {d['best'].get('metrics',{}).get('precision',0):.6f}  "
          f"Δ={vm.get('precision',0)-d['best'].get('metrics',{}).get('precision',0):+.6f})")
        L(f"  Recall       : {vm.get('recall',0):.6f}   "
          f"(train: {d['best'].get('metrics',{}).get('recall',0):.6f}  "
          f"Δ={vm.get('recall',0)-d['best'].get('metrics',{}).get('recall',0):+.6f})")
        L("")

        # ── Métricas extendidas escaneadas desde /workspace/models/ ───────────
        L("  ══ ANÁLISIS EXTENDIDO — TODOS LOS CAMPOS EN /workspace/models/ ══")
        L("")
        raw = _scan_all_metrics(_DET_BASE)
        _render_extended_metrics(L, "det", raw, _DET_BASE)

    def _build_graficos(self, d):
        L = self.canvas.append
        csvs = d.get("csvs", {})
        if not csvs:
            L("  Sin CSVs de entrenamiento disponibles.")
            return

        gt = self.GRAPH_TYPES[self.gi]
        combos_sorted = sorted(csvs.keys())
        COLORS_LBL = ["[1]","[2]","[3]","[4]"]
        BAR = "█"
        BW  = 35

        # Selección de combos: 0=todos, 1-4=individual
        if self.combo_sel == 0:
            sel_combos = combos_sorted
        else:
            name = f"combo_00{self.combo_sel}"
            sel_combos = [name] if name in csvs else combos_sorted

        L(f"  ══ {gt.upper()} ══")
        L(f"  Mostrando: {'TODOS' if self.combo_sel==0 else sel_combos[0] if sel_combos else '—'}")
        L("")

        def get_col(df, name):
            for c in df.columns:
                if c.strip() == name:
                    return pd.to_numeric(df[c], errors="coerce").values
            return None

        def mini_chart(vals, label, W=BW, H=6, color_pfx=""):
            """Genera línea de texto ASCII de una serie."""
            lines = []
            mn = float(np.nanmin(vals)); mx = float(np.nanmax(vals))
            rng = mx - mn if mx != mn else 1e-9
            cts = [0]*W
            for i, x in enumerate(vals):
                xi = min(int(i / len(vals) * W), W-1)
                yi = int((x - mn) / rng * (H-1))
                cts[xi] = max(cts[xi], yi)
            lines.append(f"  {label:<22}  {mn:.4f}──{'─'*W}──{mx:.4f}")
            for ri in range(H, 0, -1):
                row = "".join(BAR if ct >= ri else " " for ct in cts)
                lines.append(f"  {'':22}  {'':8} │{row}│")
            lines.append(f"  {'':22}  {'':8}  ep1{'─'*(W-6)}ep{len(vals)}")
            return lines

        # ── Gráfico 0: Train Loss ─────────────────────────────────────────────
        if self.gi == 0:
            for name in sel_combos:
                df = csvs[name]
                L(f"  ── {name} ───────────────────────────────────────────────────")
                for col_n, lbl in [("train/box_loss","box_loss"),
                                   ("train/cls_loss","cls_loss"),
                                   ("train/dfl_loss","dfl_loss")]:
                    v = get_col(df, col_n)
                    if v is not None:
                        for ln in mini_chart(v, lbl, H=4):
                            L(ln)
                        L(f"  {'':22}  final={v[-1]:.5f}  min={np.nanmin(v):.5f}")
                L("")

        # ── Gráfico 1: Val Loss ───────────────────────────────────────────────
        elif self.gi == 1:
            for name in sel_combos:
                df = csvs[name]
                L(f"  ── {name} ───────────────────────────────────────────────────")
                for col_n, lbl in [("val/box_loss","val box_loss"),
                                   ("val/cls_loss","val cls_loss"),
                                   ("val/dfl_loss","val dfl_loss")]:
                    v = get_col(df, col_n)
                    if v is not None:
                        for ln in mini_chart(v, lbl, H=4):
                            L(ln)
                        L(f"  {'':22}  final={v[-1]:.5f}  min={np.nanmin(v):.5f}")
                L("")

        # ── Gráfico 2: mAP@50 todos ───────────────────────────────────────────
        elif self.gi == 2:
            L(f"  {'Combo':<12}  {'ep1':>8}  {'ep10':>8}  {'ep20':>8}  {'ep30':>8}  {'max':>8}  Curva")
            L("  " + "─"*75)
            for name in combos_sorted:
                df = csvs.get(name)
                if df is None: continue
                v = get_col(df, "metrics/mAP50(B)")
                if v is None: continue
                mx_v = np.nanmax(v)
                bar  = BAR * max(1, int(mx_v * BW))
                ep_vals = [v[0], v[min(9,len(v)-1)], v[min(19,len(v)-1)], v[-1]]
                ep_str  = "  ".join(f"{x:.4f}" for x in ep_vals)
                best_m  = "★" if name == f"combo_00{d['best'].get('combination_id',1)}" else " "
                L(f"  {name}{best_m}  {ep_str}  {mx_v:.4f}  │{bar:<{BW}}│")
            L("")
            # Curva por época de cada combo
            for name in combos_sorted:
                df = csvs.get(name)
                if df is None: continue
                v = get_col(df, "metrics/mAP50(B)")
                if v is None: continue
                for ln in mini_chart(v, name, H=5):
                    L(ln)
                L("")

        # ── Gráfico 3: mAP@50-95 todos ────────────────────────────────────────
        elif self.gi == 3:
            L(f"  {'Combo':<12}  {'ep1':>8}  {'ep10':>8}  {'ep20':>8}  {'ep30':>8}  {'max':>8}  Curva")
            L("  " + "─"*75)
            for name in combos_sorted:
                df = csvs.get(name)
                if df is None: continue
                v = get_col(df, "metrics/mAP50-95(B)")
                if v is None: continue
                mx_v = np.nanmax(v)
                bar  = BAR * max(1, int(mx_v * BW))
                ep_vals = [v[0], v[min(9,len(v)-1)], v[min(19,len(v)-1)], v[-1]]
                ep_str  = "  ".join(f"{x:.4f}" for x in ep_vals)
                best_m  = "★" if name == f"combo_00{d['best'].get('combination_id',1)}" else " "
                L(f"  {name}{best_m}  {ep_str}  {mx_v:.4f}  │{bar:<{BW}}│")
            L("")
            for name in combos_sorted:
                df = csvs.get(name)
                if df is None: continue
                v = get_col(df, "metrics/mAP50-95(B)")
                if v is None: continue
                for ln in mini_chart(v, name, H=5):
                    L(ln)
                L("")

        # ── Gráfico 4: Precision & Recall ─────────────────────────────────────
        elif self.gi == 4:
            for name in sel_combos:
                df = csvs.get(name)
                if df is None: continue
                L(f"  ── {name} ───────────────────────────────────────────────────")
                vp = get_col(df, "metrics/precision(B)")
                vr = get_col(df, "metrics/recall(B)")
                if vp is not None:
                    for ln in mini_chart(vp, "Precision", H=5):
                        L(ln)
                    L(f"  {'':22}  final={vp[-1]:.5f}  max={np.nanmax(vp):.5f}")
                if vr is not None:
                    for ln in mini_chart(vr, "Recall", H=5):
                        L(ln)
                    L(f"  {'':22}  final={vr[-1]:.5f}  max={np.nanmax(vr):.5f}")
                # F1 derivado
                if vp is not None and vr is not None:
                    f1 = 2*vp*vr/(vp+vr+1e-9)
                    for ln in mini_chart(f1, "F1 (deriv.)", H=5):
                        L(ln)
                    L(f"  {'':22}  final={f1[-1]:.5f}  max={np.nanmax(f1):.5f}")
                L("")

        # ── Gráfico 5: Learning Rate ───────────────────────────────────────────
        elif self.gi == 5:
            for name in sel_combos:
                df = csvs.get(name)
                if df is None: continue
                L(f"  ── {name} ───────────────────────────────────────────────────")
                for col_n, lbl in [("lr/pg0","LR pg0 (backbone)"),
                                   ("lr/pg1","LR pg1 (head bias)"),
                                   ("lr/pg2","LR pg2 (head wts)")]:
                    v = get_col(df, col_n)
                    if v is not None:
                        mn_v = np.nanmin(v); mx_v = np.nanmax(v)
                        bar_w = 50
                        row = ""
                        for x in v:
                            xi = int((x - mn_v) / (mx_v - mn_v + 1e-9) * bar_w)
                            row += "▪" if xi > bar_w//2 else "·"
                        L(f"  {lbl:<22}  max={mx_v:.2e}  min={mn_v:.2e}")
                        L(f"  {'':22}  │{row}│")
                L("")

        # ── Gráfico 6: Comparativa final barras ────────────────────────────────
        elif self.gi == 6:
            grid_d = d.get("grid", [])
            val_d  = d.get("val", {}).get("metrics", {})
            MLIST  = [
                ("mAP50",    "mAP@50      "),
                ("mAP50-95", "mAP@50-95   "),
                ("precision","Precision   "),
                ("recall",   "Recall      "),
            ]
            for mk, ml in MLIST:
                L(f"  ── {ml.strip()} ─────────────────────────────────────────────")
                vals = [g["metrics"].get(mk, 0) for g in grid_d]
                max_v = max(vals) if vals else 1
                for g, v in zip(grid_d, vals):
                    bar  = BAR * max(1, int(v / max_v * BW))
                    best_m = "★" if g["combination_id"] == d["best"].get("combination_id",1) else " "
                    L(f"  combo_{g['combination_id']:03d}{best_m} │{bar:<{BW}}│ {v:.5f}")
                # línea de validación test
                vv = val_d.get(mk, None)
                if vv:
                    bar2 = "░" * max(1, int(vv / max_v * BW))
                    L(f"  test_set   │{bar2:<{BW}}│ {vv:.5f}  (test set)")
                L("")

    # ── draw ──────────────────────────────────────────────────────────────────
    def draw_body(self):
        self._sz()
        rs = 3; PW = 26
        self.statusbar()

        # Subtabs header
        for i, t in enumerate(self.SUBTABS):
            at = curses.color_pair(cS)|curses.A_BOLD if i == self.sub else dim()
            sa(self.scr, rs, 2 + i*16, f" {t} ", at)

        if self.sub == 2:
            # Panel derecho: lista de gráficos
            hl(self.scr, rs, self.w-PW-2, PW+1)
            sa(self.scr, rs, self.w-PW-1, " GRAFICOS ", curses.color_pair(cH)|curses.A_BOLD)
            for i, g in enumerate(self.GRAPH_TYPES):
                rw = rs+1+i
                if rw >= self.h-4: break
                at = curses.color_pair(cS)|curses.A_BOLD if i == self.gi else dim()
                sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g, PW-4)}", at)
            # Selector de combo
            sep = rs + len(self.GRAPH_TYPES) + 2
            sa(self.scr, sep, self.w-PW-1,
               f" Combo: {'Todos' if self.combo_sel==0 else f'combo_00{self.combo_sel}'}",
               curses.color_pair(cMT))
            sa(self.scr, sep+1, self.w-PW-1, " [0]Todos [1-4]Combo", dim())

        DW = (self.w - PW - 3) if self.sub == 2 else self.w - 3

        hl(self.scr, rs+1, 0, DW)
        gt_lbl = self.GRAPH_TYPES[self.gi] if self.sub == 2 else self.SUBTABS[self.sub]
        sa(self.scr, rs+1, 2, f" {gt_lbl} ", curses.color_pair(cGR)|curses.A_BOLD)

        if not self._built or not self.canvas:
            self._build()

        vis = self.h - rs - 6
        ms  = max(0, len(self.canvas) - vis)
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(vis):
            idx = self.scroll + i
            if idx >= len(self.canvas): break
            ln  = self.canvas[idx]
            if   "★" in ln:                         at = curses.color_pair(cMX)|curses.A_BOLD
            elif "══" in ln or "──" in ln:           at = curses.color_pair(cP)|curses.A_BOLD
            elif "│" in ln and "█" in ln:            at = curses.color_pair(cGR)
            elif "│" in ln and "░" in ln:            at = curses.color_pair(cMT)
            elif ln.strip().startswith("combo_001"): at = curses.color_pair(cOK)
            elif "Δ" in ln and "+" in ln:            at = curses.color_pair(cER)
            elif "Δ" in ln and "-" in ln:            at = curses.color_pair(cOK)
            else:                                    at = curses.color_pair(cN)
            sa(self.scr, rs+2+i, 1, tr(ln, DW-2), at)

        if self.sub == 2:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Scroll  [←→]Gráfico  [0-4]Combo  [1-7]Gráfico directo "
                     " [Tab]Sig. pestaña  [q]Salir ")
        else:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Scroll  [1/2/3]Sub-tab  [Tab]Sig. pestaña  [q]Salir ")

    def key(self, k):
        if k in (ord('1'),) and self.sub != 0:
            self.sub=0; self._built=False; self.canvas=[]; self.scroll=0
        elif k in (ord('2'),) and self.sub != 1:
            self.sub=1; self._built=False; self.canvas=[]; self.scroll=0
        elif k in (ord('3'),) and self.sub != 2:
            self.sub=2; self._built=False; self.canvas=[]; self.scroll=0
        elif self.sub == 2:
            if k == curses.KEY_LEFT:
                self.gi = (self.gi-1) % len(self.GRAPH_TYPES)
                self._built=False; self.canvas=[]; self.scroll=0
            elif k == curses.KEY_RIGHT:
                self.gi = (self.gi+1) % len(self.GRAPH_TYPES)
                self._built=False; self.canvas=[]; self.scroll=0
            elif k == curses.KEY_UP:   self.scroll = max(0, self.scroll-1)
            elif k == curses.KEY_DOWN: self.scroll += 1
            elif ord('0') <= k <= ord('4'):
                self.combo_sel = k - ord('0')
                self._built=False; self.canvas=[]; self.scroll=0
            elif ord('1') <= k <= ord('7'):
                self.gi = k - ord('1')
                self._built=False; self.canvas=[]; self.scroll=0
        else:
            if k == curses.KEY_UP:   self.scroll = max(0, self.scroll-1)
            elif k == curses.KEY_DOWN: self.scroll += 1
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 11c  PANTALLA 6b — CLASIFICACIÓN (F2)
#   5 sub-modelos: inceptionv3 | production_cls | resnet50 | swin | vit
#   Cada sub-modelo → [1]Detalle  [2]Métricas  [3]Gráficos
# ══════════════════════════════════════════════════════════════════════════════

# Definición de cada sub-modelo de clasificación
_CLS_MODELS = [
    {
        "key":   "inceptionv3",
        "label": "InceptionV3",
        "base":  Path("/workspace/models/phase2_classification/inceptionv3/production_inception"),
        "metric_key": "best_val_accuracy",
        "metric_lbl": "Best Val Accuracy",
        "history_keys": ["train_loss","train_acc","val_loss","val_acc"],
        "has_csv": False,   # usa history.json
    },
    {
        "key":   "production_cls",
        "label": "YOLOv11s-cls",
        "base":  Path("/workspace/models/phase2_classification/production_cls"),
        "metric_key": "top1_accuracy",
        "metric_lbl": "Top-1 Accuracy",
        "history_keys": None,  # tiene results.csv
        "has_csv": True,
    },
    {
        "key":   "resnet50",
        "label": "ResNet50",
        "base":  Path("/workspace/models/phase2_classification/resnet50/production_resnet"),
        "metric_key": "best_val_accuracy",
        "metric_lbl": "Best Val Accuracy",
        "history_keys": ["train_loss","train_acc","val_loss","val_acc"],
        "has_csv": False,
    },
    {
        "key":   "swin",
        "label": "Swin Transformer",
        "base":  Path("/workspace/models/phase2_classification/swin/production_swin"),
        "metric_key": "best_val_accuracy",
        "metric_lbl": "Best Val Accuracy",
        "history_keys": ["train_loss","train_acc","val_loss","val_acc"],
        "has_csv": False,
    },
    {
        "key":   "vit",
        "label": "Vision Transformer",
        "base":  Path("/workspace/models/phase2_classification/vit/production_vit"),
        "metric_key": "best_val_accuracy",
        "metric_lbl": "Best Val Accuracy",
        "history_keys": ["train_loss","train_acc","val_loss","val_acc"],
        "has_csv": False,
    },
]


def _load_cls_data(mdl: dict) -> dict:
    """Carga all_results.json, best_combination.json y validation_metrics.json
    para un sub-modelo de clasificación. Devuelve dict unificado."""
    base = mdl["base"]
    d = {}
    d["all"]  = _load_json(base / "all_results.json") or []
    d["best"] = _load_json(base / "best_combination.json") or {}
    d["val"]  = _load_json(base / "validation_metrics.json") or {}
    d["mdl"]  = mdl
    # Cargar históricos por combo
    d["histories"] = {}
    for combo in d["all"]:
        cid  = combo.get("combination_id", 0)
        cdir = base / f"combo_{cid:03d}"
        if mdl["has_csv"]:
            cp = cdir / "results.csv"
            if cp.exists():
                try:
                    d["histories"][f"combo_{cid:03d}"] = pd.read_csv(cp)
                except Exception:
                    pass
        else:
            hj = cdir / "history.json"
            if hj.exists():
                d["histories"][f"combo_{cid:03d}"] = _load_json(hj) or {}
    return d


class _ClasificacionScreen(Screen):
    """
    Sub-pantalla de Clasificación dentro de ModelosScreen.
    Panel izquierdo: 5 sub-modelos navegables con [↑↓].
    Sub-tabs: [1]Detalle  [2]Métricas  [3]Gráficos
    """
    SUBTABS    = ["[1]Detalle", "[2]Metricas", "[3]Graficos"]
    GRAPH_TYPES = [
        "Train Loss / Val Loss",
        "Train Acc / Val Acc",
        "Best Val Acc — todos los combos",
        "Comparativa final (barras)",
        "Learning Rate (YOLOv11s-cls)",
    ]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.mdl_idx  = 0       # índice en _CLS_MODELS
        self.sub      = 0       # 0=Detalle 1=Métricas 2=Gráficos
        self.scroll   = 0
        self.canvas   = []
        self.gi       = 0
        self.combo_sel= 0       # 0=todos, 1-4=individual
        self._cache   = {}      # {mdl_key: data}
        self._built   = False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _mdl(self):
        return _CLS_MODELS[self.mdl_idx]

    def _data(self):
        key = self._mdl()["key"]
        if key not in self._cache:
            self._cache[key] = _load_cls_data(self._mdl())
        return self._cache[key]

    def _reset(self):
        self.canvas = []; self.scroll = 0; self._built = False

    # ── builders ──────────────────────────────────────────────────────────────
    def _build(self):
        self.canvas = []
        d = self._data()
        if   self.sub == 0: self._build_detalle(d)
        elif self.sub == 1: self._build_metricas(d)
        elif self.sub == 2: self._build_graficos(d)
        self._built = True

    def _build_detalle(self, d):
        L = self.canvas.append
        mdl  = d["mdl"]
        best = d["best"]
        grid = d["all"]
        val  = d["val"]
        best_id = best.get("combination_id", best.get("best_combination_id", 1))

        L("  ══════════════════════════════════════════════════════════════════")
        L(f"    DETALLE — F2 CLASIFICACIÓN  ({mdl['label']})")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")
        L(f"  Sub-modelo   : {mdl['label']}")
        L(f"  Ruta base    : {mdl['base']}")
        L(f"  Experimentos : {len(grid)}")
        L(f"  Mejor combo  : combo_{best_id:03d}" if isinstance(best_id,int)
          else f"  Mejor combo  : combo_{best_id}")
        L("")

        # ── Tabla resumen ─────────────────────────────────────────────────────
        mk  = mdl["metric_key"]
        ml  = mdl["metric_lbl"]
        L("  ── RESUMEN POR COMBINACIÓN ──────────────────────────────────────")
        L(f"  {'Combo':<12} {'Batch':>6} {'lr0':>8} {'Epochs':>7} "
          f"{ml[:14]:>14} {'Time(h)':>9}")
        L("  " + "─"*62)
        for g in grid:
            hp  = g.get("hyperparameters", {})
            m   = g.get("metrics", {})
            cid = g.get("combination_id", "?")
            bm  = " ★" if cid == best_id else "  "
            acc = m.get(mk, m.get("top1_accuracy", 0))
            L(f"  combo_{cid:03d}{bm}  "
              f"{hp.get('batch','-'):>6}  "
              f"{hp.get('lr0',0):.5f}  "
              f"{hp.get('epochs','-'):>6}  "
              f"{acc:>14.6f}  "
              f"{g.get('training_time_hours',0):>9.4f}h")
        L("")

        # ── Hiperparámetros — tabla comparativa automática ────────────────────
        if grid:
            all_hkeys = list(grid[0].get("hyperparameters", {}).keys())
            def hp_v(g, k):
                v = g.get("hyperparameters",{}).get(k,"—")
                return str(v) if not isinstance(v, list) else "/".join(str(x) for x in v)
            def hpvaries(k):
                vals = {hp_v(g, k) for g in grid}
                return len(vals) > 1

            CW = 14; N = len(grid)
            L("  ── HIPERPARÁMETROS — TODOS LOS EXPERIMENTOS ────────────────────")
            L(f"  {'Parámetro':<20}" +
              "".join(f" {'combo_'+str(g['combination_id'])+'★' if g['combination_id']==best_id else 'combo_'+str(g['combination_id']):<{CW}}"
                      for g in grid))
            L("  " + "─" * (20 + (CW+1)*N))

            varying  = [k for k in all_hkeys if hpvaries(k)]
            constant = [k for k in all_hkeys if not hpvaries(k)]

            if varying:
                L("  ·· VARÍAN entre experimentos ·····························")
                for k in varying:
                    row = f"  {k:<20}"
                    for g in grid:
                        row += f" {hp_v(g,k):<{CW}}"
                    L(row)
                L("")

            L("  ·· COMUNES a todos los experimentos ························")
            for k in constant:
                row = f"  {k:<20}"
                for g in grid:
                    row += f" {hp_v(g,k):<{CW}}"
                L(row)
            L("")

        # ── Recursos de cómputo ───────────────────────────────────────────────
        L("  ── RECURSOS DE CÓMPUTO ─────────────────────────────────────────")
        res = best.get("resources", {})
        gpus = res.get("gpus", [])
        if gpus:
            for gi2, gpu in enumerate(gpus):
                L(f"  GPU {gi2}         : {gpu.get('name','—')}  "
                  f"{gpu.get('memory_used_mb',0):.0f}/{gpu.get('memory_total_mb',0):.0f} MB")
        else:
            L(f"  GPU           : {res.get('gpu_name','—')}")
        L(f"  RAM usada     : {res.get('ram_used_gb',0):.2f} GB  ({res.get('ram_percent',0):.1f}%)")
        L(f"  CPU util      : {res.get('cpu_percent',0):.1f}%")
        L(f"  Timestamp     : {res.get('timestamp','—')}")
        L("")

        # ── Tiempos ───────────────────────────────────────────────────────────
        L("  ── TIEMPOS DE ENTRENAMIENTO POR COMBO ───────────────────────────")
        L(f"  {'Combo':<12} {'Segundos':>12} {'Horas':>10} {'Batch':>8}")
        L("  " + "─"*46)
        for g in grid:
            hp2 = g.get("hyperparameters", {})
            L(f"  combo_{g['combination_id']:03d}    "
              f"{g.get('training_time_seconds',0):>12.2f}  "
              f"{g.get('training_time_hours',0):>10.4f}  "
              f"{hp2.get('batch','-'):>8}")
        L("")

        # ── Validación ────────────────────────────────────────────────────────
        if val:
            vm = val.get("metrics", {})
            L("  ── VALIDACIÓN EN TEST SET ───────────────────────────────────────")
            for k2, v2 in vm.items():
                if isinstance(v2, (int, float)):
                    L(f"  {k2:<22} : {v2:.6f}")
            vt = val.get("validation_time_seconds", val.get("validation_time", None))
            if vt:
                L(f"  Tiempo val          : {vt:.2f}s")
            vres = val.get("resources", {})
            if vres.get("gpus"):
                for gi2, gpu in enumerate(vres["gpus"]):
                    L(f"  GPU {gi2} val mem     : "
                      f"{gpu.get('memory_used_mb',0):.0f} MB")

    def _build_metricas(self, d):
        L = self.canvas.append
        mdl  = d["mdl"]
        grid = d["all"]
        val  = d["val"]
        mk   = mdl["metric_key"]
        ml   = mdl["metric_lbl"]
        best_id = d["best"].get("combination_id",
                  d["best"].get("best_combination_id", 1))
        BW = 38

        L("  ══════════════════════════════════════════════════════════════════")
        L(f"    MÉTRICAS — F2 CLASIFICACIÓN  ({mdl['label']})")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")

        # ── Tabla comparativa ─────────────────────────────────────────────────
        all_mkeys = []
        if grid:
            all_mkeys = [k for k, v in grid[0].get("metrics",{}).items()
                         if isinstance(v, (int, float))]

        CW = 12; N = len(grid)
        L(f"  {'Métrica':<22}" +
          "".join(f" {'combo_'+str(g['combination_id']):>{CW}}" for g in grid))
        L("  " + "─"*(22+(CW+1)*N))
        for k2 in all_mkeys:
            vals   = [g.get("metrics",{}).get(k2, 0) for g in grid]
            is_loss = "loss" in k2.lower()
            best_v  = min(vals) if is_loss else max(vals)
            row = f"  {k2:<22}"
            for v in vals:
                mk2 = " ★" if abs(v-best_v) < 1e-9 else "  "
                row += f" {v:>{CW-2}.5f}{mk2}"
            L(row)
        L("")
        L("  ★ = mejor valor")
        L("")

        # ── Barras visuales accuracy ──────────────────────────────────────────
        acc_vals = [g.get("metrics",{}).get(mk,
                    g.get("metrics",{}).get("top1_accuracy",0)) for g in grid]
        if acc_vals:
            max_a = max(acc_vals) if max(acc_vals) > 0 else 1
            L(f"  ── {ml} (barra visual) ─────────────────────────────────────")
            for g, v in zip(grid, acc_vals):
                bar  = "█" * max(1, int(v/max_a*BW))
                bm   = " ★" if g["combination_id"] == best_id else "  "
                L(f"  combo_{g['combination_id']:03d}{bm} │{bar:<{BW}}│ {v:.5f}")
            L("")

        # ── Métricas finales de entrenamiento ─────────────────────────────────
        fin_keys = ["final_train_loss","final_val_loss",
                    "final_train_acc","final_val_acc","best_val_accuracy",
                    "top1_accuracy","top5_accuracy"]
        L("  ── MÉTRICAS FINALES POR COMBO ────────────────────────────────────")
        L(f"  {'Métrica':<24}" +
          "".join(f" {'combo_'+str(g['combination_id']):>12}" for g in grid))
        L("  " + "─"*(24+13*N))
        for fk in fin_keys:
            row_vals = [g.get("metrics",{}).get(fk, None) for g in grid]
            if any(v is not None for v in row_vals):
                row = f"  {fk:<24}"
                for v in row_vals:
                    row += f" {v:>12.6f}" if v is not None else f" {'—':>12}"
                L(row)
        L("")

        # ── Validación en test set ────────────────────────────────────────────
        if val:
            vm = val.get("metrics", {})
            bid = val.get("best_combination_id", best_id)
            L(f"  ── VALIDACIÓN TEST SET (mejor: combo_{bid:03d}) ─────────────────")
            for k2, v2 in vm.items():
                if isinstance(v2, (int, float)):
                    best_train = d["best"].get("metrics",{}).get(k2, None)
                    if best_train is not None:
                        L(f"  {k2:<22} : {v2:.6f}  "
                          f"(train: {best_train:.6f}  Δ={v2-best_train:+.6f})")
                    else:
                        L(f"  {k2:<22} : {v2:.6f}")

        # ── Métricas extendidas escaneadas desde /workspace/models/ ───────────
        L("  ══ ANÁLISIS EXTENDIDO — TODOS LOS CAMPOS EN /workspace/models/ ══")
        L("")
        raw = _scan_all_metrics(d["mdl"]["base"])
        _render_extended_metrics(L, "cls", raw, d["mdl"]["base"])

    def _build_graficos(self, d):
        L = self.canvas.append
        mdl  = d["mdl"]
        grid = d["all"]
        hist = d["histories"]
        best_id = d["best"].get("combination_id",
                  d["best"].get("best_combination_id", 1))
        BW = 38; BAR = "█"

        combos_sorted = sorted(hist.keys())
        sel_combos = (combos_sorted if self.combo_sel == 0
                      else [f"combo_00{self.combo_sel}"]
                      if f"combo_00{self.combo_sel}" in hist else combos_sorted)

        gt = self.GRAPH_TYPES[self.gi]
        L(f"  ══ {gt.upper()} — {mdl['label']} ══")
        L(f"  Mostrando: {'TODOS' if self.combo_sel==0 else sel_combos[0] if sel_combos else '—'}")
        L("")

        def mini_chart(vals, label, W=BW, H=5):
            lines = []
            if not vals: return [f"  {label}: sin datos"]
            mn = float(min(vals)); mx = float(max(vals))
            rng = mx-mn if mx!=mn else 1e-9
            cts = [0]*W
            for i, x in enumerate(vals):
                xi = min(int(i/len(vals)*W), W-1)
                yi = int((x-mn)/rng*(H-1))
                cts[xi] = max(cts[xi], yi)
            lines.append(f"  {label:<22}  {mn:.4f}──{'─'*W}──{mx:.4f}")
            for ri in range(H, 0, -1):
                row = "".join(BAR if ct >= ri else " " for ct in cts)
                lines.append(f"  {'':22}  {'':8} │{row}│")
            lines.append(f"  {'':22}  ep1{'─'*(W-6)}ep{len(vals)}")
            return lines

        def get_series(combo_name, series_key):
            """Extrae una serie del historial (JSON dict o CSV DataFrame)."""
            h = hist.get(combo_name)
            if h is None: return []
            if isinstance(h, dict):
                return h.get(series_key, [])
            # DataFrame (CSV) — mapeo de nombres
            col_map = {
                "train_loss":    "train/loss",
                "val_loss":      "val/loss",
                "train_acc":     "metrics/accuracy_top1",
                "val_acc":       "metrics/accuracy_top1",
                "lr":            "lr/pg0",
            }
            col = col_map.get(series_key, series_key)
            for c in h.columns:
                if c.strip() == col:
                    return pd.to_numeric(h[c], errors="coerce").dropna().tolist()
            return []

        # ── Gráfico 0: Train Loss / Val Loss ─────────────────────────────────
        if self.gi == 0:
            for name in sel_combos:
                L(f"  ── {name} ──────────────────────────────────────────────────")
                for sk, sl in [("train_loss","Train Loss"),("val_loss","Val Loss")]:
                    v = get_series(name, sk)
                    if v:
                        for ln in mini_chart(v, sl): L(ln)
                        L(f"  {'':22}  final={v[-1]:.5f}  min={min(v):.5f}")
                L("")

        # ── Gráfico 1: Train Acc / Val Acc ───────────────────────────────────
        elif self.gi == 1:
            for name in sel_combos:
                L(f"  ── {name} ──────────────────────────────────────────────────")
                for sk, sl in [("train_acc","Train Acc"),("val_acc","Val Acc")]:
                    v = get_series(name, sk)
                    if v:
                        for ln in mini_chart(v, sl): L(ln)
                        L(f"  {'':22}  final={v[-1]:.5f}  max={max(v):.5f}")
                L("")

        # ── Gráfico 2: Best Val Acc todos los combos ─────────────────────────
        elif self.gi == 2:
            mk   = mdl["metric_key"]
            ml   = mdl["metric_lbl"]
            vals = []
            for g in grid:
                v = g.get("metrics",{}).get(mk, g.get("metrics",{}).get("top1_accuracy",0))
                vals.append((g["combination_id"], v))
            max_v = max(v for _,v in vals) if vals else 1
            L(f"  {'Combo':<12}  {ml:<18}  Barra")
            L("  " + "─"*60)
            for cid, v in sorted(vals):
                bar  = BAR * max(1, int(v/max_v*BW))
                bm   = " ★" if cid == best_id else "  "
                L(f"  combo_{cid:03d}{bm}  {v:>18.6f}  │{bar:<{BW}}│")
            L("")
            # Curvas por combo
            for name in combos_sorted:
                g = next((g for g in grid if f"combo_{g['combination_id']:03d}" == name), None)
                v_hist = get_series(name, "val_acc")
                if v_hist:
                    for ln in mini_chart(v_hist, name, H=5): L(ln)
                    L(f"  {'':22}  max={max(v_hist):.5f}")
                    L("")

        # ── Gráfico 3: Comparativa final barras ───────────────────────────────
        elif self.gi == 3:
            mk = mdl["metric_key"]
            for metric_k, metric_l in [
                (mk,                    mdl["metric_lbl"]),
                ("final_train_loss",    "Train Loss (final)"),
                ("final_val_loss",      "Val Loss (final)"),
            ]:
                vals = [(g["combination_id"],
                         g.get("metrics",{}).get(metric_k, None)) for g in grid]
                vals = [(c,v) for c,v in vals if v is not None]
                if not vals: continue
                max_v = max(v for _,v in vals) if vals else 1
                if max_v == 0: max_v = 1
                L(f"  ── {metric_l} ──────────────────────────────────────────")
                for cid, v in sorted(vals):
                    bar  = BAR * max(1, int(abs(v)/max_v*BW))
                    bm   = " ★" if cid == best_id else "  "
                    L(f"  combo_{cid:03d}{bm} │{bar:<{BW}}│ {v:.5f}")
                # Validación
                vm_val = d["val"].get("metrics",{}).get(metric_k)
                if vm_val:
                    bar2 = "░" * max(1, int(abs(vm_val)/max_v*BW))
                    L(f"  test_set    │{bar2:<{BW}}│ {vm_val:.5f}  (test)")
                L("")

        # ── Gráfico 4: Learning Rate (solo YOLOv11s-cls tiene CSV) ────────────
        elif self.gi == 4:
            if not mdl["has_csv"]:
                L(f"  Learning Rate: solo disponible para YOLOv11s-cls (tiene results.csv)")
                L(f"  El modelo {mdl['label']} usa optimizadores sin LR schedule per-epoch en CSV.")
                return
            for name in sel_combos:
                L(f"  ── {name} ──────────────────────────────────────────────────")
                df = hist.get(name)
                if df is None: continue
                for col_n in ["lr/pg0","lr/pg1","lr/pg2"]:
                    for c in df.columns:
                        if c.strip() == col_n:
                            v = pd.to_numeric(df[c], errors="coerce").dropna().tolist()
                            if v:
                                mn_v = min(v); mx_v = max(v)
                                L(f"  {col_n:<15}  max={mx_v:.2e}  min={mn_v:.2e}")
                                bar_w = BW
                                row = ""
                                for x in v:
                                    xi = int((x-mn_v)/(mx_v-mn_v+1e-9)*bar_w)
                                    row += "▪" if xi > bar_w//2 else "·"
                                L(f"  {'':15}  │{row}│")
                L("")

    # ── draw body ─────────────────────────────────────────────────────────────
    def draw_body(self):
        self._sz()
        rs = 3; PW = 24; MW = 18  # MW = ancho panel sub-modelos

        self.statusbar()

        # Panel izquierdo: lista de sub-modelos
        hl(self.scr, rs, 0, MW)
        sa(self.scr, rs, 1, " MODELOS CLS ", curses.color_pair(cH)|curses.A_BOLD)
        for i, mdl in enumerate(_CLS_MODELS):
            rw  = rs+1+i
            sel = i == self.mdl_idx
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else dim()
            sa(self.scr, rw, 1, f" {tr(mdl['label'], MW-3)}", at)

        # Separador vertical
        for r in range(rs, self.h-4):
            sa(self.scr, r, MW, "│", curses.color_pair(cBR))

        # Sub-tabs
        for i, t in enumerate(self.SUBTABS):
            at = curses.color_pair(cS)|curses.A_BOLD if i == self.sub else dim()
            sa(self.scr, rs, MW+2 + i*16, f" {t} ", at)

        # Panel derecho: lista de gráficos (solo en sub-tab 2)
        if self.sub == 2:
            hl(self.scr, rs, self.w-PW-2, PW+1)
            sa(self.scr, rs, self.w-PW-1, " GRAFICOS ", curses.color_pair(cH)|curses.A_BOLD)
            for i, g in enumerate(self.GRAPH_TYPES):
                rw = rs+1+i
                if rw >= self.h-4: break
                at = curses.color_pair(cS)|curses.A_BOLD if i == self.gi else dim()
                sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g, PW-4)}", at)
            sep = rs + len(self.GRAPH_TYPES) + 2
            sa(self.scr, sep, self.w-PW-1,
               f" Combo: {'Todos' if self.combo_sel==0 else f'combo_00{self.combo_sel}'}",
               curses.color_pair(cMT))
            sa(self.scr, sep+1, self.w-PW-1, " [0]Todos [1-4]Combo", dim())

        DW = (self.w - PW - MW - 3) if self.sub == 2 else (self.w - MW - 2)

        # Área de contenido
        hl(self.scr, rs+1, MW+1, DW)
        mdl_lbl = self._mdl()["label"]
        tab_lbl = self.GRAPH_TYPES[self.gi] if self.sub==2 else self.SUBTABS[self.sub]
        sa(self.scr, rs+1, MW+2, f" {mdl_lbl} · {tab_lbl} ",
           curses.color_pair(cGR)|curses.A_BOLD)

        if not self._built or not self.canvas:
            self._build()

        vis = self.h - rs - 6
        ms  = max(0, len(self.canvas) - vis)
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(vis):
            idx = self.scroll + i
            if idx >= len(self.canvas): break
            ln  = self.canvas[idx]
            if   "★" in ln:                       at = curses.color_pair(cMX)|curses.A_BOLD
            elif "══" in ln or "──" in ln:         at = curses.color_pair(cP)|curses.A_BOLD
            elif "│" in ln and "█" in ln:          at = curses.color_pair(cGR)
            elif "│" in ln and "░" in ln:          at = curses.color_pair(cMT)
            elif "·· VARÍAN" in ln:                at = curses.color_pair(cER)|curses.A_BOLD
            elif "·· COMUNES" in ln:               at = curses.color_pair(cDM)
            else:                                  at = curses.color_pair(cN)
            sa(self.scr, rs+2+i, MW+2, tr(ln, DW-2), at)

        if self.sub == 2:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Modelo  [←→]Gráfico  [0-4]Combo  [1-5]Gráfico directo "
                     " [1/2/3]Sub-tab  [Tab]Sig. ")
        else:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Modelo  [1/2/3]Sub-tab  [j/k]Scroll  [Tab]Sig. ")

    # ── key handler ───────────────────────────────────────────────────────────
    def key(self, k):
        # Cambio de sub-modelo
        if k == curses.KEY_UP:
            self.mdl_idx = max(0, self.mdl_idx-1); self._reset()
        elif k == curses.KEY_DOWN:
            self.mdl_idx = min(len(_CLS_MODELS)-1, self.mdl_idx+1); self._reset()
        # Cambio de sub-tab
        elif k == ord('1') and self.sub != 0:
            self.sub = 0; self._reset()
        elif k == ord('2') and self.sub != 1:
            self.sub = 1; self._reset()
        elif k == ord('3') and self.sub != 2:
            self.sub = 2; self._reset()
        # Scroll del canvas
        elif k == ord('j'): self.scroll += 1
        elif k == ord('k'): self.scroll = max(0, self.scroll-1)
        # Navegación de gráficos
        elif self.sub == 2:
            if k == curses.KEY_LEFT:
                self.gi = (self.gi-1) % len(self.GRAPH_TYPES); self._reset()
            elif k == curses.KEY_RIGHT:
                self.gi = (self.gi+1) % len(self.GRAPH_TYPES); self._reset()
            elif ord('0') <= k <= ord('4'):
                self.combo_sel = k-ord('0'); self._reset()
            elif ord('1') <= k <= ord('5'):
                self.gi = k-ord('1'); self._reset()
        return None



# ══════════════════════════════════════════════════════════════════════════════
# § 11d  VLM (F3) y LLM (F4) — pantallas genéricas base+highcap
# ══════════════════════════════════════════════════════════════════════════════

# ── Definición de modelos ─────────────────────────────────────────────────────
_VLM_MODELS = [
    {
        "key":   "llavanext",
        "label": "LLaVA-Next",
        "base":  Path("/workspace/models/phase3_vlm/llavanext"),
        "checkpoint": "checkpoint-982",
    },
    {
        "key":   "qwen2vl",
        "label": "Qwen2-VL",
        "base":  Path("/workspace/models/phase3_vlm/qwen2vl"),
        "checkpoint": "checkpoint-2786",
    },
]

_LLM_MODELS = [
    {
        "key":   "mistral",
        "label": "Mistral-7B",
        "base":  Path("/workspace/models/phase4_llm/mistral-7b"),
        "checkpoint": "checkpoint-13710",
    },
    {
        "key":   "qwen2",
        "label": "Qwen2-7B",
        "base":  Path("/workspace/models/phase4_llm/qwen2-7b"),
        "checkpoint": "checkpoint-13710",
    },
]

# Variantes fijas para VLM/LLM
_MODEL_VARIANTS = ["base_model", "highcap_model"]


def _load_vlm_llm_data(mdl: dict) -> dict:
    """Carga datos de un modelo VLM o LLM (base + highcap)."""
    base = mdl["base"]
    ckpt = mdl["checkpoint"]
    d = {"mdl": mdl, "variants": {}}

    for variant in _MODEL_VARIANTS:
        vpath = base / variant
        vd = {"path": vpath, "exists": vpath.exists()}

        if variant == "base_model":
            # Solo archivos de configuración del modelo base
            cfg = _load_json(vpath / "config.json") or {}
            gen = _load_json(vpath / "generation_config.json") or {}
            vd["config"]      = cfg
            vd["gen_config"]  = gen
            vd["is_lora"]     = False
            vd["metrics"]     = {}
            vd["trainer"]     = {}
            vd["history"]     = {}
        else:
            # highcap_model: metrics.json + checkpoint/trainer_state.json
            vd["metrics"]    = _load_json(vpath / "metrics.json") or {}
            vd["adapter_cfg"]= _load_json(vpath / "final_model" / "adapter_config.json") or {}
            vd["is_lora"]    = (vpath / "final_model" / "adapter_config.json").exists()

            # trainer_state.json del checkpoint (contiene el log_history con pérdidas)
            ts_path = vpath / ckpt / "trainer_state.json"
            ts = _load_json(ts_path) or {}
            vd["trainer"] = ts

            # Extraer history de pérdidas del log_history
            log_history = ts.get("log_history", [])
            train_loss = [e["loss"] for e in log_history
                          if "loss" in e and "eval_loss" not in e]
            eval_loss  = [e["eval_loss"] for e in log_history
                          if "eval_loss" in e]
            eval_rouge = [e.get("eval_rouge1", e.get("eval_rougeL", None))
                          for e in log_history if "eval_rouge1" in e or "eval_rougeL" in e]
            eval_bleu  = [e.get("eval_bleu", None)
                          for e in log_history if "eval_bleu" in e]
            vd["history"] = {
                "train_loss": train_loss,
                "eval_loss":  eval_loss,
                "eval_rouge": [x for x in eval_rouge if x is not None],
                "eval_bleu":  [x for x in eval_bleu  if x is not None],
                "log_history": log_history,
            }
            vd["config"] = _load_json(vpath / "final_model" / "adapter_config.json") or {}

        d["variants"][variant] = vd
    return d


class _VLMLLMScreen(Screen):
    """
    Sub-pantalla genérica para VLM (F3) o LLM (F4).
    Panel izquierdo: modelos (ej: LLaVA-Next, Qwen2-VL)
    Sub-panel: base_model / highcap_model  (navegable con [←→])
    Sub-tabs: [1]Detalle  [2]Métricas  [3]Gráficos
    """
    SUBTABS     = ["[1]Detalle", "[2]Metricas", "[3]Graficos"]
    GRAPH_TYPES = [
        "Train Loss por paso",
        "Eval Loss por evaluación",
        "Eval ROUGE / BLEU",
        "Comparativa base vs highcap",
    ]

    def __init__(self, scr, st, models_list: list, phase_label: str):
        super().__init__(scr, st)
        self._models   = models_list
        self._phase    = phase_label       # "VLM (F3)" o "LLM (F4)"
        self.mdl_idx   = 0                 # índice en models_list
        self.var_idx   = 0                 # 0=base_model 1=highcap_model
        self.sub       = 0
        self.scroll    = 0
        self.canvas    = []
        self.gi        = 0
        self._cache    = {}
        self._built    = False

    def _mdl(self):  return self._models[self.mdl_idx]
    def _var(self):  return _MODEL_VARIANTS[self.var_idx]

    def _data(self):
        key = self._mdl()["key"]
        if key not in self._cache:
            self._cache[key] = _load_vlm_llm_data(self._mdl())
        return self._cache[key]

    def _vdata(self):
        return self._data()["variants"][self._var()]

    def _reset(self):
        self.canvas = []; self.scroll = 0; self._built = False

    # ── builders ──────────────────────────────────────────────────────────────
    def _build(self):
        self.canvas = []
        vd = self._vdata()
        if   self.sub == 0: self._build_detalle(vd)
        elif self.sub == 1: self._build_metricas(vd)
        elif self.sub == 2: self._build_graficos(vd)
        self._built = True

    def _build_detalle(self, vd):
        L = self.canvas.append
        mdl     = self._mdl()
        variant = self._var()
        is_high = (variant == "highcap_model")

        L("  ══════════════════════════════════════════════════════════════════")
        L(f"    DETALLE — {self._phase}  |  {mdl['label']}  |  {variant}")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")
        L(f"  Modelo    : {mdl['label']}")
        L(f"  Variante  : {variant}")
        L(f"  Ruta      : {vd['path']}")
        L(f"  Existe    : {'✓ Sí' if vd['exists'] else '✗ No encontrado'}")
        L(f"  Es LoRA   : {'✓ Sí (adaptador sobre base)' if vd.get('is_lora') else '✗ No (modelo completo)'}")
        L("")

        if is_high:
            # ── Configuración del adaptador LoRA ─────────────────────────────
            ac = vd.get("adapter_cfg", {})
            if ac:
                L("  ── CONFIGURACIÓN LORA (adapter_config.json) ─────────────────")
                lora_keys = [
                    ("r",               "Rango LoRA (r)"),
                    ("lora_alpha",      "Alpha LoRA"),
                    ("lora_dropout",    "Dropout LoRA"),
                    ("bias",            "Bias"),
                    ("peft_type",       "Tipo PEFT"),
                    ("task_type",       "Tipo tarea"),
                    ("base_model_name_or_path", "Modelo base"),
                    ("target_modules",  "Módulos objetivo"),
                    ("inference_mode",  "Modo inferencia"),
                ]
                for k, lbl in lora_keys:
                    v = ac.get(k)
                    if v is not None:
                        if isinstance(v, list):
                            v = ", ".join(str(x) for x in v[:8])
                        L(f"  {lbl:<30} : {v}")
                L("")

            # ── Métricas finales del entrenamiento ────────────────────────────
            mt = vd.get("metrics", {})
            if mt:
                L("  ── MÉTRICAS FINALES (metrics.json) ──────────────────────────")
                for k, v in sorted(mt.items()):
                    if isinstance(v, (int, float)):
                        L(f"  {k:<35} : {v:.6f}" if isinstance(v, float) else
                          f"  {k:<35} : {v}")
                    elif isinstance(v, str):
                        L(f"  {k:<35} : {v}")
                L("")

            # ── Estado del trainer ────────────────────────────────────────────
            ts = vd.get("trainer", {})
            if ts:
                L("  ── ESTADO DEL TRAINER (trainer_state.json) ─────────────────")
                info_keys = [
                    ("best_metric",        "Mejor métrica"),
                    ("best_model_checkpoint","Mejor checkpoint"),
                    ("epoch",              "Épocas completadas"),
                    ("global_step",        "Pasos totales"),
                    ("max_steps",          "Pasos máximos"),
                    ("num_train_epochs",   "Épocas configuradas"),
                    ("total_flos",         "FLOPs totales"),
                    ("train_batch_size",   "Batch train"),
                    ("trial_name",         "Trial"),
                    ("trial_params",       "Parámetros trial"),
                ]
                for k, lbl in info_keys:
                    v = ts.get(k)
                    if v is not None:
                        if isinstance(v, float):
                            L(f"  {lbl:<30} : {v:.6f}")
                        else:
                            L(f"  {lbl:<30} : {v}")

                # Último log entry
                log_h = ts.get("log_history", [])
                if log_h:
                    last = log_h[-1]
                    L("")
                    L("  ── ÚLTIMO PUNTO DE LOG ──────────────────────────────────")
                    for k, v in sorted(last.items()):
                        if isinstance(v, float):
                            L(f"  {k:<30} : {v:.6f}")
                        else:
                            L(f"  {k:<30} : {v}")
                L("")

            # ── Checkpoint ────────────────────────────────────────────────────
            ckpt_dir = vd["path"] / mdl["checkpoint"]
            L(f"  ── CHECKPOINT ───────────────────────────────────────────────")
            L(f"  Directorio : {ckpt_dir}")
            L(f"  Existe     : {'✓ Sí' if ckpt_dir.exists() else '✗ No'}")
            files = list(ckpt_dir.glob("*")) if ckpt_dir.exists() else []
            if files:
                L(f"  Archivos   : {', '.join(f.name for f in sorted(files))}")
        else:
            # ── Base model ────────────────────────────────────────────────────
            cfg = vd.get("config", {})
            if cfg:
                L("  ── CONFIGURACIÓN DEL MODELO BASE (config.json) ──────────────")
                cfg_keys = [
                    ("model_type",           "Tipo de modelo"),
                    ("architectures",        "Arquitecturas"),
                    ("hidden_size",          "Hidden size"),
                    ("num_hidden_layers",    "Capas ocultas"),
                    ("num_attention_heads",  "Cabezas atención"),
                    ("intermediate_size",    "Tamaño intermedio"),
                    ("max_position_embeddings","Pos. max."),
                    ("vocab_size",           "Vocab size"),
                    ("torch_dtype",          "Dtype"),
                    ("transformers_version", "Versión transformers"),
                ]
                for k, lbl in cfg_keys:
                    v = cfg.get(k)
                    if v is not None:
                        if isinstance(v, list): v = ", ".join(str(x) for x in v[:4])
                        L(f"  {lbl:<30} : {v}")
            gen = vd.get("gen_config", {})
            if gen:
                L("")
                L("  ── CONFIGURACIÓN DE GENERACIÓN ──────────────────────────────")
                for k, v in sorted(gen.items()):
                    if not k.startswith("_"):
                        L(f"  {k:<30} : {v}")
            L("")
            L("  Nota: el base_model no tiene métricas de fine-tuning.")
            L("  Selecciona highcap_model para ver resultados de entrenamiento LoRA.")

    def _build_metricas(self, vd):
        L = self.canvas.append
        mdl     = self._mdl()
        variant = self._var()
        is_high = (variant == "highcap_model")

        L("  ══════════════════════════════════════════════════════════════════")
        L(f"    MÉTRICAS — {self._phase}  |  {mdl['label']}  |  {variant}")
        L("  ══════════════════════════════════════════════════════════════════")
        L("")

        if not is_high:
            L("  El base_model no tiene métricas de fine-tuning.")
            L("  Selecciona [→] highcap_model para ver métricas de entrenamiento LoRA.")
            return

        mt  = vd.get("metrics", {})
        ts  = vd.get("trainer", {})
        h   = vd.get("history", {})
        BW  = 40

        # ── Métricas principales ──────────────────────────────────────────────
        if mt:
            L("  ── MÉTRICAS FINALES ─────────────────────────────────────────────")
            numeric_mt = {k: v for k, v in mt.items() if isinstance(v, (int, float))}
            # Detectar métrica principal
            main_metric_keys = ["eval_rouge1","eval_rougeL","eval_bleu",
                                 "eval_meteor","eval_loss","train_loss"]
            for mk in main_metric_keys:
                if mk in numeric_mt:
                    v = numeric_mt[mk]
                    bar = "█" * max(1, min(BW, int(v * BW))) if v <= 1 else ""
                    L(f"  {mk:<30} : {v:.6f}  {bar}")
            L("")
            # Resto de métricas
            L(f"  {'Métrica':<35} {'Valor':>12}")
            L("  " + "─"*50)
            for k, v in sorted(numeric_mt.items()):
                if isinstance(v, float):
                    L(f"  {k:<35} {v:>12.6f}")
                else:
                    L(f"  {k:<35} {v:>12}")
            L("")

        # ── Resumen del entrenamiento ─────────────────────────────────────────
        if ts:
            L("  ── RESUMEN DEL ENTRENAMIENTO ────────────────────────────────────")
            tr_keys = [
                ("epoch",              "Épocas"),
                ("global_step",        "Pasos totales"),
                ("best_metric",        "Mejor métrica"),
                ("best_model_checkpoint", "Mejor checkpoint"),
            ]
            for k, lbl in tr_keys:
                v = ts.get(k)
                if v is not None:
                    L(f"  {lbl:<25} : {v:.6f}" if isinstance(v, float) else
                      f"  {lbl:<25} : {v}")
            L("")

        # ── Evolución de métricas por evaluación ──────────────────────────────
        log_h = ts.get("log_history", [])
        eval_entries = [e for e in log_h if "eval_loss" in e]
        if eval_entries:
            L("  ── MÉTRICAS POR EVALUACIÓN (todos los checkpoints) ─────────────")
            # Detectar columnas disponibles
            eval_cols = [k for k in eval_entries[0].keys()
                         if k not in ("epoch", "step") and "eval" in k]
            hdr = f"  {'Step':>8}  {'Epoch':>7}"
            for c in eval_cols[:5]:
                hdr += f"  {c[:14]:>14}"
            L(hdr)
            L("  " + "─"*max(50, len(hdr)-2))
            for e in eval_entries:
                row = f"  {e.get('step',0):>8}  {e.get('epoch',0):>7.2f}"
                for c in eval_cols[:5]:
                    v = e.get(c, 0)
                    row += f"  {v:>14.6f}"
                L(row)
            L("")

        # ── Comparativa base vs highcap ───────────────────────────────────────
        d      = self._data()
        base_d = d["variants"]["base_model"]
        high_d = d["variants"]["highcap_model"]
        L("  ── COMPARATIVA BASE vs HIGHCAP ──────────────────────────────────")
        L(f"  {'Aspecto':<30} {'base_model':<20} {'highcap_model'}")
        L("  " + "─"*70)
        L(f"  {'Es LoRA':<30} {'No':<20} {'Sí (adapter)'}")
        hm  = high_d.get("metrics", {})
        for k in ["eval_loss","eval_rouge1","eval_rougeL","eval_bleu","eval_meteor"]:
            v = hm.get(k)
            if v is not None:
                L(f"  {k:<30} {'N/A (base)':<20} {v:.6f}")
        ht = high_d.get("trainer", {})
        L(f"  {'Pasos entrenados':<30} {'0':<20} {ht.get('global_step','—')}")
        L(f"  {'Épocas':<30} {'0':<20} {ht.get('epoch','—')}")

        # ── Métricas extendidas escaneadas desde /workspace/models/ ───────────
        phase_k = "vlm" if "VLM" in self._phase else "llm"
        L(f"  ══ ANÁLISIS EXTENDIDO — TODOS LOS CAMPOS EN /workspace/models/ ══")
        L("")
        scan_path = self._mdl()["base"]
        raw_all = _scan_all_metrics(scan_path)
        _render_extended_metrics(L, phase_k, raw_all, scan_path)

    def _build_graficos(self, vd):
        L = self.canvas.append
        mdl     = self._mdl()
        variant = self._var()
        is_high = (variant == "highcap_model")
        BW      = 45; BAR = "█"

        L(f"  ══ {self.GRAPH_TYPES[self.gi].upper()} — {mdl['label']} | {variant} ══")
        L("")

        if not is_high and self.gi < 3:
            L("  El base_model no tiene historial de entrenamiento.")
            L("  Selecciona [→] highcap_model o usa el Gráfico 4 (comparativa).")
            return

        h  = vd.get("history", {})
        ts = vd.get("trainer", {})

        def mini_chart(vals, label, W=BW, H=6, fmt=".4f"):
            if not vals:
                return [f"  {label}: sin datos"]
            lines = []
            mn = float(min(vals)); mx = float(max(vals))
            rng = mx - mn if mx != mn else 1e-9
            N = len(vals); step = max(1, N // W)
            sampled = [vals[i] for i in range(0, N, step)][:W]
            cts = [int((x-mn)/rng*(H-1)) for x in sampled]
            lines.append(f"  {label:<22}  {mn:{fmt}}──{'─'*W}──{mx:{fmt}}")
            for ri in range(H, 0, -1):
                row = "".join(BAR if ct >= ri else " " for ct in cts)
                lines.append(f"  {'':22}  {'':10} │{row}│")
            lines.append(f"  {'':22}  paso_1{'─'*(W-12)}paso_{N}")
            return lines

        # ── Gráfico 0: Train Loss ─────────────────────────────────────────────
        if self.gi == 0:
            tl = h.get("train_loss", [])
            if tl:
                for ln in mini_chart(tl, "Train Loss"): L(ln)
                L(f"  n={len(tl)}  inicio={tl[0]:.5f}  "
                  f"final={tl[-1]:.5f}  min={min(tl):.5f}")
                # Mostrar primeros y últimos 5
                L("")
                L(f"  Primeros 5 pasos: {[round(x,4) for x in tl[:5]]}")
                L(f"  Últimos  5 pasos: {[round(x,4) for x in tl[-5:]]}")
            else:
                L("  Sin datos de train_loss en trainer_state.json")

        # ── Gráfico 1: Eval Loss ──────────────────────────────────────────────
        elif self.gi == 1:
            el = h.get("eval_loss", [])
            if el:
                for ln in mini_chart(el, "Eval Loss"): L(ln)
                L(f"  n={len(el)}  inicio={el[0]:.5f}  "
                  f"final={el[-1]:.5f}  min={min(el):.5f}")
                log_h = ts.get("log_history", [])
                eval_entries = [e for e in log_h if "eval_loss" in e]
                if eval_entries:
                    L("")
                    L(f"  {'Eval#':>6}  {'Step':>8}  {'Epoch':>7}  {'eval_loss':>12}")
                    L("  " + "─"*40)
                    for i, e in enumerate(eval_entries[:20]):
                        L(f"  {i+1:>6}  {e.get('step',0):>8}  "
                          f"{e.get('epoch',0):>7.2f}  {e.get('eval_loss',0):>12.6f}")
                    if len(eval_entries) > 20:
                        L(f"  ... ({len(eval_entries)-20} más)")
            else:
                L("  Sin datos de eval_loss")

        # ── Gráfico 2: ROUGE / BLEU ───────────────────────────────────────────
        elif self.gi == 2:
            er = h.get("eval_rouge", [])
            eb = h.get("eval_bleu", [])
            if er:
                for ln in mini_chart(er, "ROUGE (eval)", fmt=".4f"): L(ln)
                L(f"  n={len(er)}  max={max(er):.5f}  final={er[-1]:.5f}")
                L("")
            if eb:
                for ln in mini_chart(eb, "BLEU (eval)", fmt=".4f"): L(ln)
                L(f"  n={len(eb)}  max={max(eb):.5f}  final={eb[-1]:.5f}")
                L("")

            # Tabla detallada de todas las métricas eval
            log_h = ts.get("log_history", [])
            eval_entries = [e for e in log_h if "eval_loss" in e]
            text_metrics = set()
            for e in eval_entries:
                text_metrics |= {k for k in e if k not in ("step","epoch","eval_loss")}
            if text_metrics and eval_entries:
                cols = sorted(text_metrics)[:6]
                hdr = f"  {'Step':>8}  {'Epoch':>7}"
                for c in cols: hdr += f"  {c[:12]:>12}"
                L(hdr)
                L("  " + "─"*max(40, len(hdr)))
                for e in eval_entries:
                    row = f"  {e.get('step',0):>8}  {e.get('epoch',0):>7.2f}"
                    for c in cols:
                        v = e.get(c, None)
                        row += f"  {v:>12.6f}" if v is not None else f"  {'—':>12}"
                    L(row)
            if not er and not eb and not text_metrics:
                L("  Sin métricas de texto disponibles en el trainer_state.")
                L("  Las métricas ROUGE/BLEU pueden estar en metrics.json")
                mt = vd.get("metrics", {})
                for k, v in sorted(mt.items()):
                    if "rouge" in k.lower() or "bleu" in k.lower():
                        if isinstance(v, float):
                            bar = "█" * max(1, min(BW, int(v*BW)))
                            L(f"  {k:<30} : {v:.6f}  {bar}")

        # ── Gráfico 3: Comparativa base vs highcap ────────────────────────────
        elif self.gi == 3:
            d      = self._data()
            base_d = d["variants"]["base_model"]
            high_d = d["variants"]["highcap_model"]
            hmt    = high_d.get("metrics", {})
            hts    = high_d.get("trainer", {})
            hh     = high_d.get("history", {})

            L("  ── CONFIGURACIÓN DEL MODELO BASE ────────────────────────────")
            bcfg = base_d.get("config", {})
            for k in ["model_type","hidden_size","num_hidden_layers",
                      "num_attention_heads","vocab_size"]:
                v = bcfg.get(k)
                if v: L(f"  {k:<30} : {v}")
            L("")

            L("  ── HIGHCAP: ADAPTADOR LoRA ──────────────────────────────────")
            hac = high_d.get("adapter_cfg", {})
            for k in ["r","lora_alpha","lora_dropout","target_modules","peft_type"]:
                v = hac.get(k)
                if v is not None:
                    if isinstance(v, list): v = ", ".join(str(x) for x in v[:8])
                    L(f"  {k:<30} : {v}")
            L("")

            L("  ── MÉTRICAS HIGHCAP FINALES ─────────────────────────────────")
            for k, v in sorted(hmt.items()):
                if isinstance(v, (int, float)):
                    bar = "█" * max(1, min(BW, int(abs(v)*BW/max(abs(v),1))))
                    L(f"  {k:<30} : {v:.6f}  {bar[:20]}")
            L("")

            L("  ── CURVA TRAIN LOSS (highcap) ───────────────────────────────")
            tl = hh.get("train_loss", [])
            if tl:
                for ln in mini_chart(tl, "Train Loss highcap", W=min(BW, 50), H=5):
                    L(ln)
                L(f"  inicio={tl[0]:.5f}  final={tl[-1]:.5f}  "
                  f"reducción={tl[0]-tl[-1]:+.5f}")

    # ── draw body ─────────────────────────────────────────────────────────────
    def draw_body(self):
        self._sz()
        rs  = 3; MW = 16; PW = 24

        self.statusbar()

        # Panel izquierdo: modelos
        hl(self.scr, rs, 0, MW)
        lbl = "MODELOS VLM" if "VLM" in self._phase else "MODELOS LLM"
        sa(self.scr, rs, 1, f" {lbl} ", curses.color_pair(cH)|curses.A_BOLD)
        for i, mdl in enumerate(self._models):
            rw  = rs+1+i
            sel = i == self.mdl_idx
            at  = curses.color_pair(cS)|curses.A_BOLD if sel else dim()
            sa(self.scr, rw, 1, f" {tr(mdl['label'], MW-3)}", at)

        # Separador vertical izquierdo
        for r in range(rs, self.h-4):
            sa(self.scr, r, MW, "│", curses.color_pair(cBR))

        # Sub-tabs y variante
        for i, t in enumerate(self.SUBTABS):
            at = curses.color_pair(cS)|curses.A_BOLD if i == self.sub else dim()
            sa(self.scr, rs, MW+2 + i*16, f" {t} ", at)

        # Variantes base/highcap
        for i, v in enumerate(_MODEL_VARIANTS):
            at = curses.color_pair(cS)|curses.A_BOLD if i == self.var_idx else dim()
            sa(self.scr, rs+1, MW+2 + i*18, f" {'[←]' if i==0 else '[→]'} {v} ", at)

        # Panel derecho en modo Gráficos
        if self.sub == 2:
            hl(self.scr, rs, self.w-PW-2, PW+1)
            sa(self.scr, rs, self.w-PW-1, " GRAFICOS ", curses.color_pair(cH)|curses.A_BOLD)
            for i, g in enumerate(self.GRAPH_TYPES):
                rw = rs+1+i
                if rw >= self.h-4: break
                at = curses.color_pair(cS)|curses.A_BOLD if i == self.gi else dim()
                sa(self.scr, rw, self.w-PW-1, f" {i+1}. {tr(g, PW-4)}", at)

        DW = (self.w - PW - MW - 3) if self.sub == 2 else (self.w - MW - 2)

        # Barra de título del área de contenido
        hl(self.scr, rs+2, MW+1, DW)
        var_lbl = _MODEL_VARIANTS[self.var_idx]
        tab_lbl = self.GRAPH_TYPES[self.gi] if self.sub==2 else self.SUBTABS[self.sub]
        sa(self.scr, rs+2, MW+2,
           f" {self._mdl()['label']} · {var_lbl} · {tab_lbl} ",
           curses.color_pair(cGR)|curses.A_BOLD)

        if not self._built or not self.canvas:
            self._build()

        vis = self.h - rs - 7
        ms  = max(0, len(self.canvas) - vis)
        self.scroll = max(0, min(self.scroll, ms))
        for i in range(vis):
            idx = self.scroll + i
            if idx >= len(self.canvas): break
            ln  = self.canvas[idx]
            if   "★" in ln:                  at = curses.color_pair(cMX)|curses.A_BOLD
            elif "══" in ln or "──" in ln:   at = curses.color_pair(cP)|curses.A_BOLD
            elif "│" in ln and "█" in ln:    at = curses.color_pair(cGR)
            elif "✓" in ln:                  at = curses.color_pair(cOK)
            elif "✗" in ln:                  at = curses.color_pair(cER)
            elif "Nota:" in ln:              at = curses.color_pair(cMT)
            else:                            at = curses.color_pair(cN)
            sa(self.scr, rs+3+i, MW+2, tr(ln, DW-2), at)

        if self.sub == 2:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Modelo  [←→]Variante  [1/2/3]Sub-tab  "
                     "[j/k]Scroll  [1-4]Gráfico  [Tab]Sig. ")
        else:
            hint_bar(self.scr, self.h-3,
                     " [↑↓]Modelo  [←→]Variante  [1/2/3]Sub-tab  "
                     "[j/k]Scroll  [Tab]Sig. ")

    # ── key handler ───────────────────────────────────────────────────────────
    def key(self, k):
        if k == curses.KEY_UP:
            self.mdl_idx = max(0, self.mdl_idx-1); self._reset()
        elif k == curses.KEY_DOWN:
            self.mdl_idx = min(len(self._models)-1, self.mdl_idx+1); self._reset()
        elif k == curses.KEY_LEFT:
            self.var_idx = max(0, self.var_idx-1); self._reset()
        elif k == curses.KEY_RIGHT:
            self.var_idx = min(1, self.var_idx+1); self._reset()
        elif k == ord('1') and self.sub != 0:
            self.sub = 0; self._reset()
        elif k == ord('2') and self.sub != 1:
            self.sub = 1; self._reset()
        elif k == ord('3') and self.sub != 2:
            self.sub = 2; self._reset()
        elif k == ord('j'): self.scroll += 1
        elif k == ord('k'): self.scroll = max(0, self.scroll-1)
        elif self.sub == 2:
            if ord('1') <= k <= ord('4'):
                self.gi = k-ord('1'); self._reset()
        return None

class ModelosScreen(Screen):
    """
    Pestaña [6] Modelos — [D]etección [C]lasificación [V]LM [L]LM
    """
    MENUS = ["[D]eteccion", "[C]lasificacion", "[V]LM", "[L]LM"]

    def __init__(self, scr, st):
        super().__init__(scr, st)
        self.menu = 0
        self._det = _DeteccionScreen(scr, st)
        self._cls = _ClasificacionScreen(scr, st)
        self._vlm = _VLMLLMScreen(scr, st, _VLM_MODELS, "VLM (F3)")
        self._llm = _VLMLLMScreen(scr, st, _LLM_MODELS, "LLM (F4)")

    def draw(self):
        self.scr.erase(); self._sz()
        hint = "  ".join(
            f" {t} " if i == self.menu else t
            for i, t in enumerate(self.MENUS)
        )
        self.title(5, hint)
        scr_map = {0: self._det, 1: self._cls, 2: self._vlm, 3: self._llm}
        s = scr_map[self.menu]
        s.h = self.h; s.w = self.w
        if self.menu == 0:
            s.draw_body()
        elif self.menu == 1:
            s.draw_body()
        else:
            s.draw_body()
        self.navbar(" [D/C/V/L]Modelo ")
        self.scr.refresh()

    def key(self, k):
        if   k in (ord('d'), ord('D')): self.menu=0; return None
        elif k in (ord('c'), ord('C')): self.menu=1; return None
        elif k in (ord('v'), ord('V')): self.menu=2; return None
        elif k in (ord('l'), ord('L')): self.menu=3; return None
        elif k == ord('\t'): return "next"
        elif k in (ord('q'), 27): return "prev"
        scr_map = {0: self._det, 1: self._cls, 2: self._vlm, 3: self._llm}
        return scr_map[self.menu].key(k)


def run_tui(stdscr, state):
    curses.curs_set(0); init_colors()
    stdscr.keypad(True); stdscr.timeout(80)
    listas   = ListasScreen(stdscr, state)
    analisis = AnálisisScreen(stdscr, state)
    modelos  = ModelosScreen(stdscr, state)
    scrs = [
        ConfigScreen(stdscr, state),   # 0 → [1] Config
        DBScreen(stdscr, state),        # 1 → [2] BD
        listas,                         # 2 → [3] Listas
        RunScreen(stdscr, state),       # 3 → [4] Ejecutar
        analisis,                       # 4 → [5] Analisis
        modelos,                        # 5 → [6] Modelos
    ]
    cur = 0
    while True:
        stdscr.timeout(150 if cur == 3 and scrs[3].running else 80)
        scrs[cur].draw()
        k = stdscr.getch()
        if k == -1: continue
        res = scrs[cur].key(k)
        if   res == "quit": break
        elif res == "next": cur = (cur+1) % len(scrs)
        elif res == "prev": cur = (cur-1) % len(scrs)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# § 12  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Suite Multiobjetivo v2 — Framework Multimodal de Tráfico\n\n"
            "ARCHIVOS DEL SISTEMA:\n"
            "  multiobjective_suite.py   ← este script (todo en uno)\n"
            "  multiobjective_db.json    ← BD de configs + runs (auto-creado)\n"
            "  results_XXXXXXXX.csv      ← generado por diagnostic_validation_full_trace.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input",  "-i", default=None,
                    help="CSV generado por diagnostic_validation_full_trace.py")
    ap.add_argument("--config", "-c", default=None, type=int,
                    help="ID de configuración en BD a cargar al inicio")
    ap.add_argument("--db",     "-d", default="multiobjective_db.json",
                    help="Ruta del JSON de BD (default: multiobjective_db.json)")
    args = ap.parse_args()

    global DB_PATH
    DB_PATH = Path(args.db)

    if not _NUM:
        print("⚠  numpy/pandas no instalados. El motor de cálculo no funcionará.")
        print("   Instala con:  pip install numpy pandas scipy")
        input("   [Enter] para continuar de todas formas...")

    state = {
        "objectives":  default_objectives(),
        "methods":     {m: True for m in METHODS},
        "rho":         0.001,
        "input_csv":   args.input or "",
        "last_result": "",
        "_db_id":      None,
        "_db_name":    None,
    }

    if args.config is not None:
        ld = db_load(args.config)
        if ld:
            state.update(ld)
            print(f"✓ Config ID={args.config} cargada: {ld.get('_db_name','?')}")
            if args.input: state["input_csv"] = args.input  # --input tiene prioridad
        else:
            print(f"⚠  Config ID={args.config} no encontrada en {DB_PATH}")

    try:
        out = os.popen("stty size","r").read().split()
        if len(out) == 2 and (int(out[0]) < 24 or int(out[1]) < 100):
            print(f"⚠  Terminal pequeña ({out[1]}×{out[0]}). Mínimo recomendado: 100×24")
            sys.exit(1)
    except Exception:
        pass

    final = curses.wrapper(run_tui, state)

    # ── Resumen al salir ──────────────────────────────────────────────────────
    en = [o for o in final["objectives"] if o["enabled"]]
    runs_list = run_list()
    print("\n"+"═"*70)
    print("  Suite Multiobjetivo v2 — Estado final al salir")
    print("═"*70)
    print(f"  BD             : {DB_PATH.resolve()}")
    print(f"  Configs en BD  : {len(db_list())}")
    print(f"  Runs en BD     : {len(runs_list)}")
    if runs_list:
        last = runs_list[0]
        print(f"  Última run     : ID={last['id']}  {last['ran_at']}  → {last['out_dir']}")
    print(f"  Objetivos activos : {len(en)}")
    for o in en:
        print(f"    {'↑' if o['dir']=='max' else '↓'} {o['lbl']:<22} w={o['weight']:.4f}  ({o['col']})")
    print(f"  Suma pesos : {wsum(final['objectives']):.4f}")
    print(f"  Métodos    : {[m for m,v in final['methods'].items() if v]}")
    print(f"  ρ ASF      : {final['rho']}")
    print("═"*70+"\n")


if __name__ == "__main__":
    main()
