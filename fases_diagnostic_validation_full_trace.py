#!/usr/bin/env python3
"""
fases_diagnostic_validation_full_trace.py
==========================================
Script de diagnóstico por fases seleccionables.

Modos de fase:
  Una fase   : F1 | F2 | F3 | F4              → 1 GPU
  Dos fases  : F1F2 | F1F3 | F1F4 | F2F3 | F2F4 | F3F4 → 2 GPUs
  Tres fases : F1F2F3 | F1F2F4 | F2F3F4       → 2 GPUs
  Cuatro fases: F1F2F3F4                       → 2 GPUs (igual a diagnostic_validation_full_trace.py)

Salida:
  Log  → fases_trace_log_YYYYMMDD_HHMMSS.txt
  CSV  → fases_results_YYYYMMDD_HHMMSS.csv  (mismos campos que script original)

Estrategia VRAM VLM/LLM:
  - base_model  → se mantiene permanente en GPU  (no se descarga nunca)
  - LoRA adapter → se carga y se libera por swap (solo los deltas delta-weights)
  - Si ambos modelos son base → se mantienen ambos en GPU simultáneamente
  - Si uno es highcap (LoRA) → el adapter se descarga al cambiar

CSV flush:
  - Una fase   : cada ejecución individual (flush inmediato)
  - Dos/tres   : cada 5 combinaciones completas
  - Cuatro     : cada 10 combinaciones completas (igual que original)
"""

# ============================================================================
# IMPORTS (idénticos al script base)
# ============================================================================
import argparse
import json
import time
import csv
import random
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import psutil
import warnings
import gc
import sys
import os
from collections import OrderedDict

warnings.filterwarnings('ignore')

from ultralytics import YOLO
import timm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE_CACHE: dict = {}

    def _nvml_get_handle(idx):
        if idx not in _NVML_HANDLE_CACHE:
            _NVML_HANDLE_CACHE[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return _NVML_HANDLE_CACHE[idx]

    def _nvml_gpu_info(idx):
        h = _nvml_get_handle(idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            'id': idx, 'name': name,
            'vram_used_mb': mem.used / (1024**2),
            'vram_total_mb': mem.total / (1024**2),
            'vram_percent': mem.used / mem.total * 100,
            'utilization': util.gpu,
            'temperature': temp
        }

    _NVML_NUM_GPUS = pynvml.nvmlDeviceGetCount()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    _NVML_NUM_GPUS = 0

if not GPU_AVAILABLE:
    try:
        import GPUtil
        GPU_AVAILABLE = True
        _NVML_NUM_GPUS = 0
    except Exception:
        GPU_AVAILABLE = False

try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = "eager"

CLASSES = ["accidente", "fuego", "trafico_denso", "trafico_escaso"]

DETECTION_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign"
]

# ============================================================================
# DEFINICIÓN DE FASES Y MODOS
# ============================================================================

# Todas las opciones de combinación de fases disponibles
PHASE_OPTIONS = {
    # Una fase
    'F1':      {'phases': [1],       'gpus': 1, 'label': 'Solo Detección'},
    'F2':      {'phases': [2],       'gpus': 1, 'label': 'Solo Clasificación'},
    'F3':      {'phases': [3],       'gpus': 1, 'label': 'Solo VLM'},
    'F4':      {'phases': [4],       'gpus': 1, 'label': 'Solo LLM'},
    # Dos fases
    'F1F2':    {'phases': [1, 2],    'gpus': 2, 'label': 'Detección + Clasificación'},
    'F1F3':    {'phases': [1, 3],    'gpus': 2, 'label': 'Detección + VLM'},
    'F1F4':    {'phases': [1, 4],    'gpus': 2, 'label': 'Detección + LLM'},
    'F2F3':    {'phases': [2, 3],    'gpus': 2, 'label': 'Clasificación + VLM'},
    'F2F4':    {'phases': [2, 4],    'gpus': 2, 'label': 'Clasificación + LLM'},
    'F3F4':    {'phases': [3, 4],    'gpus': 2, 'label': 'VLM + LLM'},
    # Tres fases
    'F1F2F3':  {'phases': [1, 2, 3], 'gpus': 2, 'label': 'Det + Cls + VLM'},
    'F1F2F4':  {'phases': [1, 2, 4], 'gpus': 2, 'label': 'Det + Cls + LLM'},
    'F2F3F4':  {'phases': [2, 3, 4], 'gpus': 2, 'label': 'Cls + VLM + LLM'},
    # Cuatro fases
    'F1F2F3F4':{'phases': [1, 2, 3, 4], 'gpus': 2, 'label': 'Pipeline completo'},
}

# ============================================================================
# Re-usar todas las clases del script base importándolas por copia
# (se copian las clases necesarias del script base)
# ============================================================================

# --- Importar clases base desde diagnostic_validation_full_trace.py ---
_BASE_SCRIPT = Path(__file__).parent / "diagnostic_validation_full_trace.py"
if _BASE_SCRIPT.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_base", _BASE_SCRIPT)
    _base = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_base)

    GroundTruthLoader          = _base.GroundTruthLoader
    MetricsCalculator          = _base.MetricsCalculator
    TraceLogger                = _base.TraceLogger
    ResourceMonitor            = _base.ResourceMonitor
    ModelValidator             = _base.ModelValidator
    ImageVisualizer            = _base.ImageVisualizer
    DiagnosticDetectionValidator    = _base.DiagnosticDetectionValidator
    DiagnosticClassificationValidator = _base.DiagnosticClassificationValidator
    DiagnosticVLMValidator     = _base.DiagnosticVLMValidator
    DiagnosticLLMValidator     = _base.DiagnosticLLMValidator
    create_inceptionv3_model   = _base.create_inceptionv3_model
    create_resnet50_model      = _base.create_resnet50_model
    create_swin_model          = _base.create_swin_model
    create_vit_model           = _base.create_vit_model
    load_state_dict_with_dataparallel_fix = _base.load_state_dict_with_dataparallel_fix
else:
    raise RuntimeError(
        f"No se encontró diagnostic_validation_full_trace.py en {_BASE_SCRIPT.parent}\n"
        "Coloque fases_diagnostic_validation_full_trace.py en el mismo directorio."
    )


# ── Parche: extender DiagnosticLLMValidator para aceptar image_path directo ──
# Necesario para que F4 (solo LLM) funcione sin contexto de fases anteriores.
def _llm_validate_single_patched(
        self, det_result=None, cls_result=None, vlm_result=None,
        gt_loader=None, image_path=None):
    """validate_single extendido con parámetro image_path directo."""

    import time
    import torch
    from pathlib import Path as _Path

    self.logger.log(f"\n{'='*80}")
    self.logger.log(f"FASE: LLM ({self.model_type.upper()})")

    state_before = self.monitor.get_current_state()
    start_time   = time.time()

    result = {
        'phase': 'llm', 'model_type': self.model_type,
        'status': 'UNKNOWN', 'description': '', 'num_tokens': 0, 'latency_ms': 0
    }

    try:
        # Resolver image_path: parámetro directo → vlm_result → None
        if image_path is None and vlm_result:
            image_path = vlm_result.get('image_path')

        # Obtener GT LLM
        gt_data = None; instruction = None; gt_output = None; selected_type = None
        if gt_loader and image_path:
            gt_data = gt_loader.get_llm_analysis(image_path)

        # Construir input
        input_parts = []

        # F4 solo: inferir categoría desde la ruta y usar como contexto mínimo
        if not cls_result and not det_result and not vlm_result and image_path:
            img_name = _Path(image_path).name
            category = None
            for part in _Path(image_path).parts:
                if part in ('accidente', 'fuego', 'trafico_denso', 'trafico_escaso'):
                    category = part; break
            if category:
                input_parts.append(f"Traffic Scene Type: {category}")
            input_parts.append(f"Image: {img_name}")

        if cls_result and cls_result.get('predicted_class'):
            input_parts.append(f"Traffic Scene Type: {cls_result['predicted_class']}")

        if det_result and det_result.get('class_counts'):
            objects_list = []
            total_vehicles = 0
            for class_name, count in det_result['class_counts'].items():
                objects_list.append(f"{count} {class_name}")
                if class_name in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']:
                    total_vehicles += count
            input_parts.append(f"Detected Objects: {', '.join(objects_list)}")
            input_parts.append(f"Total Vehicles: {total_vehicles}")
            has_person  = 'person' in det_result['class_counts']
            has_signals = any(k in det_result['class_counts'] for k in ['traffic light', 'stop sign'])
            input_parts.append(f"People Present: {'Yes' if has_person else 'No'}")
            input_parts.append(f"Traffic Signals: {'Yes' if has_signals else 'No'}")

        if vlm_result and vlm_result.get('description'):
            vlm_desc = vlm_result['description']
            if "[/INST]" in vlm_desc:
                vlm_desc = vlm_desc.split("[/INST]")[-1].strip()
            elif "ASSISTANT:" in vlm_desc:
                vlm_desc = vlm_desc.split("ASSISTANT:")[-1].strip()
            input_parts.append(f"Visual Description: {vlm_desc}")

        llm_input = "\n".join(input_parts)

        if gt_data and isinstance(gt_data, dict):
            instruction   = gt_data.get('instruction', '')
            gt_output     = gt_data.get('output', '')
            selected_type = gt_data.get('type', '?')
            prompt = f"{instruction}\n\n{llm_input}\n\nResponse:"
            self.logger.log(f"  💭 Ejecutando LLM con instrucción del GT (Type {selected_type})...")
            self.logger.log(f"\n  📥 INSTRUCCIÓN (del dataset):\n     {instruction}\n")
            self.logger.log(f"  📋 INPUT CONSTRUIDO:")
            for line in input_parts:
                self.logger.log(f"     {line}")
            self.logger.log("")
        else:
            prompt = (f"Basándote en el siguiente análisis de una escena de tráfico, "
                      f"proporciona un resumen conciso y coherente:\n\n{llm_input}\n\nAnálisis final:")
            self.logger.log(f"  💭 Ejecutando LLM (sin GT disponible)...")
            self.logger.log(f"\n  📥 CONTEXTO INTEGRADO:\n")
            for line in input_parts:
                self.logger.log(f"     {line}")
            self.logger.log("")

        self.logger.log(f"  📝 PROMPT COMPLETO:")
        display = prompt[:400] + "...\n" if len(prompt) > 400 else prompt + "\n"
        self.logger.log(f"     {display}")

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            padding=True, truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            def _gen():
                return self.model.generate(
                    **inputs, max_new_tokens=150, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            outputs, _gpu_util_llm = self.monitor.sample_gpu_util_during(
                int(self.device.split(':')[1]) if ':' in self.device else 0, _gen
            )
            self._gpu_util_sampled = _gpu_util_llm

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Response:" in full_text:
            prediction_only = full_text.split("Response:")[-1].strip()
        elif "Análisis final:" in full_text:
            prediction_only = full_text.split("Análisis final:")[-1].strip()
        else:
            prediction_only = full_text.strip()

        result['full_output']      = full_text
        result['context_prompt']   = prompt
        result['description']      = prediction_only
        result['num_tokens']       = len(prediction_only.split())
        result['selected_type']    = selected_type if selected_type else 'N/A'
        result['image_path']       = str(image_path) if image_path else ''

        latency = (time.time() - start_time) * 1000
        result['latency_ms'] = latency
        result['status'] = 'SUCCESS'

        self.logger.log(f"  ✅ LLM exitoso", "SUCCESS")
        self.logger.log(f"\n  📋 OUTPUT COMPLETO ({result['num_tokens']} tokens):")
        if selected_type:
            self.logger.log(f"  🎲 Tipo seleccionado: {selected_type}")
        self.logger.log(f"\n  🎯 PREDICCIÓN (Análisis del modelo):")
        for line in prediction_only.split('\n')[:10]:
            self.logger.log(f"     {line}")
        self.logger.log(f"\n  ⏱️  Latencia: {latency:.2f} ms")

        # Métricas
        if gt_output:
            self.logger.log(f"\n📊 CÁLCULO DE MÉTRICAS - LLM")
            metrics = MetricsCalculator.calculate_text_metrics(
                hypothesis=prediction_only, reference=gt_output, metric_type='llm'
            )
            self.logger.log(f"\n  📈 Métricas LLM:")
            self.logger.log(f"     BLEU-1: {metrics.get('bleu_1', 0):.3f}")
            self.logger.log(f"     METEOR: {metrics.get('meteor', 0):.3f}")
            if 'rouge1_f' in metrics:
                self.logger.log(f"     ROUGE-1: {metrics['rouge1_f']:.3f}")
            result['metrics'] = metrics
        else:
            proxy = vlm_result.get('description') if vlm_result else None
            if proxy:
                self.logger.log(f"\n⚠️  GT LLM no disponible — usando VLM como proxy")
                metrics = MetricsCalculator.calculate_text_metrics(
                    hypothesis=prediction_only, reference=proxy, metric_type='llm'
                )
                metrics['_gt_source'] = 'vlm_proxy'
                result['metrics'] = metrics
            else:
                img_ref = str(image_path) if image_path else ''
                if img_ref:
                    self.logger.log(f"\n⚠️  Ground Truth LLM no disponible para: {img_ref}")

        state_after = self.monitor.log_resource_usage("LLM", state_before)
        vram_delta  = 0
        if state_after.get('gpus') and state_before.get('gpus'):
            for i, ga in enumerate(state_after['gpus']):
                if i < len(state_before['gpus']):
                    vram_delta += ga['vram_used_mb'] - state_before['gpus'][i]['vram_used_mb']
        result['model_vram_mb'] = self.model_vram_mb
        result['extra_vram_mb'] = max(0, vram_delta)
        gpu_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
        res_metrics = self.monitor.get_resource_metrics_for_phase(
            state_before, state_after, phase_gpu_idx=gpu_idx,
            latency_ms=result.get('latency_ms', 1), num_items=1,
            gpu_util_sampled=getattr(self, '_gpu_util_sampled', None)
        )
        result['phase_ram_gb'] = res_metrics['ram_delta_gb']
        result['throughput']   = res_metrics['throughput_img_s']
        result['flops']        = res_metrics['flops_approx']
        result['gpu_util']     = res_metrics['gpu_util_pct']
        result['cpu_util']     = res_metrics['cpu_util_pct']

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        result['latency_ms'] = latency
        result['status'] = 'FAILED'
        result['error']  = str(e)
        self.logger.log(f"  ❌ Error en LLM: {e}", "ERROR")

    return result

DiagnosticLLMValidator.validate_single = _llm_validate_single_patched


# ── Parche: proteger predicted_class en ImageVisualizer.visualize_and_save ──
_orig_visualize = ImageVisualizer.visualize_and_save
def _visualize_safe(self, image_path, predictions, output_name):
    if 'classification' in predictions and isinstance(predictions['classification'], dict):
        if 'predicted_class' not in predictions['classification']:
            predictions['classification']['predicted_class'] = \
                predictions['classification'].get('status', 'N/A')
    return _orig_visualize(self, image_path, predictions, output_name)
ImageVisualizer.visualize_and_save = _visualize_safe


# ============================================================================
# FASE SELECTOR — menú interactivo de fases y modelos
# ============================================================================

class FaseSelector:
    """Selección interactiva de fases y modelos según modo elegido."""

    # ── Selector genérico de ítems (imágenes o modelos) ──────────────────────
    @staticmethod
    def _pick(items: list, label: str, allow_all: bool = True) -> list:
        """Presentar lista y devolver selección del usuario."""
        print(f"\n{'─'*70}")
        print(f"  {label.upper()}  ({len(items)} disponibles)")
        print(f"{'─'*70}")
        for i, item in enumerate(items):
            name = item.name if hasattr(item, 'name') else str(item)
            print(f"  {i+1:3d}.  {name}")
        print(f"{'─'*70}")
        hint = "Enter=todos  " if allow_all else ""
        print(f"  Opciones: número | 1,3,5 | 2-5 | mixto | {hint}q=cancelar")

        while True:
            raw = input("  ➤ ").strip()
            if raw.lower() == 'q':
                return []
            if raw == '' and allow_all:
                return list(items)
            try:
                indices = set()
                for part in raw.split(','):
                    part = part.strip()
                    if '-' in part:
                        a, b = part.split('-', 1)
                        indices.update(range(int(a)-1, int(b)))
                    else:
                        indices.add(int(part)-1)
                chosen = [items[i] for i in sorted(indices) if 0 <= i < len(items)]
                if chosen:
                    return chosen
            except (ValueError, IndexError):
                pass
            print("  ❌ Entrada inválida, intente de nuevo.")

    # ── Selección de fase ─────────────────────────────────────────────────────
    @staticmethod
    def select_phase_mode() -> str:
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + "  SELECCIÓN DE MODO DE FASES".center(68) + "║")
        print("╚" + "═"*68 + "╝")

        groups = [
            ("UNA FASE (1 GPU)",
             ['F1','F2','F3','F4']),
            ("DOS FASES (2 GPUs)",
             ['F1F2','F1F3','F1F4','F2F3','F2F4','F3F4']),
            ("TRES FASES (2 GPUs)",
             ['F1F2F3','F1F2F4','F2F3F4']),
            ("CUATRO FASES / PIPELINE COMPLETO (2 GPUs)",
             ['F1F2F3F4']),
        ]

        options = []
        for group_name, keys in groups:
            print(f"\n  ── {group_name} ──")
            for key in keys:
                info = PHASE_OPTIONS[key]
                n = len(options) + 1
                options.append(key)
                print(f"  {n:2d}.  {key:<12}  {info['label']}")

        print()
        while True:
            raw = input("  ➤ Elegir opción (número o código, ej: 5 ó F1F2): ").strip().upper()
            if raw in PHASE_OPTIONS:
                return raw
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print("  ❌ Opción inválida.")

    # ── Selección de imágenes ─────────────────────────────────────────────────
    @classmethod
    def select_images(cls) -> list:
        print(f"\n{'─'*70}")
        print(f"  SELECCIÓN DE IMÁGENES  —  ¿Qué fuente desea usar?")
        print(f"{'─'*70}")
        print(f"  1.  Dataset plano     — {Path('/workspace/dataset_labeled_split/test/images')}")
        print(f"                          Selección libre de imágenes individuales.")
        print(f"  2.  Dataset por clase — /workspace/DSM/images/test/{{accidente,fuego,...}}")
        print(f"                          N imágenes por cada categoría automáticamente.")
        print(f"{'─'*70}")

        while True:
            raw = input("  ➤ Opción (1 / 2): ").strip()
            if raw in ("1", "2"):
                break
            print("  ❌ Ingrese 1 ó 2.")

        # ── Opción 1: comportamiento original ─────────────────────────────────
        if raw == "1":
            test_dir = Path("/workspace/dataset_labeled_split/test/images")
            images = sorted(
                list(test_dir.glob("*.jpg"))  +
                list(test_dir.glob("*.jpeg")) +
                list(test_dir.glob("*.png"))  +
                list(test_dir.glob("*.JPEG"))
            )
            if not images:
                print(f"  ❌ No hay imágenes en {test_dir}")
                return []
            return cls._pick(images, "IMÁGENES DE TEST (dataset plano)", allow_all=True)

        # ── Opción 2: N imágenes por categoría desde DSM ──────────────────────
        DSM_IMAGES = Path("/workspace/DSM/images/test")
        DSM_LABELS = Path("/workspace/DSM/labels/test")
        CATEGORIES = ["accidente", "fuego", "trafico_denso", "trafico_escaso"]
        IMG_EXTS   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

        # Mostrar cuántas imágenes hay por categoría
        print(f"\n{'─'*70}")
        print(f"  IMÁGENES DISPONIBLES POR CATEGORÍA (con label .txt)")
        print(f"{'─'*70}")
        cat_images: dict = {}
        for cat in CATEGORIES:
            img_dir = DSM_IMAGES / cat
            lbl_dir = DSM_LABELS / cat
            if not img_dir.exists():
                print(f"  ⚠️  {cat:<18} — directorio no encontrado: {img_dir}")
                cat_images[cat] = []
                continue
            # Solo imágenes que tienen su .txt correspondiente y NO está vacío
            all_imgs = []
            for ext in IMG_EXTS:
                all_imgs.extend(img_dir.glob(ext))
            valid = sorted([
                p for p in all_imgs
                if (lbl_dir / (p.stem + ".txt")).exists()
                and (lbl_dir / (p.stem + ".txt")).stat().st_size > 0
            ])
            cat_images[cat] = valid
            print(f"  {cat:<18} — {len(valid):>5} imágenes con label")
        print(f"{'─'*70}")

        # Pedir N
        while True:
            raw_n = input(
                "  ➤ ¿Cuántas imágenes por categoría? (número, Enter=todas): "
            ).strip()
            if raw_n == "":
                n_per_cat = None   # todas
                break
            try:
                n_per_cat = int(raw_n)
                if n_per_cat > 0:
                    break
                print("  ❌ Debe ser un número mayor que 0.")
            except ValueError:
                print("  ❌ Ingrese un número entero.")

        # Selección aleatoria o completa
        import random
        selected_images: list = []
        print(f"\n  Imágenes seleccionadas por categoría:")
        for cat in CATEGORIES:
            pool = cat_images[cat]
            if not pool:
                print(f"  ⚠️  {cat:<18} — sin imágenes disponibles, se omite.")
                continue
            if n_per_cat is None or n_per_cat >= len(pool):
                chosen = pool
            else:
                chosen = random.sample(pool, n_per_cat)
                chosen = sorted(chosen)
            print(f"  ✅ {cat:<18} — {len(chosen)} imágenes")
            selected_images.extend(chosen)

        if not selected_images:
            print("  ❌ No se encontraron imágenes válidas.")
            return []

        print(f"\n  Total imágenes seleccionadas: {len(selected_images)}")
        return selected_images

    # ── Selección de modelos según fases activas ──────────────────────────────
    @classmethod
    def select_models_for_phases(cls, available: dict, active_phases: list) -> dict:
        selected = {'detection': [], 'classification': [], 'vlm': [], 'llm': []}
        phase_map = {1: 'detection', 2: 'classification', 3: 'vlm', 4: 'llm'}
        phase_names = {1: 'DETECCIÓN (F1)', 2: 'CLASIFICACIÓN (F2)',
                       3: 'VLM (F3)', 4: 'LLM (F4)'}

        for ph in active_phases:
            key = phase_map[ph]
            models = available.get(key, [])
            if not models:
                print(f"⚠️  No se encontraron modelos para {phase_names[ph]}")
                continue
            chosen = cls._pick(models, phase_names[ph], allow_all=True)
            if not chosen:
                print(f"❌ No se seleccionaron modelos para {phase_names[ph]}, cancelando.")
                return {}
            selected[key] = chosen
        return selected


# ============================================================================
# GESTOR DE VRAM PARA VLM / LLM CON ESTRATEGIA BASE + LoRA
# ============================================================================

class LoRAVRAMManager:
    """
    Mantiene el modelo BASE permanente en GPU y carga/descarga solo el
    adaptador LoRA al cambiar de highcap a base o entre highcaps distintos.

    Estructura esperada en disco:
        /workspace/models/phase3_vlm/llavanext/
            base_model/          ← sin adapter_config.json → es base puro
            highcap_model/
                final_model/     ← adapter_config.json existe → LoRA adapter
    """

    def __init__(self, logger, phase: str):
        """
        phase: 'vlm' | 'llm'
        """
        self.logger  = logger
        self.phase   = phase           # 'vlm' o 'llm'
        self._base_cache: dict = {}    # {base_path_str: model_data}  permanente
        self._active_name: str = None  # nombre del modelo actualmente activo
        self._active_data: dict = None # model_data activo completo

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _is_lora(model_path: str) -> bool:
        p = Path(model_path) / "final_model" / "adapter_config.json"
        return p.exists()

    @staticmethod
    def _base_path_of(model_path: str) -> str:
        """Devuelve ruta del base_model hermano del highcap."""
        return str(Path(model_path).parent / "base_model")

    # ── API principal ──────────────────────────────────────────────────────────
    def get(self, model_info: dict, model_validator, device: str) -> dict:
        """
        Devuelve model_data listo para inferencia.
        Estrategia:
          - base_model puro  → cargar una vez en caché, reusar siempre
          - highcap (LoRA)   → cargar base (caché) + apply LoRA → merge_and_unload
                               Al cambiar a otro highcap: liberar merged, recargar base (ya en caché),
                               aplicar nuevo LoRA.
        """
        name = model_info['name']
        if name == self._active_name:
            self.logger.log(f"♻️  {self.phase.upper()} en caché: {name}")
            return self._active_data

        is_lora = self._is_lora(model_info['path'])

        if is_lora:
            base_path = self._base_path_of(model_info['path'])
            # Cargar base si no está en caché permanente
            if base_path not in self._base_cache:
                self.logger.log(f"🔄 Cargando base permanente {self.phase.upper()}: {Path(base_path).name}")
                base_info = dict(model_info, path=base_path,
                                 name=Path(base_path).name,
                                 type=model_info['type'])
                data = model_validator.load_and_validate_model(
                    base_path, model_info['type'], device,
                    architecture=model_info['type'])
                self._base_cache[base_path] = data
            else:
                self.logger.log(f"♻️  Base {self.phase.upper()} ya en caché permanente")

            base_data = self._base_cache[base_path]

            # Liberar LoRA merged anterior si era distinto
            if self._active_data is not None and self._active_name != name:
                self._free_merged(self._active_data)

            # Aplicar LoRA sobre copia del base (sin tocar la caché base)
            self.logger.log(f"🔧 Aplicando LoRA {self.phase.upper()}: {name}")
            merged_data = self._apply_lora(base_data, model_info, device)
            self._active_name = name
            self._active_data = merged_data

        else:
            # Modelo base puro: cachear permanente
            if model_info['path'] not in self._base_cache:
                self.logger.log(f"🔄 Cargando base {self.phase.upper()}: {name}")
                data = model_validator.load_and_validate_model(
                    model_info['path'], model_info['type'], device,
                    architecture=model_info['type'])
                self._base_cache[model_info['path']] = data
            # Liberar merged LoRA anterior
            if self._active_data is not None and self._active_name != name:
                if self._is_lora(self._active_data.get('_source_path', '')):
                    self._free_merged(self._active_data)
            self._active_name = name
            self._active_data = self._base_cache[model_info['path']]
            self.logger.log(f"♻️  Base {self.phase.upper()}: {name}")

        return self._active_data

    def _apply_lora(self, base_data: dict, model_info: dict, device: str) -> dict:
        """Tomar el modelo base en memoria y aplicar LoRA sobre él in-place."""
        import copy
        from peft import PeftModel

        base_model = base_data['model']
        adapter_path = str(Path(model_info['path']) / "final_model")

        # Aplicar LoRA y fusionar (no modifica el base original en caché)
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged = peft_model.merge_and_unload()
        merged.eval()

        result = dict(base_data)   # shallow copy con mismo processor/tokenizer
        result['model']        = merged
        result['status']       = 'LOADED'
        result['_source_path'] = model_info['path']
        result['_is_merged_lora'] = True
        return result

    def _free_merged(self, data: dict):
        """Liberar solo el modelo merged (no toca la caché base)."""
        if data and data.get('_is_merged_lora') and data.get('model') is not None:
            self.logger.log(f"🧹 Liberando LoRA merged anterior ({self.phase.upper()})")
            data['model'].cpu()
            del data['model']
            data['model'] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def free_all(self):
        """Liberar todo al terminar."""
        if self._active_data and self._active_data.get('_is_merged_lora'):
            self._free_merged(self._active_data)
        for path, data in self._base_cache.items():
            if data.get('model') is not None:
                self.logger.log(f"🧹 Liberando base {self.phase.upper()}: {Path(path).name}")
                data['model'].cpu()
                del data['model']
        self._base_cache.clear()
        self._active_data = None
        self._active_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# LOGGER CON PREFIJO 'fases_'
# ============================================================================

class FasesTraceLogger(TraceLogger):
    """Igual que TraceLogger pero con prefijo fases_ en nombres de archivo."""

    def __init__(self, output_dir: str, phase_key: str = ''):
        # Llamar __init__ del padre pero sobreescribir nombres
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"_{phase_key}" if phase_key else ''
        self.log_file  = self.output_dir / f"fases_trace_log{prefix}_{ts}.txt"
        self.csv_file  = self.output_dir / f"res{prefix}_{ts}.csv"
        self.image_dir = self.output_dir / "processed_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self._buffer    = []
        self._FLUSH_SIZE = 100


# ============================================================================
# VALIDADOR POR FASES
# ============================================================================

class FasesDiagnosticValidator:
    """
    Validador que ejecuta solo las fases seleccionadas,
    con soporte para estrategia LoRA VRAM y flush CSV adaptativo.
    """

    # ── CSV: todos los campos del script original (vacíos para fases no activas)
    CSV_HEADERS = [
        'combination_id','combination_name',
        'det_model','cls_model','cls_architecture','vlm_model','llm_model',
        'image_path','image_name',
        # F1
        'det_status','det_objects',
        'det_precision','det_recall','det_f1',
        'det_map_50','det_map_50_95','det_iou',
        'det_latency_ms','det_model_vram_mb','det_extra_vram_mb',
        'det_ram_gb','det_throughput','det_flops','det_gpu_util','det_cpu_util',
        # F2
        'cls_status','cls_predicted','cls_confidence',
        'cls_accuracy','cls_precision','cls_recall','cls_f1',
        'cls_latency_ms','cls_model_vram_mb','cls_extra_vram_mb',
        'cls_ram_gb','cls_throughput','cls_flops','cls_gpu_util','cls_cpu_util',
        # F3
        'vlm_status','vlm_tokens',
        'vlm_bleu1','vlm_bleu2','vlm_bleu3','vlm_bleu4','vlm_meteor',
        'vlm_rouge1_f','vlm_rouge2_f','vlm_rougeL_f',
        'vlm_perplexity','vlm_bertscore',
        'vlm_cider','vlm_spice','vlm_clipscore',
        'vlm_latency_ms','vlm_model_vram_mb','vlm_extra_vram_mb',
        'vlm_ram_gb','vlm_throughput','vlm_flops','vlm_gpu_util','vlm_cpu_util',
        # F4
        'llm_status','llm_tokens','llm_type',
        'llm_bleu1','llm_bleu2','llm_bleu3','llm_bleu4','llm_meteor',
        'llm_rouge1_f','llm_rouge2_f','llm_rougeL_f',
        'llm_perplexity','llm_bertscore','llm_mmlu','llm_mauve',
        'llm_latency_ms','llm_model_vram_mb','llm_extra_vram_mb',
        'llm_ram_gb','llm_throughput','llm_flops','llm_gpu_util','llm_cpu_util',
        # Totales
        'total_latency_ms','vram_used_mb','ram_used_gb','timestamp',
    ]

    # Fila vacía por defecto (campos no activos se llenan con '')
    _EMPTY = {h: '' for h in CSV_HEADERS}

    def __init__(self, phase_key: str, selected_models: dict,
                 image_paths: list, output_dir: str):

        self.phase_key    = phase_key
        self.phase_info   = PHASE_OPTIONS[phase_key]
        self.active_phases = self.phase_info['phases']
        self.image_paths  = image_paths
        self.output_dir   = output_dir

        # Logger y monitor
        self.logger  = FasesTraceLogger(output_dir, phase_key=phase_key)
        self.monitor = ResourceMonitor(self.logger)
        self.model_validator = ModelValidator(self.logger, self.monitor)
        self.visualizer = ImageVisualizer(self.logger, self.logger.image_dir)

        # GPU assignment
        self.gpu_assignment = self._assign_gpus()

        # Ground truth
        try:
            self.gt_loader = GroundTruthLoader(self.logger)
        except Exception as e:
            self.logger.log(f"⚠️  GT loader no disponible: {e}", "WARNING")
            self.gt_loader = None

        # Modelos seleccionados
        self.sel = selected_models  # dict con listas por fase

        # Caché ligeros (det, cls)
        self._det_cache  : dict = {}
        self._cls_cache  : dict = {}

        # Gestores LoRA VRAM para VLM y LLM
        self._vlm_mgr = LoRAVRAMManager(self.logger, 'vlm') if 3 in self.active_phases else None
        self._llm_mgr = LoRAVRAMManager(self.logger, 'llm') if 4 in self.active_phases else None

        # CSV state
        self.csv_rows          = []
        self._pending          = []
        self._header_written   = False
        self._flush_every      = self._compute_flush_every()

    # ── GPU assignment ─────────────────────────────────────────────────────────
    def _assign_gpus(self) -> dict:
        n = torch.cuda.device_count()
        req = self.phase_info['gpus']
        self.logger.log(f"\n🔍 GPUs disponibles: {n} | Requeridas por modo {self.phase_key}: {req}")

        if n == 0:
            raise RuntimeError("Se requiere al menos 1 GPU.")

        # Modo una fase → todo en GPU 0
        if req == 1 or n == 1:
            if req == 2 and n == 1:
                self.logger.log("⚠️  Solo 1 GPU disponible, usando GPU 0 para todo.", "WARNING")
            return {'detection': 'cuda:0', 'classification': 'cuda:0',
                    'vlm': 'cuda:0', 'llm': 'cuda:0'}

        # 2+ GPUs → det/cls en GPU 0, VLM+LLM en GPU 1
        assignment = {'detection': 'cuda:0', 'classification': 'cuda:0',
                      'vlm': 'cuda:1', 'llm': 'cuda:1'}
        if n >= 3:
            assignment['llm'] = 'cuda:2'
            self.logger.log("✅ Det+Cls→GPU0, VLM→GPU1, LLM→GPU2")
        else:
            self.logger.log("✅ Det+Cls→GPU0, VLM+LLM→GPU1")
        return assignment

    # ── Flush cadencia ─────────────────────────────────────────────────────────
    def _compute_flush_every(self) -> int:
        n = len(self.active_phases)
        if n == 1:   return 1   # flush inmediato tras cada ejecución
        if n <= 3:   return 5   # dos/tres fases
        return 10                # cuatro fases

    # ── Carga modelos ligeros ──────────────────────────────────────────────────
    def _get_det(self, model_info: dict):
        name = model_info['name']
        if name not in self._det_cache:
            self.logger.log(f"🔄 Cargando Detection: {name}")
            data = self.model_validator.load_and_validate_model(
                model_info['path'], 'yolo11',
                self.gpu_assignment['detection'], architecture='yolo11')
            data['_vram_at_load'] = data.get('vram_used_mb', 0)
            self._det_cache[name] = data
        else:
            self.logger.log(f"♻️  Detection caché: {name}")
        d = self._det_cache[name]
        d['vram_used_mb'] = d.get('_vram_at_load', 0)
        return d

    def _get_cls(self, model_info: dict):
        name = model_info['name']
        if name not in self._cls_cache:
            self.logger.log(f"🔄 Cargando Classification: {name}")
            data = self.model_validator.load_and_validate_model(
                model_info['path'], model_info['type'],
                self.gpu_assignment['classification'],
                architecture=model_info.get('architecture', model_info['type']))
            data['_vram_at_load'] = data.get('vram_used_mb', 0)
            self._cls_cache[name] = data
        else:
            self.logger.log(f"♻️  Classification caché: {name}")
        d = self._cls_cache[name]
        d['vram_used_mb'] = d.get('_vram_at_load', 0)
        return d

    # ── Flush CSV ──────────────────────────────────────────────────────────────
    def _flush(self):
        if not self._pending:
            return
        mode = 'a' if self._header_written else 'w'
        with open(self.logger.csv_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self._pending)
        self._pending.clear()

    def _append_row(self, row: dict):
        full = dict(self._EMPTY)
        full.update(row)
        self.csv_rows.append(full)
        self._pending.append(full)
        if self._flush_every == 1:
            self._flush()

    # ── Construcción de CSV row vacío por defecto ──────────────────────────────
    @staticmethod
    def _empty_result(status='N/A') -> dict:
        return {'status': status, 'latency_ms': 0}

    # ── Ejecutar UNA combinación sobre TODAS las imágenes ─────────────────────
    def _run_combination(self, combo_id: int, combo: dict):
        """
        combo = {
            'det': model_info | None,
            'cls': model_info | None,
            'vlm': model_info | None,
            'llm': model_info | None,
        }
        """
        active = self.active_phases

        # Nombre de combinación con solo fases activas
        parts = []
        if 1 in active and combo.get('det'): parts.append(combo['det']['name'])
        if 2 in active and combo.get('cls'): parts.append(combo['cls']['name'])
        if 3 in active and combo.get('vlm'): parts.append(combo['vlm']['name'])
        if 4 in active and combo.get('llm'): parts.append(combo['llm']['name'])
        combo_name = '-'.join(parts)

        self.logger.log_section(f"COMBINACIÓN {combo_id}: {combo_name}")

        # Cargar modelos necesarios
        det_data = self._get_det(combo['det']) if 1 in active else None
        cls_data = self._get_cls(combo['cls']) if 2 in active else None
        vlm_data = (self._vlm_mgr.get(combo['vlm'], self.model_validator,
                                       self.gpu_assignment['vlm'])
                    if 3 in active else None)
        llm_data = (self._llm_mgr.get(combo['llm'], self.model_validator,
                                       self.gpu_assignment['llm'])
                    if 4 in active else None)

        # Verificar cargas exitosas
        for label, data in [('Det', det_data), ('Cls', cls_data),
                             ('VLM', vlm_data), ('LLM', llm_data)]:
            if data is not None and data.get('status') != 'LOADED':
                self.logger.log(f"❌ {label} no cargado — combinación saltada", "ERROR")
                return

        # Crear validadores
        det_val = (DiagnosticDetectionValidator(det_data, self.logger, self.monitor)
                   if det_data else None)
        cls_val = (DiagnosticClassificationValidator(
                       cls_data, self.logger, self.monitor,
                       combo['cls'].get('architecture', combo['cls']['type']))
                   if cls_data else None)
        vlm_val = (DiagnosticVLMValidator(vlm_data, self.logger, self.monitor,
                                          combo['vlm']['type'])
                   if vlm_data else None)
        llm_val = (DiagnosticLLMValidator(llm_data, self.logger, self.monitor,
                                          combo['llm']['type'])
                   if llm_data else None)

        # Procesar cada imagen
        for img_idx, image_path in enumerate(self.image_paths):
            self.logger.log_section(f"IMG {img_idx+1}/{len(self.image_paths)}: {image_path.name}")

            t0      = time.time()
            st_0    = self.monitor.get_current_state()

            # ── Resultados vacíos por defecto ──────────────────────────────────
            det_result = {'status': 'N/A', 'latency_ms': 0, 'num_objects': 0}
            cls_result = {'status': 'N/A', 'latency_ms': 0}
            vlm_result = {'status': 'N/A', 'latency_ms': 0, 'description': '',
                          'num_tokens': 0}
            llm_result = {'status': 'N/A', 'latency_ms': 0}

            # ── FASE 1 ─────────────────────────────────────────────────────────
            if det_val:
                det_result = det_val.validate_single(image_path, gt_loader=self.gt_loader)

            # ── FASE 2 ─────────────────────────────────────────────────────────
            if cls_val:
                cls_result = cls_val.validate_single(image_path, gt_loader=self.gt_loader)

            # ── FASE 3 — VLM recibe contexto de fases activas previas ──────────
            if vlm_val:
                # Pasar det y cls solo si esas fases están activas
                vlm_result = vlm_val.validate_single(
                    image_path,
                    det_result=det_result if 1 in active else None,
                    cls_result=cls_result if 2 in active else None,
                    gt_loader=self.gt_loader
                )

            # ── FASE 4 — LLM recibe contexto de todas las fases activas previas
            if llm_val:
                if vlm_result.get('status') == 'SUCCESS' or 3 not in active:
                    llm_result = llm_val.validate_single(
                        det_result=det_result if 1 in active else None,
                        cls_result=cls_result if 2 in active else None,
                        vlm_result=vlm_result if 3 in active else None,
                        gt_loader=self.gt_loader,
                        image_path=image_path,
                    )
                else:
                    self.logger.log("⚠️  LLM saltado (VLM falló)", "WARNING")

            # ── Métricas totales ───────────────────────────────────────────────
            total_ms = (time.time() - t0) * 1000
            st_1 = self.monitor.get_current_state()

            vram_used = 0
            if st_1['gpus']:
                for i, gpu in enumerate(st_1['gpus']):
                    if i < len(st_0['gpus']):
                        vram_used += gpu['vram_used_mb'] - st_0['gpus'][i]['vram_used_mb']

            ram_delta = max(0.0, st_1.get('ram_used_gb', 0) - st_0.get('ram_used_gb', 0))

            # ── Visualización ──────────────────────────────────────────────────
            viz_name = f"combo_{combo_id:03d}_img_{img_idx:03d}_{image_path.stem}.jpg"
            self.visualizer.visualize_and_save(image_path, {
                'detection': det_result,
                'classification': cls_result,
                'vlm': vlm_result,
                'llm': llm_result,
            }, viz_name)

            # ── CSV row ────────────────────────────────────────────────────────
            row = {
                'combination_id':   combo_id,
                'combination_name': combo_name,
                'det_model':   combo['det']['name']  if combo.get('det')  else '',
                'cls_model':   combo['cls']['name']  if combo.get('cls')  else '',
                'cls_architecture': combo['cls'].get('architecture', combo['cls']['type'])
                                    if combo.get('cls') else '',
                'vlm_model':   combo['vlm']['name']  if combo.get('vlm')  else '',
                'llm_model':   combo['llm']['name']  if combo.get('llm')  else '',
                'image_path':  str(image_path),
                'image_name':  image_path.name,

                # F1
                'det_status':   det_result.get('status', ''),
                'det_objects':  det_result.get('num_objects', ''),
                'det_precision':  det_result.get('metrics', {}).get('precision', ''),
                'det_recall':     det_result.get('metrics', {}).get('recall', ''),
                'det_f1':         det_result.get('metrics', {}).get('f1', ''),
                'det_map_50':     det_result.get('metrics', {}).get('map_50', ''),
                'det_map_50_95':  det_result.get('metrics', {}).get('map_50_95', ''),
                'det_iou':        det_result.get('metrics', {}).get('iou', ''),
                'det_latency_ms':     det_result.get('latency_ms', ''),
                'det_model_vram_mb':  det_result.get('model_vram_mb', ''),
                'det_extra_vram_mb':  det_result.get('extra_vram_mb', ''),
                'det_ram_gb':         det_result.get('phase_ram_gb', ''),
                'det_throughput':     det_result.get('throughput', ''),
                'det_flops':          det_result.get('flops', ''),
                'det_gpu_util':       det_result.get('gpu_util', ''),
                'det_cpu_util':       det_result.get('cpu_util', ''),

                # F2
                'cls_status':    cls_result.get('status', ''),
                'cls_predicted': cls_result.get('predicted_class', ''),
                'cls_confidence':cls_result.get('confidence', ''),
                'cls_accuracy':  cls_result.get('metrics', {}).get('accuracy', ''),
                'cls_precision': cls_result.get('metrics', {}).get('precision', ''),
                'cls_recall':    cls_result.get('metrics', {}).get('recall', ''),
                'cls_f1':        cls_result.get('metrics', {}).get('f1', ''),
                'cls_latency_ms':    cls_result.get('latency_ms', ''),
                'cls_model_vram_mb': cls_result.get('model_vram_mb', ''),
                'cls_extra_vram_mb': cls_result.get('extra_vram_mb', ''),
                'cls_ram_gb':        cls_result.get('phase_ram_gb', ''),
                'cls_throughput':    cls_result.get('throughput', ''),
                'cls_flops':         cls_result.get('flops', ''),
                'cls_gpu_util':      cls_result.get('gpu_util', ''),
                'cls_cpu_util':      cls_result.get('cpu_util', ''),

                # F3
                'vlm_status':   vlm_result.get('status', ''),
                'vlm_tokens':   vlm_result.get('num_tokens', ''),
                'vlm_bleu1':    vlm_result.get('metrics', {}).get('bleu_1', ''),
                'vlm_bleu2':    vlm_result.get('metrics', {}).get('bleu_2', ''),
                'vlm_bleu3':    vlm_result.get('metrics', {}).get('bleu_3', ''),
                'vlm_bleu4':    vlm_result.get('metrics', {}).get('bleu_4', ''),
                'vlm_meteor':   vlm_result.get('metrics', {}).get('meteor', ''),
                'vlm_rouge1_f': vlm_result.get('metrics', {}).get('rouge1_f', ''),
                'vlm_rouge2_f': vlm_result.get('metrics', {}).get('rouge2_f', ''),
                'vlm_rougeL_f': vlm_result.get('metrics', {}).get('rougeL_f', ''),
                'vlm_perplexity':vlm_result.get('metrics', {}).get('perplexity', ''),
                'vlm_bertscore': vlm_result.get('metrics', {}).get('bertscore_f1', ''),
                'vlm_cider':     vlm_result.get('metrics', {}).get('cider', ''),
                'vlm_spice':     vlm_result.get('metrics', {}).get('spice', ''),
                'vlm_clipscore': vlm_result.get('metrics', {}).get('clipscore', ''),
                'vlm_latency_ms':    vlm_result.get('latency_ms', ''),
                'vlm_model_vram_mb': vlm_result.get('model_vram_mb', ''),
                'vlm_extra_vram_mb': vlm_result.get('extra_vram_mb', ''),
                'vlm_ram_gb':        vlm_result.get('phase_ram_gb', ''),
                'vlm_throughput':    vlm_result.get('throughput', ''),
                'vlm_flops':         vlm_result.get('flops', ''),
                'vlm_gpu_util':      vlm_result.get('gpu_util', ''),
                'vlm_cpu_util':      vlm_result.get('cpu_util', ''),

                # F4
                'llm_status':   llm_result.get('status', ''),
                'llm_tokens':   llm_result.get('num_tokens', ''),
                'llm_type':     llm_result.get('selected_type', ''),
                'llm_bleu1':    llm_result.get('metrics', {}).get('bleu_1', ''),
                'llm_bleu2':    llm_result.get('metrics', {}).get('bleu_2', ''),
                'llm_bleu3':    llm_result.get('metrics', {}).get('bleu_3', ''),
                'llm_bleu4':    llm_result.get('metrics', {}).get('bleu_4', ''),
                'llm_meteor':   llm_result.get('metrics', {}).get('meteor', ''),
                'llm_rouge1_f': llm_result.get('metrics', {}).get('rouge1_f', ''),
                'llm_rouge2_f': llm_result.get('metrics', {}).get('rouge2_f', ''),
                'llm_rougeL_f': llm_result.get('metrics', {}).get('rougeL_f', ''),
                'llm_perplexity':llm_result.get('metrics', {}).get('perplexity', ''),
                'llm_bertscore': llm_result.get('metrics', {}).get('bertscore_f1', ''),
                'llm_mmlu':      llm_result.get('metrics', {}).get('mmlu', ''),
                'llm_mauve':     llm_result.get('metrics', {}).get('mauve', ''),
                'llm_latency_ms':    llm_result.get('latency_ms', ''),
                'llm_model_vram_mb': llm_result.get('model_vram_mb', ''),
                'llm_extra_vram_mb': llm_result.get('extra_vram_mb', ''),
                'llm_ram_gb':        llm_result.get('phase_ram_gb', ''),
                'llm_throughput':    llm_result.get('throughput', ''),
                'llm_flops':         llm_result.get('flops', ''),
                'llm_gpu_util':      llm_result.get('gpu_util', ''),
                'llm_cpu_util':      llm_result.get('cpu_util', ''),

                # Totales
                'total_latency_ms': total_ms,
                'vram_used_mb':     vram_used,
                'ram_used_gb':      ram_delta,
                'timestamp':        datetime.now().isoformat(),
            }
            self._append_row(row)

    # ── Generar todas las combinaciones según fases activas ────────────────────
    def _build_combinations(self) -> list:
        """Generar lista de combos con orden optimizado (modelos pesados cambian menos)."""
        active = self.active_phases
        det_list = self.sel.get('detection', [None]) if 1 in active else [None]
        cls_list = self.sel.get('classification', [None]) if 2 in active else [None]
        vlm_list = self.sel.get('vlm', [None]) if 3 in active else [None]
        llm_list = self.sel.get('llm', [None]) if 4 in active else [None]

        combos = []
        # Orden optimizado: modelos más costosos (VLM, LLM) en outer loop
        for vlm in vlm_list:
            for llm in llm_list:
                for det in det_list:
                    for cls in cls_list:
                        combos.append({'det': det, 'cls': cls, 'vlm': vlm, 'llm': llm})
        return combos

    # ── Punto de entrada principal ─────────────────────────────────────────────
    def run(self):
        self.logger.log_section(f"INICIO — MODO {self.phase_key}: {self.phase_info['label']}")
        self.logger.log(f"Fases activas: {self.active_phases}")
        self.logger.log(f"Imágenes: {len(self.image_paths)}")

        combos = self._build_combinations()
        total  = len(combos)
        imgs   = len(self.image_paths)
        self.logger.log(f"Combinaciones: {total}  ×  {imgs} imágenes = {total*imgs} ejecuciones")

        for idx, combo in enumerate(combos, start=1):
            try:
                self._run_combination(idx, combo)
            except Exception as e:
                self.logger.log(f"❌ Error combinación {idx}: {e}", "ERROR")
                import traceback
                self.logger.log(traceback.format_exc(), "ERROR")
                continue

            # Flush periódico (para una fase ya es inmediato por _flush_every=1)
            if self._flush_every > 1 and idx % self._flush_every == 0:
                self._flush()
                self.logger.log(f"💾 CSV guardado parcialmente ({len(self.csv_rows)} filas)")

        # Liberación final
        self._cleanup()

        # Flush final
        self._flush()

        # Resumen
        elapsed = time.time() - self.logger.start_time
        self.logger.log_section("COMPLETADO")
        self.logger.log(f"✅ Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)", "SUCCESS")
        self.logger.log(f"📊 Total filas CSV: {len(self.csv_rows)}")
        self.logger.log(f"   Log: {self.logger.log_file}")
        self.logger.log(f"   CSV: {self.logger.csv_file}")
        self.logger._flush()

    # ── Limpieza ───────────────────────────────────────────────────────────────
    def _cleanup(self):
        self.logger.log("\n🧹 Limpieza de VRAM...")
        # Det caché
        for name, data in list(self._det_cache.items()):
            if data.get('model') is not None:
                data['model'].cpu()
                del data['model']
        self._det_cache.clear()
        # Cls caché
        for name, data in list(self._cls_cache.items()):
            if data.get('model') is not None:
                data['model'].cpu()
                del data['model']
        self._cls_cache.clear()
        # VLM y LLM managers
        if self._vlm_mgr: self._vlm_mgr.free_all()
        if self._llm_mgr: self._llm_mgr.free_all()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log("✅ VRAM liberada.")


# ============================================================================
# DESCUBRIMIENTO DE MODELOS (reutiliza lógica del script base)
# ============================================================================

def discover_models(logger) -> dict:
    """Descubrir todos los modelos disponibles en /workspace/models."""
    base = Path("/workspace/models")
    models = {'detection': [], 'classification': [], 'vlm': [], 'llm': []}

    # Detection
    det_dir = base / "phase1_detection" / "production_det"
    if det_dir.exists():
        for combo in sorted(det_dir.glob("combo_*")):
            pt = combo / "weights" / "best.pt"
            if pt.exists():
                models['detection'].append({
                    'name': combo.name, 'path': str(pt), 'type': 'yolo11'
                })

    # Classification
    cls_base = base / "phase2_classification"
    arch_map = {
        'production_cls':      ('yolo11',     'yolo11'),
        'inceptionv3':         ('inceptionv3','inceptionv3'),
        'resnet50':            ('resnet50',   'resnet50'),
        'swin':                ('swin',       'swin'),
        'vit':                 ('vit',        'vit'),
    }
    for arch_dir, (type_key, arch_key) in arch_map.items():
        prod_name = 'production_' + (arch_dir if arch_dir != 'production_cls' else 'cls')
        for sub in sorted(cls_base.glob(f"{arch_dir}/*")):
            if not sub.is_dir(): continue
            # buscar en production_*
            for prod in sub.glob("production_*"):
                for combo in sorted(prod.glob("combo_*")):
                    pt = combo / "weights" / "best.pt"
                    if pt.exists():
                        models['classification'].append({
                            'name': f"{arch_dir}_{combo.name}",
                            'path': str(pt),
                            'type': type_key,
                            'architecture': arch_key
                        })
        # alternativa: arch_dir/combo_*/weights/best.pt directamente
        for combo in sorted(cls_base.glob(f"{arch_dir}/combo_*/weights/best.pt")):
            models['classification'].append({
                'name': f"{arch_dir}_{combo.parent.parent.name}",
                'path': str(combo),
                'type': type_key,
                'architecture': arch_key
            })

    # Dedup classification
    seen = set()
    models['classification'] = [m for m in models['classification']
                                  if m['name'] not in seen and not seen.add(m['name'])]

    # VLM
    vlm_base = base / "phase3_vlm"
    for arch in ['llavanext', 'qwen2vl']:
        arch_dir = vlm_base / arch
        if not arch_dir.exists(): continue
        vlm_type = 'llava' if arch == 'llavanext' else 'qwen2vl'
        for variant in ['base_model', 'highcap_model']:
            vp = arch_dir / variant
            if vp.exists():
                models['vlm'].append({
                    'name': f"{arch}_{variant.replace('_model','')}",
                    'path': str(vp), 'type': vlm_type
                })

    # LLM
    llm_base = base / "phase4_llm"
    for arch in ['mistral-7b', 'qwen2-7b']:
        arch_dir = llm_base / arch
        if not arch_dir.exists(): continue
        llm_type = 'mistral' if 'mistral' in arch else 'qwen2'
        for variant in ['base_model', 'highcap_model']:
            lp = arch_dir / variant
            if lp.exists():
                models['llm'].append({
                    'name': f"{arch.replace('-7b','')}_{variant.replace('_model','')}",
                    'path': str(lp), 'type': llm_type
                })

    logger.log(f"Modelos descubiertos: det={len(models['detection'])} "
               f"cls={len(models['classification'])} "
               f"vlm={len(models['vlm'])} llm={len(models['llm'])}", "SUCCESS")
    return models


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico por fases seleccionables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python fases_diagnostic_validation_full_trace.py
  python fases_diagnostic_validation_full_trace.py --phase F1F2
  python fases_diagnostic_validation_full_trace.py --phase F1F2F3F4 --output-dir /workspace/results
        """
    )
    parser.add_argument('--phase', type=str, default=None,
                        help='Modo de fases (ej: F1, F1F2, F1F2F3F4). Si no se indica, menú interactivo.')
    parser.add_argument('--output-dir', type=str,
                        default='/workspace/diagnostic_results',
                        help='Directorio de salida')
    args = parser.parse_args()

    print("\n" + "═"*70)
    print("  DIAGNÓSTICO POR FASES SELECCIONABLES")
    print("═"*70)

    # Crear logger temporal para descubrimiento
    tmp_logger = FasesTraceLogger(args.output_dir)
    available  = discover_models(tmp_logger)

    # Selección de fases
    if args.phase and args.phase.upper() in PHASE_OPTIONS:
        phase_key = args.phase.upper()
        print(f"\n✅ Modo de fase: {phase_key} — {PHASE_OPTIONS[phase_key]['label']}")
    else:
        phase_key = FaseSelector.select_phase_mode()

    phase_info = PHASE_OPTIONS[phase_key]
    print(f"\n📋 Fases activas: {phase_info['phases']}  |  GPUs recomendadas: {phase_info['gpus']}")

    # Selección de imágenes
    images = FaseSelector.select_images()
    if not images:
        print("❌ No se seleccionaron imágenes.")
        return

    # Selección de modelos según fases
    selected = FaseSelector.select_models_for_phases(available, phase_info['phases'])
    if not selected:
        print("❌ Selección de modelos cancelada.")
        return

    # Resumen
    print("\n" + "═"*70)
    print("  RESUMEN DE SELECCIÓN")
    print("═"*70)
    print(f"  Modo:     {phase_key} — {phase_info['label']}")
    print(f"  Imágenes: {len(images)}")
    for ph, key in [(1,'detection'),(2,'classification'),(3,'vlm'),(4,'llm')]:
        if ph in phase_info['phases']:
            names = [m['name'] for m in selected.get(key, [])]
            print(f"  F{ph} ({key}): {names}")

    # Calcular total
    counts = [max(1, len(selected.get(k, [])))
              for ph, k in [(1,'detection'),(2,'classification'),(3,'vlm'),(4,'llm')]
              if ph in phase_info['phases']]
    total = 1
    for c in counts: total *= c
    print(f"\n  Combinaciones: {total}  ×  {len(images)} imágenes = {total*len(images)} ejecuciones")
    print("═"*70)

    confirm = input("\n  ¿Continuar? (s/n): ").strip().lower()
    if confirm != 's':
        print("❌ Cancelado.")
        return

    # Crear y ejecutar validador
    validator = FasesDiagnosticValidator(
        phase_key=phase_key,
        selected_models=selected,
        image_paths=images,
        output_dir=args.output_dir,
    )
    validator.run()


if __name__ == "__main__":
    main()
