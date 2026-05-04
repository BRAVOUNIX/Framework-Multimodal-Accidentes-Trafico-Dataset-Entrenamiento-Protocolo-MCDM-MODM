"""
FASE 1: Grid Search Optimizado para YOLO11
===========================================
Plataforma: RunPod
Búsqueda exhaustiva de hiperparámetros con análisis completo
"""

import argparse
import sys
import json
import yaml
import time
import psutil
import GPUtil
import logging
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging(experiment_name):
    """Configura logging con archivo y consola"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"phase1_grid_search_{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('GridSearch')
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file


# ============================================================================
# CONFIGURACIÓN BASE
# ============================================================================

class Config:
    """Configuración del grid search"""
    
    # Paths
    BASE_DIR = Path("/workspace")
    DATASET_DIR = BASE_DIR / "DSM"
    IMAGES_DIR = DATASET_DIR / "images"
    LABELS_DIR = DATASET_DIR / "labels"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models/phase1_detection"
    LOGS_DIR = BASE_DIR / "logs/phase1_detection"
    
    # Dataset
    DATA_YAML = OUTPUT_DIR / "detection_data.yaml"
    
    # Modelos disponibles
    AVAILABLE_MODELS = {
        'YOLOv11n': 'yolo11n.pt',
        'YOLOv11s': 'yolo11s.pt',
        'YOLOv11m': 'yolo11m.pt',
        'YOLOv11l': 'yolo11l.pt',
        'YOLOv11x': 'yolo11x.pt'
    }
    
    # Hiperparámetros por defecto
    DEFAULT_HYPERPARAMS = {
        'imgsz': [640],
        'lr0': [0.01],
        'lrf': [0.01],
        'momentum': [0.937],
        'weight_decay': [0.0005],
        'warmup_epochs': [3.0],
        'warmup_momentum': [0.8],
        'warmup_bias_lr': [0.1],
        'box': [7.5],
        'cls': [0.5],
        'dfl': [1.5],
        'close_mosaic': [10],
        'amp': [True],
        'fraction': [1.0],
        'device': [0]
    }
    
    DEFAULT_IMAGE_LIMIT = 20


# ============================================================================
# UTILIDADES
# ============================================================================

class ResourceMonitor:
    """Monitor de recursos computacionales"""
    
    def __init__(self):
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()
        
    def start(self):
        self.start_time = time.time()
        
    def get_metrics(self):
        """Obtiene métricas de recursos"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
            'gpu_available': self.gpu_available
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({
                        'gpu_name': gpu.name,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'gpu_utilization_percent': gpu.load * 100,
                        'gpu_temperature_c': gpu.temperature
                    })
            except:
                pass
        
        return metrics


def save_json(data, path):
    """Guarda datos en JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_hyperparam_list(value_str, param_name=None):
    """
    Parsea string de hiperparámetros separados por coma
    
    Manejo especial para device:
        device="0,1" -> [[0,1]] (una sola opción: multi-GPU)
        device="0"   -> [0]     (una opción: GPU 0)
    
    Otros parámetros:
        "25,100" -> [25, 100]
        "0.01,0.02" -> [0.01, 0.02]
        "SGD,Adam" -> ["SGD", "Adam"]
        "True,False" -> [True, False]
    """
    # Manejo especial para device con múltiples GPUs
    if param_name == 'device' and ',' in value_str:
        # device="0,1" se interpreta como [0,1] (lista de GPUs)
        # Retornar como una sola opción envuelta en lista
        gpu_list = []
        for v in value_str.split(','):
            v = v.strip()
            try:
                gpu_list.append(int(v))
            except:
                gpu_list.append(v)
        # Retornar como UNA opción (lista dentro de lista)
        return [gpu_list]
    
    # Procesamiento normal para otros parámetros
    values = []
    for v in value_str.split(','):
        v = v.strip()
        
        # Boolean
        if v.lower() in ['true', 'false']:
            values.append(v.lower() == 'true')
        # Float
        elif '.' in v:
            try:
                values.append(float(v))
            except:
                values.append(v)
        # Int
        else:
            try:
                values.append(int(v))
            except:
                values.append(v)
    
    return values


# ============================================================================
# GENERADOR DE DATASET LIMITADO
# ============================================================================

class DatasetManager:
    """Gestiona creación de datasets limitados"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def create_limited_yaml(self, image_limit):
        """Crea YAML con dataset limitado usando directorio temporal con symlinks"""
        import shutil
        import sys
        
        if image_limit == 0:
            self.logger.info("📦 Usando dataset COMPLETO (sin límite)")
            sys.stdout.flush()
            return self.config.DATA_YAML
        
        self.logger.info(f"🔨 Creando dataset limitado: {image_limit} imágenes por categoría")
        sys.stdout.flush()
        
        # Crear directorio temporal
        temp_dir = self.config.OUTPUT_DIR / f'temp_limited_dataset_{image_limit}'
        if temp_dir.exists():
            self.logger.info(f"🧹 Limpiando directorio temporal existente...")
            sys.stdout.flush()
            shutil.rmtree(temp_dir)
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"📁 Directorio temporal creado: {temp_dir}")
        sys.stdout.flush()
        
        # Crear estructura y copiar symlinks
        total_train = 0
        total_val = 0
        total_test = 0
        
        for split in ['train', 'val', 'test']:
            self.logger.info(f"\n   Procesando split: {split}")
            sys.stdout.flush()
            
            split_dir = self.config.IMAGES_DIR / split
            temp_split_img = temp_dir / 'images' / split
            temp_split_lbl = temp_dir / 'labels' / split
            temp_split_img.mkdir(parents=True, exist_ok=True)
            temp_split_lbl.mkdir(parents=True, exist_ok=True)
            
            split_count = 0
            for category_dir in sorted(split_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                
                category = category_dir.name
                self.logger.info(f"      Categoría: {category}")
                sys.stdout.flush()
                
                # Crear directorios de categoría
                temp_cat_img = temp_split_img / category
                temp_cat_lbl = temp_split_lbl / category
                temp_cat_img.mkdir(exist_ok=True)
                temp_cat_lbl.mkdir(exist_ok=True)
                
                # Obtener imágenes limitadas
                images = sorted(list(category_dir.glob('*.jpg')) + 
                              list(category_dir.glob('*.png')) +
                              list(category_dir.glob('*.JPEG')))
                images = images[:image_limit]
                
                # Crear symlinks
                for img_path in images:
                    # Symlink de imagen
                    img_link = temp_cat_img / img_path.name
                    if not img_link.exists():
                        img_link.symlink_to(img_path)
                    
                    # Symlink de label
                    label_path = self.config.LABELS_DIR / split / category / img_path.with_suffix('.txt').name
                    if label_path.exists():
                        lbl_link = temp_cat_lbl / label_path.name
                        if not lbl_link.exists():
                            lbl_link.symlink_to(label_path)
                
                split_count += len(images)
                self.logger.info(f"         ✅ {len(images)} imágenes")
                sys.stdout.flush()
            
            if split == 'train':
                total_train = split_count
            elif split == 'val':
                total_val = split_count
            else:
                total_test = split_count
        
        # Crear YAML apuntando al directorio temporal
        yaml_data = {
            'path': str(temp_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 8,
            'names': ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
        }
        
        yaml_path = self.config.OUTPUT_DIR / f"detection_data_limited_{image_limit}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        
        self.logger.info(f"\n✅ Dataset temporal creado exitosamente")
        self.logger.info(f"   📄 YAML: {yaml_path}")
        self.logger.info(f"   📁 Temp dir: {temp_dir}")
        self.logger.info(f"   📊 Train: {total_train} imágenes")
        self.logger.info(f"   📊 Val: {total_val} imágenes")
        self.logger.info(f"   📊 Test: {total_test} imágenes")
        self.logger.info(f"   💾 Cache esperado: ~{total_train * 1.2 / 1024:.1f} MB\n")
        sys.stdout.flush()
        
        return yaml_path

class GridSearchEngine:
    """Motor de búsqueda de hiperparámetros"""
    
    def __init__(self, config, logger, experiment_name):
        self.config = config
        self.logger = logger
        self.experiment_name = experiment_name
        self.monitor = ResourceMonitor()
        
        # Resultados
        self.results = []
        self.best_combination = None
        self.best_map = 0.0
        
        # Directorio de experimento
        self.experiment_dir = self.config.MODELS_DIR / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache compartido entre combinaciones
        self.shared_cache_yaml = None  # YAML con cache cargado
        self.cache_loaded = False  # Flag para saber si ya se cargó
    
    def generate_combinations(self, hyperparams_grid):
        """
        Genera todas las combinaciones de hiperparámetros
        
        Args:
            hyperparams_grid: Dict con listas de valores para cada hiperparámetro
        
        Returns:
            Lista de diccionarios con combinaciones
        """
        keys = hyperparams_grid.keys()
        values = hyperparams_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def train_combination(self, combo_id, model_name, hyperparams, data_yaml):
        """
        Entrena una combinación específica de hiperparámetros
        
        Args:
            combo_id: ID de la combinación
            model_name: Nombre del modelo (YOLOv11n, YOLOv11s, etc.)
            hyperparams: Diccionario con hiperparámetros
            data_yaml: Path al archivo YAML del dataset
        
        Returns:
            Diccionario con métricas
        """
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"COMBINACIÓN {combo_id}: {model_name}")
        sys.stdout.flush()
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Hiperparámetros:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"{'='*80}\n")
        
        # Iniciar monitoreo
        self.monitor.start()
        
        # Cargar modelo
        model_path = self.config.AVAILABLE_MODELS[model_name]
        model = YOLO(model_path)
        
        # ==================================================================
        # CACHE COMPARTIDO: Usar YAML con cache si ya existe
        # ==================================================================
        if self.cache_loaded and self.shared_cache_yaml:
            # Usar YAML con cache ya cargado
            actual_data_yaml = self.shared_cache_yaml
            self.logger.info("♻️  Reutilizando cache del Combo 1")
            sys.stdout.flush()
        else:
            # Primera vez: YOLO cargará y cacheará
            actual_data_yaml = data_yaml
            if not self.cache_loaded:
                self.logger.info("📦 Primera combinación: cacheando imágenes...")
                self.logger.info("   Los siguientes combos reutilizarán este cache")
                sys.stdout.flush()
        
        # Directorio de salida
        combo_dir = self.experiment_dir / f"combo_{combo_id:03d}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        
        # Entrenar
        try:
            train_start = time.time()
            
            results = model.train(
                data=str(actual_data_yaml),
                epochs=hyperparams['epochs'],
                batch=hyperparams['batch'],
                imgsz=hyperparams['imgsz'],
                lr0=hyperparams['lr0'],
                lrf=hyperparams.get('lrf', 0.01),
                momentum=hyperparams.get('momentum', 0.937),
                weight_decay=hyperparams.get('weight_decay', 0.0005),
                warmup_epochs=hyperparams.get('warmup_epochs', 3.0),
                warmup_momentum=hyperparams.get('warmup_momentum', 0.8),
                warmup_bias_lr=hyperparams.get('warmup_bias_lr', 0.1),
                box=hyperparams.get('box', 7.5),
                cls=hyperparams.get('cls', 0.5),
                dfl=hyperparams.get('dfl', 1.5),
                optimizer=hyperparams.get('optimizer', 'SGD'),  # Valor por defecto: SGD
                close_mosaic=hyperparams.get('close_mosaic', 10),
                patience=hyperparams.get('patience', 50),  # Valor por defecto: 50
                amp=hyperparams.get('amp', True),
                fraction=hyperparams.get('fraction', 1.0),
                device=hyperparams.get('device', 0),
                workers=hyperparams.get('workers', 8),  # Valor por defecto: 8 workers
                cache=hyperparams.get('cache', False),  # Valor por defecto: False
                project=str(combo_dir.parent),
                name=combo_dir.name,
                exist_ok=True,
                verbose=False
            )
            
            train_time = time.time() - train_start
            
            # Guardar referencia al cache si es la primera combinación
            if not self.cache_loaded:
                self.shared_cache_yaml = data_yaml
                self.cache_loaded = True
                self.logger.info("✅ Cache cargado y guardado para próximas combinaciones")
                sys.stdout.flush()
            
            # Extraer métricas (compatible con DDP)
            import pandas as pd
            
            # Intentar obtener de results_dict (single GPU)
            if hasattr(results, 'results_dict') and results.results_dict:
                mAP50 = float(results.results_dict.get('metrics/mAP50(B)', 0))
                mAP50_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0))
                precision = float(results.results_dict.get('metrics/precision(B)', 0))
                recall = float(results.results_dict.get('metrics/recall(B)', 0))
                box_loss = float(results.results_dict.get('train/box_loss', 0))
                cls_loss = float(results.results_dict.get('train/cls_loss', 0))
                dfl_loss = float(results.results_dict.get('train/dfl_loss', 0))
            else:
                # DDP: Leer desde results.csv
                results_csv = combo_dir / 'results.csv'
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    last_row = df.iloc[-1]
                    mAP50 = float(last_row.get('metrics/mAP50(B)', last_row.get('mAP50', 0)))
                    mAP50_95 = float(last_row.get('metrics/mAP50-95(B)', last_row.get('mAP50-95', 0)))
                    precision = float(last_row.get('metrics/precision(B)', last_row.get('precision', 0)))
                    recall = float(last_row.get('metrics/recall(B)', last_row.get('recall', 0)))
                    box_loss = float(last_row.get('train/box_loss', 0))
                    cls_loss = float(last_row.get('train/cls_loss', 0))
                    dfl_loss = float(last_row.get('train/dfl_loss', 0))
                else:
                    self.logger.warning("⚠️  No se pudo leer results.csv")
                    mAP50 = mAP50_95 = precision = recall = 0
                    box_loss = cls_loss = dfl_loss = 0
            
            metrics = {
                'combination_id': combo_id,
                'model': model_name,
                'hyperparameters': hyperparams,
                'training_time_seconds': train_time,
                'training_time_hours': train_time / 3600,
                'metrics': {
                    'mAP50': mAP50,
                    'mAP50-95': mAP50_95,
                    'precision': precision,
                    'recall': recall,
                    'box_loss': box_loss,
                    'cls_loss': cls_loss,
                    'dfl_loss': dfl_loss
                },
                'resources': self.monitor.get_metrics(),
                'paths': {
                    'best_weights': str(combo_dir / 'weights' / 'best.pt'),
                    'results_dir': str(combo_dir)
                },
                'success': True,
                'error': None
            }
            
            # Actualizar mejor combinación
            if metrics['metrics']['mAP50-95'] > self.best_map:
                self.best_map = metrics['metrics']['mAP50-95']
                self.best_combination = metrics
            
            self.logger.info(f"✅ Combinación {combo_id} completada")
            self.logger.info(f"   mAP50-95: {metrics['metrics']['mAP50-95']:.4f}")
            self.logger.info(f"   Tiempo: {train_time/60:.2f} minutos")
            
        except Exception as e:
            self.logger.error(f"❌ Error en combinación {combo_id}: {e}")
            
            metrics = {
                'combination_id': combo_id,
                'model': model_name,
                'hyperparameters': hyperparams,
                'success': False,
                'error': str(e)
            }
        
        return metrics
    

    def cleanup_gpu_after_combination(self):
        """Liberar memoria GPU (NO cache) después de cada combinación"""
        import gc
        import torch
        
        self.logger.info("🧹 Liberando memoria GPU...")
        sys.stdout.flush()
        
        # Garbage collection
        gc.collect()
        
        # Limpiar cache de GPU SOLAMENTE
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log de uso actual
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.info(f"   GPU {i}: {mem_allocated:.2f}GB allocated, "
                               f"{mem_reserved:.2f}GB reserved")
        
        self.logger.info("✅ GPU limpia (cache de datos preservado)")
        sys.stdout.flush()
    
        def validate_best(self, data_yaml):
        """Valida la mejor combinación en el test set"""
        
        if self.best_combination is None:
            self.logger.warning("No hay mejor combinación para validar")
            return None
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("VALIDACIÓN DE MEJOR COMBINACIÓN EN TEST SET")
        sys.stdout.flush()
        self.logger.info(f"{'='*80}\n")
        
        # Cargar mejor modelo
        best_weights = self.best_combination['paths']['best_weights']
        model = YOLO(best_weights)
        
        # Validar en test
        val_start = time.time()
        results = model.val(data=str(data_yaml), split='test')
        val_time = time.time() - val_start
        
        validation_metrics = {
            'combination_id': self.best_combination['combination_id'],
            'validation_type': 'test_set',
            'metrics': {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0))
            },
            'validation_time_seconds': val_time,
            'resources': self.monitor.get_metrics()
        }
        
        self.logger.info(f"✅ Validación completada")
        self.logger.info(f"   Test mAP50-95: {validation_metrics['metrics']['mAP50-95']:.4f}")
        
        return validation_metrics
    
    def save_results(self):
        """Guarda resultados del grid search"""
        
        # Guardar todos los resultados
        results_file = self.experiment_dir / 'grid_search_results.json'
        save_json(self.results, results_file)
        
        self.logger.info(f"\n📄 Resultados guardados:")
        self.logger.info(f"   Todos: {results_file}")
        
        # Guardar mejor combinación si existe
        if self.best_combination:
            best_file = self.experiment_dir / 'best_combination.json'
            save_json(self.best_combination, best_file)
            self.logger.info(f"   Mejor: {best_file}")
        else:
            self.logger.warning("   ⚠️  No se encontró mejor combinación (ningún entrenamiento exitoso)")


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

class Visualizer:
    """Genera gráficas comparativas"""
    
    def __init__(self, experiment_dir, logger):
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.plots_dir = experiment_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
    
    def create_all_plots(self, results, best_combination):
        """Crea todas las gráficas"""
        
        self.logger.info("\n📊 Generando gráficas comparativas...")
        
        # Filtrar solo combinaciones exitosas
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            self.logger.warning("No hay combinaciones exitosas para graficar")
            return
        
        # Crear DataFrame
        df = self._create_dataframe(successful)
        
        # Gráficas
        self.plot_metrics_comparison(df)
        self.plot_hyperparams_impact(df)
        self.plot_training_time(df)
        self.plot_resources(df)
        self.plot_top_combinations(df, best_combination)
        self.plot_pareto_front(df)
        
        self.logger.info(f"✅ Gráficas guardadas en: {self.plots_dir}")
    
    def _create_dataframe(self, results):
        """Crea DataFrame desde resultados"""
        
        data = []
        for r in results:
            row = {
                'combo_id': r['combination_id'],
                'model': r['model'],
                'epochs': r['hyperparameters']['epochs'],
                'batch': r['hyperparameters']['batch'],
                'lr0': r['hyperparameters']['lr0'],
                'optimizer': r['hyperparameters']['optimizer'],
                'patience': r['hyperparameters']['patience'],
                'workers': r['hyperparameters']['workers'],
                'mAP50': r['metrics']['mAP50'],
                'mAP50-95': r['metrics']['mAP50-95'],
                'precision': r['metrics']['precision'],
                'recall': r['metrics']['recall'],
                'training_time_hours': r['training_time_hours'],
                'gpu_memory_mb': r['resources'].get('gpu_memory_used_mb', 0),
                'gpu_utilization': r['resources'].get('gpu_utilization_percent', 0)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_metrics_comparison(self, df):
        """Gráfica de comparación de métricas"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Métricas por Combinación', fontsize=16, fontweight='bold')
        
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Ordenar por métrica
            df_sorted = df.sort_values(metric, ascending=False).head(20)
            
            bars = ax.barh(df_sorted['combo_id'].astype(str), df_sorted[metric])
            
            # Colorear mejor
            bars[0].set_color('green')
            
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel('Combination ID', fontsize=12)
            ax.set_title(f'Top 20 por {metric}', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hyperparams_impact(self, df):
        """Impacto de hiperparámetros en mAP50-95"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Impacto de Hiperparámetros en mAP50-95', fontsize=16, fontweight='bold')
        
        hyperparams = ['epochs', 'batch', 'lr0', 'optimizer', 'patience', 'workers']
        
        for idx, param in enumerate(hyperparams):
            ax = axes[idx // 3, idx % 3]
            
            if param in ['optimizer', 'model']:
                # Categorical
                grouped = df.groupby(param)['mAP50-95'].agg(['mean', 'std'])
                grouped.plot(kind='bar', y='mean', yerr='std', ax=ax, capsize=4)
                ax.set_ylabel('mAP50-95', fontsize=11)
            else:
                # Numerical
                ax.scatter(df[param], df['mAP50-95'], alpha=0.6, s=100)
                ax.set_xlabel(param, fontsize=11)
                ax.set_ylabel('mAP50-95', fontsize=11)
            
            ax.set_title(f'{param} vs mAP50-95', fontsize=12)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'hyperparams_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_time(self, df):
        """Tiempo de entrenamiento"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Por combinación
        df_sorted = df.sort_values('training_time_hours')
        ax1.barh(df_sorted['combo_id'].astype(str), df_sorted['training_time_hours'])
        ax1.set_xlabel('Training Time (hours)', fontsize=12)
        ax1.set_ylabel('Combination ID', fontsize=12)
        ax1.set_title('Tiempo de Entrenamiento por Combinación', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)
        
        # Tiempo vs mAP
        scatter = ax2.scatter(
            df['training_time_hours'],
            df['mAP50-95'],
            c=df['batch'],
            s=100,
            cmap='viridis',
            alpha=0.6
        )
        ax2.set_xlabel('Training Time (hours)', fontsize=12)
        ax2.set_ylabel('mAP50-95', fontsize=12)
        ax2.set_title('Tiempo vs Precisión (color = batch size)', fontsize=14)
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Batch Size')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_resources(self, df):
        """Uso de recursos"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # GPU Memory
        ax1.scatter(df['gpu_memory_mb'], df['mAP50-95'], s=100, alpha=0.6)
        ax1.set_xlabel('GPU Memory Used (MB)', fontsize=12)
        ax1.set_ylabel('mAP50-95', fontsize=12)
        ax1.set_title('Memoria GPU vs Precisión', fontsize=14)
        ax1.grid(alpha=0.3)
        
        # GPU Utilization
        ax2.scatter(df['gpu_utilization'], df['mAP50-95'], s=100, alpha=0.6, color='orange')
        ax2.set_xlabel('GPU Utilization (%)', fontsize=12)
        ax2.set_ylabel('mAP50-95', fontsize=12)
        ax2.set_title('Utilización GPU vs Precisión', fontsize=14)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'resources.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top_combinations(self, df, best_combination):
        """Top 10 combinaciones"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_top = df.nlargest(10, 'mAP50-95')
        
        x = range(len(df_top))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], df_top['mAP50'], width, label='mAP50', alpha=0.8)
        ax.bar([i + width/2 for i in x], df_top['mAP50-95'], width, label='mAP50-95', alpha=0.8)
        
        # Highlight best
        if best_combination:
            best_idx = df_top['combo_id'].tolist().index(best_combination['combination_id'])
            ax.axvline(best_idx, color='red', linestyle='--', linewidth=2, label='Best')
        
        ax.set_xlabel('Combination ID', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('Top 10 Combinaciones', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_top['combo_id'].astype(str))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'top_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_front(self, df):
        """Frontera de Pareto: Precisión vs Tiempo"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            df['training_time_hours'],
            df['mAP50-95'],
            c=df['combo_id'],
            s=150,
            cmap='rainbow',
            alpha=0.7,
            edgecolors='black'
        )
        
        # Identificar frontera de Pareto
        pareto_front = []
        for idx, row in df.iterrows():
            is_pareto = True
            for _, other_row in df.iterrows():
                if (other_row['mAP50-95'] > row['mAP50-95'] and 
                    other_row['training_time_hours'] <= row['training_time_hours']):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_front.append(idx)
        
        # Marcar frontera
        if pareto_front:
            pareto_df = df.loc[pareto_front].sort_values('training_time_hours')
            ax.plot(
                pareto_df['training_time_hours'],
                pareto_df['mAP50-95'],
                'r--',
                linewidth=2,
                label='Pareto Front'
            )
        
        ax.set_xlabel('Training Time (hours)', fontsize=12)
        ax.set_ylabel('mAP50-95', fontsize=12)
        ax.set_title('Frontera de Pareto: Precisión vs Tiempo', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Combination ID')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# REPORTE
# ============================================================================

class ReportGenerator:
    """Genera reporte final"""
    
    def __init__(self, experiment_dir, logger):
        self.experiment_dir = experiment_dir
        self.logger = logger
    
    def generate_summary(self, results, best_combination, validation_metrics):
        """Genera resumen ejecutivo"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 RESUMEN EJECUTIVO DE GRID SEARCH")
        self.logger.info("="*80)
        
        # Estadísticas generales
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        self.logger.info(f"\n🎯 ESTADÍSTICAS GENERALES:")
        self.logger.info(f"   Total combinaciones: {len(results)}")
        self.logger.info(f"   Exitosas: {len(successful)}")
        self.logger.info(f"   Fallidas: {len(failed)}")
        
        if successful:
            # mAP stats
            maps = [r['metrics']['mAP50-95'] for r in successful]
            self.logger.info(f"\n📈 PRECISIÓN (mAP50-95):")
            self.logger.info(f"   Mejor: {max(maps):.4f}")
            self.logger.info(f"   Peor: {min(maps):.4f}")
            self.logger.info(f"   Promedio: {np.mean(maps):.4f}")
            self.logger.info(f"   Std Dev: {np.std(maps):.4f}")
            
            # Tiempo stats
            times = [r['training_time_hours'] for r in successful]
            self.logger.info(f"\n⏱️  TIEMPO DE ENTRENAMIENTO:")
            self.logger.info(f"   Total: {sum(times):.2f} horas")
            self.logger.info(f"   Promedio: {np.mean(times):.2f} horas")
            self.logger.info(f"   Rango: {min(times):.2f} - {max(times):.2f} horas")
        
        # Mejor combinación
        if best_combination:
            self.logger.info(f"\n🏆 MEJOR COMBINACIÓN:")
            sys.stdout.flush()
            self.logger.info(f"   ID: {best_combination['combination_id']}")
            self.logger.info(f"   Modelo: {best_combination['model']}")
            self.logger.info(f"   mAP50-95: {best_combination['metrics']['mAP50-95']:.4f}")
            self.logger.info(f"\n   Hiperparámetros:")
            for key, value in best_combination['hyperparameters'].items():
                self.logger.info(f"      {key}: {value}")
        
        # Validación en test
        if validation_metrics:
            self.logger.info(f"\n✅ VALIDACIÓN EN TEST SET:")
            self.logger.info(f"   mAP50-95: {validation_metrics['metrics']['mAP50-95']:.4f}")
            self.logger.info(f"   mAP50: {validation_metrics['metrics']['mAP50']:.4f}")
            self.logger.info(f"   Precision: {validation_metrics['metrics']['precision']:.4f}")
            self.logger.info(f"   Recall: {validation_metrics['metrics']['recall']:.4f}")
        
        self.logger.info("\n" + "="*80 + "\n")
        
        # Guardar resumen en archivo
        summary = {
            'total_combinations': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'best_combination': best_combination,
            'validation_metrics': validation_metrics
        }
        
        save_json(summary, self.experiment_dir / 'summary.json')


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search Optimizado para YOLO11'
    )
    
    # Modo
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'validate', 'both'],
        default='both',
        help='Modo de ejecución'
    )
    
    # Experimento
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='grid_search_exp',
        help='Nombre del experimento'
    )
    
    # Modelo base
    parser.add_argument(
        '--model_base',
        type=str,
        default='YOLOv11s',
        choices=['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x'],
        help='Modelo base a usar'
    )
    
    # Hiperparámetros (listas separadas por coma)
    parser.add_argument(
        '--hyperparam',
        action='append',
        nargs='+',
        help='Hiperparámetros en formato key=val1,val2,val3'
    )
    
    # Límite de imágenes
    parser.add_argument(
        '--image-limit',
        type=int,
        default=20,
        help='Límite de imágenes por split (0 = todas)'
    )
    
    # Directorio base
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/workspace',
        help='Directorio base'
    )
    
    args = parser.parse_args()
    
    # Configurar paths
    config = Config()
    config.BASE_DIR = Path(args.base_dir)
    config.DATASET_DIR = config.BASE_DIR / "DSM"
    config.IMAGES_DIR = config.DATASET_DIR / "images"
    config.LABELS_DIR = config.DATASET_DIR / "labels"
    config.OUTPUT_DIR = config.BASE_DIR / "outputs"
    config.MODELS_DIR = config.BASE_DIR / "models/phase1_detection"
    
    # Setup logging
    logger, log_file = setup_logging(args.experiment_name)
    
    logger.info("="*80)
    logger.info("🚀 GRID SEARCH OPTIMIZADO PARA YOLO11")
    logger.info("="*80)
    logger.info(f"\nExperimento: {args.experiment_name}")
    logger.info(f"Modelo base: {args.model_base}")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Límite de imágenes: {args.image_limit}")
    
    # Parsear hiperparámetros
    hyperparams_grid = dict(config.DEFAULT_HYPERPARAMS)
    
    if args.hyperparam:
        logger.info(f"\n📝 Hiperparámetros configurados:")
        
        # ADVERTENCIA: Multi-GPU en grid search
        device_values = None
        for param_list in args.hyperparam:
            for param in param_list:
                if '=' in param:
                    key, value = param.split('=', 1)
                    if key == 'device' and ',' in value:
                        device_values = value
                        logger.info(f"\n✅ Multi-GPU detectado: device={value}")
                        logger.info(f"   Las GPUs trabajarán en PARALELO usando DDP")
                        logger.info(f"   Batch se dividirá entre las GPUs")
                        logger.info(f"   Ejemplo: batch=128 → 64 por GPU\n")
                        sys.stdout.flush()
        
        for param_list in args.hyperparam:
            for param in param_list:
                if '=' in param:
                    key, values_str = param.split('=', 1)
                    values = parse_hyperparam_list(values_str, param_name=key)
                    hyperparams_grid[key] = values
                    logger.info(f"   {key}: {values}")
    
    # Calcular número total de combinaciones
    total_combinations = 1
    for values in hyperparams_grid.values():
        total_combinations *= len(values)
    
    logger.info(f"\n🎯 Total de combinaciones: {total_combinations}")
    
    # Crear dataset manager
    dataset_manager = DatasetManager(config, logger)
    data_yaml = dataset_manager.create_limited_yaml(args.image_limit)
    
    # Iniciar grid search
    grid_search = GridSearchEngine(config, logger, args.experiment_name)
    
    # Generar combinaciones
    combinations = grid_search.generate_combinations(hyperparams_grid)
    
    logger.info(f"\n🚀 Iniciando grid search...")
    sys.stdout.flush()
    logger.info(f"   Combinaciones a probar: {len(combinations)}")
    
    # Entrenar cada combinación
    if args.mode in ['train', 'both']:
        for i, combo in enumerate(tqdm(combinations, desc="Grid Search"), 1):
            result = grid_search.train_combination(
                combo_id=i,
                model_name=args.model_base,
                hyperparams=combo,
                data_yaml=data_yaml
            )
            grid_search.results.append(result)
            
            # Limpiar GPU (mantener cache de datos)
            grid_search.cleanup_gpu_after_combination()
        
        # Guardar resultados
        grid_search.save_results()
    
    # Validar mejor combinación
    validation_metrics = None
    if args.mode in ['validate', 'both']:
        if grid_search.best_combination:
            validation_metrics = grid_search.validate_best(data_yaml)
            
            # Guardar validación
            val_file = grid_search.experiment_dir / 'validation_metrics.json'
            save_json(validation_metrics, val_file)
    
    # Generar visualizaciones
    visualizer = Visualizer(grid_search.experiment_dir, logger)
    visualizer.create_all_plots(grid_search.results, grid_search.best_combination)
    
    # Generar reporte
    report = ReportGenerator(grid_search.experiment_dir, logger)
    report.generate_summary(grid_search.results, grid_search.best_combination, validation_metrics)
    
    logger.info("\n✅ Grid Search completado exitosamente")
    logger.info(f"📁 Resultados en: {grid_search.experiment_dir}")
    logger.info(f"📄 Log file: {log_file}\n")
    
    # Limpiar recursos
    logger.info("🧹 Cerrando procesos y liberando recursos...")
    sys.stdout.flush()
    
    # Forzar garbage collection
    import gc
    gc.collect()
    
    # Limpiar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Recursos liberados correctamente\n")
    sys.stdout.flush()
    
    # Salir explícitamente
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
