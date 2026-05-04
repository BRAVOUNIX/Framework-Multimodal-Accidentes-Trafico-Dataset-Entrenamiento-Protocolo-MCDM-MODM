"""
FASE 2: Grid Search para YOLO11 Clasificación
==============================================
Plataforma: RunPod
Búsqueda exhaustiva de hiperparámetros para clasificación
Estructura: DSM/images/{train,val,test}/{clase}/{img.jpg}
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
    log_file = f"phase2_grid_search_{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('GridSearch_CLS')
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file


# ============================================================================
# CONFIGURACIÓN BASE
# ============================================================================

class Config:
    """Configuración del grid search para clasificación"""
    
    # Paths
    BASE_DIR = Path("/workspace")
    DATASET_DIR = BASE_DIR / "DSM"
    IMAGES_DIR = DATASET_DIR / "images"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models/phase2_classification"
    LOGS_DIR = BASE_DIR / "logs/phase2_classification"
    
    # Clases
    CLASSES = ["accidente", "fuego", "trafico_denso", "trafico_escaso"]
    
    # Modelos disponibles para clasificación
    AVAILABLE_MODELS = {
        'YOLOv11n-cls': 'yolo11n-cls.pt',
        'YOLOv11s-cls': 'yolo11s-cls.pt',
        'YOLOv11m-cls': 'yolo11m-cls.pt',
        'YOLOv11l-cls': 'yolo11l-cls.pt',
        'YOLOv11x-cls': 'yolo11x-cls.pt'
    }
    
    # Hiperparámetros por defecto
    DEFAULT_HYPERPARAMS = {
        'imgsz': [224],
        'lr0': [0.001],
        'lrf': [0.01],
        'momentum': [0.9],
        'weight_decay': [0.0005],
        'warmup_epochs': [3.0],
        'warmup_momentum': [0.8],
        'warmup_bias_lr': [0.1],
        'amp': [True],
        'optimizer': ['Adam'],
        'device': [0]
    }
    
    DEFAULT_IMAGE_LIMIT = 0


# ============================================================================
# UTILIDADES
# ============================================================================

def save_json(data, filepath):
    """Guarda datos en JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def parse_hyperparam_list(value_str, param_name=''):
    """
    Parsea una lista de valores separados por comas
    
    Ejemplos:
        "64,128" → [64, 128]
        "0.001,0.01" → [0.001, 0.01]
        "SGD,Adam" → ['SGD', 'Adam']
        "0,1" → [0, 1] (excepto si param_name es 'device')
        "True,False" → [True, False]
    """
    # Caso especial: device con múltiples GPUs
    if param_name == 'device':
        if ',' in value_str:
            # "0,1" → [[0, 1]] (una sola configuración multi-GPU)
            try:
                devices = [int(x.strip()) for x in value_str.split(',')]
                return [devices]  # Lista con un solo elemento (la lista de GPUs)
            except ValueError:
                return [[0, 1]]  # Default multi-GPU
        else:
            # "0" → [[0]] (single GPU)
            try:
                return [[int(value_str.strip())]]
            except ValueError:
                return [[0]]
    
    # Para otros parámetros
    values = []
    for val_str in value_str.split(','):
        val_str = val_str.strip()
        
        # Intentar parsear como int
        try:
            values.append(int(val_str))
            continue
        except ValueError:
            pass
        
        # Intentar parsear como float
        try:
            values.append(float(val_str))
            continue
        except ValueError:
            pass
        
        # Boolean
        if val_str.lower() == 'true':
            values.append(True)
        elif val_str.lower() == 'false':
            values.append(False)
        else:
            # String
            values.append(val_str)
    
    return values


class ResourceMonitor:
    """Monitor de recursos computacionales"""
    
    def __init__(self):
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()
        
    def start(self):
        self.start_time = time.time()
        
    def get_metrics(self):
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
                    metrics['gpus'] = []
                    for gpu in gpus:
                        metrics['gpus'].append({
                            'name': gpu.name,
                            'memory_used_mb': gpu.memoryUsed,
                            'memory_total_mb': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'utilization_percent': gpu.load * 100,
                            'temperature_c': gpu.temperature
                        })
            except:
                pass
                
        return metrics


# ============================================================================
# DATASET MANAGER
# ============================================================================

class DatasetManager:
    """Gestiona la creación de datasets limitados para clasificación"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def create_limited_dataset(self, image_limit):
        """
        Crea un dataset limitado con symlinks
        
        Estructura esperada:
        DSM/images/
            ├── train/
            │   ├── accidente/
            │   ├── fuego/
            │   ├── trafico_denso/
            │   └── trafico_escaso/
            ├── val/
            └── test/
        """
        if image_limit == 0:
            # Usar dataset completo
            self.logger.info("📁 Usando dataset completo (sin límite)")
            return str(self.config.IMAGES_DIR)
        
        self.logger.info(f"🔨 Creando dataset limitado: {image_limit} imágenes por categoría")
        
        # Crear directorio temporal
        temp_dir = self.config.OUTPUT_DIR / f"temp_limited_cls_dataset_{image_limit}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Directorio temporal creado: {temp_dir}")
        
        total_images = {'train': 0, 'val': 0, 'test': 0}
        
        # Procesar cada split
        for split in ['train', 'val', 'test']:
            split_src = self.config.IMAGES_DIR / split
            split_dst = temp_dir / split
            split_dst.mkdir(parents=True, exist_ok=True)
            
            if not split_src.exists():
                self.logger.warning(f"⚠️  Split {split} no encontrado: {split_src}")
                continue
            
            self.logger.info(f"\n   Procesando split: {split}")
            
            # Procesar cada categoría
            for category in self.config.CLASSES:
                category_src = split_src / category
                category_dst = split_dst / category
                category_dst.mkdir(parents=True, exist_ok=True)
                
                if not category_src.exists():
                    self.logger.warning(f"      ⚠️  Categoría {category} no encontrada")
                    continue
                
                self.logger.info(f"       Categoría: {category}")
                
                # Obtener imágenes
                images = list(category_src.glob('*.jpg')) + list(category_src.glob('*.png'))
                images = images[:image_limit]
                
                # Crear symlinks
                for img in images:
                    dst_link = category_dst / img.name
                    if not dst_link.exists():
                        dst_link.symlink_to(img)
                
                total_images[split] += len(images)
                self.logger.info(f"          ✅ {len(images)} imágenes")
        
        self.logger.info(f"\n✅ Dataset temporal creado exitosamente")
        self.logger.info(f"   📁 Path: {temp_dir}")
        self.logger.info(f"   📊 Train: {total_images['train']} imágenes")
        self.logger.info(f"   📊 Val: {total_images['val']} imágenes")
        self.logger.info(f"   📊 Test: {total_images['test']} imágenes")
        
        return str(temp_dir)


# ============================================================================
# GRID SEARCH ENGINE
# ============================================================================

class GridSearchEngine:
    """Motor de Grid Search para clasificación"""
    
    def __init__(self, config, logger, experiment_name):
        self.config = config
        self.logger = logger
        self.experiment_name = experiment_name
        self.monitor = ResourceMonitor()
        
        # Resultados
        self.results = []
        self.best_combination = None
        self.best_accuracy = 0.0
        
        # Directorio de experimento
        self.experiment_dir = self.config.MODELS_DIR / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_combinations(self, hyperparams_grid):
        """Genera todas las combinaciones de hiperparámetros"""
        keys = hyperparams_grid.keys()
        values = hyperparams_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def train_combination(self, combo_id, model_name, hyperparams, data_path):
        """Entrena una combinación específica de hiperparámetros"""
        
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"COMBINACIÓN {combo_id}: {model_name}")
            self.logger.info(f"{'='*80}")
            
            # Directorio de combinación
            combo_dir = self.experiment_dir / f'combo_{combo_id:03d}'
            combo_dir.mkdir(parents=True, exist_ok=True)
            
            # Log de hiperparámetros
            self.logger.info("Hiperparámetros:")
            for key, value in hyperparams.items():
                self.logger.info(f"   {key}: {value}")
            self.logger.info("=" * 80)
            sys.stdout.flush()
            
            # Cargar modelo
            model_path = self.config.AVAILABLE_MODELS[model_name]
            model = YOLO(model_path)
            
            # Iniciar monitoreo
            self.monitor.start()
            train_start = time.time()
            
            # Entrenar
            results = model.train(
                data=data_path,
                project=str(self.experiment_dir),
                name=f'combo_{combo_id:03d}',
                exist_ok=True,
                verbose=True,
                save=True,
                plots=True,
                **hyperparams
            )
            
            train_time = time.time() - train_start
            
            # Extraer métricas (compatible con DDP)
            import pandas as pd
            
            if hasattr(results, 'results_dict') and results.results_dict:
                top1 = float(results.results_dict.get('metrics/accuracy_top1', 0))
                top5 = float(results.results_dict.get('metrics/accuracy_top5', 0))
            else:
                # Leer desde results.csv (para DDP)
                results_csv = combo_dir / 'results.csv'
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    last_row = df.iloc[-1]
                    top1 = float(last_row.get('metrics/accuracy_top1', 
                                             last_row.get('top1', 0)))
                    top5 = float(last_row.get('metrics/accuracy_top5', 
                                             last_row.get('top5', 0)))
                else:
                    self.logger.warning("⚠️  No se pudo leer results.csv")
                    top1 = top5 = 0
            
            metrics = {
                'combination_id': combo_id,
                'model': model_name,
                'hyperparameters': hyperparams,
                'training_time_seconds': train_time,
                'training_time_hours': train_time / 3600,
                'metrics': {
                    'top1_accuracy': top1,
                    'top5_accuracy': top5,
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
            if metrics['metrics']['top1_accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['metrics']['top1_accuracy']
                self.best_combination = metrics
            
            self.logger.info(f"✅ Combinación {combo_id} completada")
            self.logger.info(f"   Top-1 Accuracy: {metrics['metrics']['top1_accuracy']:.4f}")
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
        """Liberar memoria GPU después de cada combinación"""
        import gc
        import torch
        
        self.logger.info("🧹 Liberando memoria GPU...")
        sys.stdout.flush()
        
        # Garbage collection
        gc.collect()
        
        # Limpiar cache de GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log de uso actual
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.info(f"   GPU {i}: {mem_allocated:.2f}GB allocated, "
                               f"{mem_reserved:.2f}GB reserved")
        
        self.logger.info("✅ GPU limpia")
        sys.stdout.flush()
    
    def validate_best(self, data_path):
        """Valida la mejor combinación en el test set"""
        
        if self.best_combination is None:
            self.logger.warning("No hay mejor combinación para validar")
            return None
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("VALIDACIÓN DE MEJOR COMBINACIÓN EN TEST SET")
        self.logger.info(f"{'='*80}")
        sys.stdout.flush()
        
        best_weights = Path(self.best_combination['paths']['best_weights'])
        
        if not best_weights.exists():
            self.logger.error(f"No se encontró: {best_weights}")
            return None
        
        self.logger.info(f"📦 Cargando modelo: {best_weights}")
        model = YOLO(str(best_weights))
        
        self.monitor.start()
        
        self.logger.info("🧪 Validando en test set...")
        sys.stdout.flush()
        
        val_start = time.time()
        results = model.val(
            data=data_path,
            split='test',
            imgsz=self.best_combination['hyperparameters'].get('imgsz', 224),
            batch=self.best_combination['hyperparameters'].get('batch', 32),
            plots=True,
            save_json=True,
            project=str(self.experiment_dir),
            name='validation_best',
            exist_ok=True
        )
        val_time = time.time() - val_start
        
        validation_metrics = {
            'best_combination_id': self.best_combination['combination_id'],
            'validation_time_seconds': val_time,
            'metrics': {
                'top1_accuracy': float(results.top1),
                'top5_accuracy': float(results.top5),
            },
            'resources': self.monitor.get_metrics()
        }
        
        if validation_metrics:
            self.logger.info(f"\n✅ VALIDACIÓN EN TEST SET:")
            self.logger.info(f"   Top-1 Accuracy: {validation_metrics['metrics']['top1_accuracy']:.4f}")
            self.logger.info(f"   Top-5 Accuracy: {validation_metrics['metrics']['top5_accuracy']:.4f}")
        
        self.logger.info("\n" + "="*80 + "\n")
        
        return validation_metrics
    
    def save_results(self):
        """Guarda todos los resultados"""
        
        # Filtrar exitosos y fallidos
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("RESUMEN DE GRID SEARCH")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total de combinaciones: {len(self.results)}")
        self.logger.info(f"Exitosas: {len(successful)}")
        self.logger.info(f"Fallidas: {len(failed)}")
        
        if self.best_combination:
            self.logger.info(f"\n🏆 MEJOR COMBINACIÓN:")
            self.logger.info(f"   ID: {self.best_combination['combination_id']}")
            self.logger.info(f"   Top-1 Accuracy: {self.best_combination['metrics']['top1_accuracy']:.4f}")
            self.logger.info(f"   Hiperparámetros:")
            for key, value in self.best_combination['hyperparameters'].items():
                self.logger.info(f"      {key}: {value}")
        
        # Guardar JSONs
        save_json(self.results, self.experiment_dir / 'all_results.json')
        save_json(successful, self.experiment_dir / 'successful_results.json')
        
        if failed:
            save_json(failed, self.experiment_dir / 'failed_results.json')
        
        if self.best_combination:
            save_json(self.best_combination, self.experiment_dir / 'best_combination.json')


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search Optimizado para YOLO11 Clasificación'
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
        default='grid_search_cls',
        help='Nombre del experimento'
    )
    
    # Modelo base
    parser.add_argument(
        '--model_base',
        type=str,
        default='YOLOv11s-cls',
        choices=['YOLOv11n-cls', 'YOLOv11s-cls', 'YOLOv11m-cls', 'YOLOv11l-cls', 'YOLOv11x-cls'],
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
        default=0,
        help='Límite de imágenes por categoría (0 = todas)'
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
    config.OUTPUT_DIR = config.BASE_DIR / "outputs"
    config.MODELS_DIR = config.BASE_DIR / "models/phase2_classification"
    
    # Setup logging
    logger, log_file = setup_logging(args.experiment_name)
    
    logger.info("="*80)
    logger.info("🚀 GRID SEARCH OPTIMIZADO PARA YOLO11 CLASIFICACIÓN")
    logger.info("="*80)
    logger.info(f"\nExperimento: {args.experiment_name}")
    logger.info(f"Modelo base: {args.model_base}")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Límite de imágenes: {args.image_limit}")
    
    # Parsear hiperparámetros
    hyperparams_grid = dict(config.DEFAULT_HYPERPARAMS)
    
    if args.hyperparam:
        logger.info(f"\n📝 Hiperparámetros configurados:")
        
        # Detectar Multi-GPU
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
    data_path = dataset_manager.create_limited_dataset(args.image_limit)
    
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
                data_path=data_path
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
            validation_metrics = grid_search.validate_best(data_path)
            
            # Guardar validación
            if validation_metrics:
                val_file = grid_search.experiment_dir / 'validation_metrics.json'
                save_json(validation_metrics, val_file)
    
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
    main()
