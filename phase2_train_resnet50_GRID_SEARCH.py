"""
FASE 2: Grid Search para ResNet50 Clasificación
===============================================
Plataforma: RunPod
Búsqueda exhaustiva de hiperparámetros para clasificación con ResNet50
Estructura: DSM/images/{train,val,test}/{clase}/{img.jpg}

CORRECCIONES INCLUIDAS:
- ConnectionResetError: pin_memory=False + persistent_workers
- Multi-GPU: DataParallel con device=0,1 como un elemento
- Cache simulado con persistent workers
"""

import argparse
import sys
import json
import time
import psutil
import GPUtil
import logging
import itertools
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging(experiment_name):
    """Configura logging con archivo y consola"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"phase2_resnet50_grid_{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('GridSearch_ResNet50')
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file


# ============================================================================
# CONFIGURACIÓN BASE
# ============================================================================

class Config:
    """Configuración del grid search para ResNet50"""
    
    # Paths
    BASE_DIR = Path("/workspace")
    DATASET_DIR = BASE_DIR / "DSM"
    IMAGES_DIR = DATASET_DIR / "images"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models/phase2_classification/resnet50"
    
    # Clases
    CLASSES = ["accidente", "fuego", "trafico_denso", "trafico_escaso"]
    NUM_CLASSES = 4
    
    # Hiperparámetros por defecto
    DEFAULT_HYPERPARAMS = {
        'imgsz': [224],  # ResNet50 usa 224x224
        'lr0': [0.001],
        'momentum': [0.9],
        'weight_decay': [0.0001],
        'optimizer': ['SGD'],
        'step_size': [7],
        'gamma': [0.1],
        'pretrained': [True],
        'freeze_layers': [0],
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
    """Parsea una lista de valores separados por comas"""
    # Caso especial: device con múltiples GPUs
    if param_name == 'device':
        if ',' in value_str:
            try:
                devices = [int(x.strip()) for x in value_str.split(',')]
                return [devices]  # Lista con un solo elemento
            except ValueError:
                return [[0, 1]]
        else:
            try:
                return [[int(value_str.strip())]]
            except ValueError:
                return [[0]]
    
    # Para otros parámetros
    values = []
    for val_str in value_str.split(','):
        val_str = val_str.strip()
        
        try:
            values.append(int(val_str))
            continue
        except ValueError:
            pass
        
        try:
            values.append(float(val_str))
            continue
        except ValueError:
            pass
        
        if val_str.lower() == 'true':
            values.append(True)
        elif val_str.lower() == 'false':
            values.append(False)
        else:
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
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
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
                        })
            except:
                pass
                
        return metrics


# ============================================================================
# DATASET MANAGER
# ============================================================================

class DatasetManager:
    """Gestiona la creación de datasets limitados"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def create_limited_dataset(self, image_limit):
        """Crea un dataset limitado con symlinks"""
        if image_limit == 0:
            self.logger.info("📁 Usando dataset completo (sin límite)")
            return str(self.config.IMAGES_DIR)
        
        self.logger.info(f"🔨 Creando dataset limitado: {image_limit} imágenes por categoría")
        
        temp_dir = self.config.OUTPUT_DIR / f"temp_limited_resnet_dataset_{image_limit}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Directorio temporal creado: {temp_dir}")
        
        total_images = {'train': 0, 'val': 0, 'test': 0}
        
        for split in ['train', 'val', 'test']:
            split_src = self.config.IMAGES_DIR / split
            split_dst = temp_dir / split
            split_dst.mkdir(parents=True, exist_ok=True)
            
            if not split_src.exists():
                self.logger.warning(f"⚠️  Split {split} no encontrado")
                continue
            
            self.logger.info(f"\n   Procesando split: {split}")
            
            for category in self.config.CLASSES:
                category_src = split_src / category
                category_dst = split_dst / category
                category_dst.mkdir(parents=True, exist_ok=True)
                
                if not category_src.exists():
                    self.logger.warning(f"      ⚠️  Categoría {category} no encontrada")
                    continue
                
                self.logger.info(f"       Categoría: {category}")
                
                images = list(category_src.glob('*.jpg')) + list(category_src.glob('*.png'))
                images = images[:image_limit]
                
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
# TRANSFORMACIONES Y DATALOADERS
# ============================================================================

def get_data_transforms(image_size=224):
    """Transformaciones - ResNet50 usa 224x224"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def create_dataloaders(data_path, batch_size, num_workers, imgsz=224, cache_mode='none'):
    """
    Crea dataloaders para train/val/test
    
    CORRECCIÓN: pin_memory=False para evitar ConnectionResetError
    cache_mode: 'none' o 'disk' (simula cache con persistent_workers)
    """
    data_path = Path(data_path)
    transforms_dict = get_data_transforms(imgsz)
    
    image_datasets = {
        x: datasets.ImageFolder(data_path / x, transforms_dict[x])
        for x in ['train', 'val', 'test']
        if (data_path / x).exists()
    }
    
    # Simular cache con persistent_workers
    use_persistent = (cache_mode == 'disk' and num_workers > 0)
    
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers,
            pin_memory=False,  # CORRECCIÓN: False para evitar ConnectionResetError
            persistent_workers=use_persistent,  # Simula cache
            drop_last=(x == 'train')  # Evita batches pequeños
        )
        for x in image_datasets
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names


# ============================================================================
# MODELO
# ============================================================================

def create_resnet50(num_classes, pretrained=True, freeze_layers=0):
    """Crea modelo ResNet50"""
    model = models.resnet50(pretrained=pretrained)
    
    # Congelar capas
    if freeze_layers > 0:
        ct = 0
        for child in model.children():
            ct += 1
            if ct < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
    
    # Modificar capa de clasificación
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


# ============================================================================
# GRID SEARCH ENGINE
# ============================================================================

class GridSearchEngine:
    """Motor de Grid Search para ResNet50"""
    
    def __init__(self, config, logger, experiment_name):
        self.config = config
        self.logger = logger
        self.experiment_name = experiment_name
        self.monitor = ResourceMonitor()
        
        self.results = []
        self.best_combination = None
        self.best_accuracy = 0.0
        
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
    
    def train_combination(self, combo_id, hyperparams, data_path, num_classes, class_names):
        """Entrena una combinación específica"""
        
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"COMBINACIÓN {combo_id}: ResNet50")
            self.logger.info(f"{'='*80}")
            
            combo_dir = self.experiment_dir / f'combo_{combo_id:03d}'
            combo_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Hiperparámetros:")
            for key, value in hyperparams.items():
                self.logger.info(f"   {key}: {value}")
            self.logger.info("=" * 80)
            sys.stdout.flush()
            
            # Setup device
            device_list = hyperparams.get('device', [0])
            if isinstance(device_list, list) and len(device_list) > 1:
                device = torch.device(f"cuda:{device_list[0]}")
                self.logger.info(f"🔧 Usando Multi-GPU: {device_list}")
                use_multi_gpu = True
            else:
                device = torch.device(f"cuda:{device_list[0] if isinstance(device_list, list) else device_list}")
                use_multi_gpu = False
            
            # Crear dataloaders
            batch_size = hyperparams.get('batch', 32)
            num_workers = hyperparams.get('workers', 4)
            imgsz = hyperparams.get('imgsz', 224)
            cache_mode = hyperparams.get('cache', 'none')
            
            self.logger.info(f"📂 Cargando dataset con cache={cache_mode}...")
            
            dataloaders, dataset_sizes, _ = create_dataloaders(
                data_path, batch_size, num_workers, imgsz, cache_mode
            )
            
            self.logger.info(f"📊 Dataset sizes:")
            for split, size in dataset_sizes.items():
                self.logger.info(f"   {split}: {size} imágenes")
            
            # Crear modelo
            model = create_resnet50(
                num_classes=num_classes,
                pretrained=hyperparams.get('pretrained', True),
                freeze_layers=hyperparams.get('freeze_layers', 0)
            )
            
            model = model.to(device)
            
            # Multi-GPU setup
            if use_multi_gpu:
                model = DataParallel(model, device_ids=device_list)
                self.logger.info(f"✅ DataParallel configurado con GPUs: {device_list}")
            
            # Loss y optimizer
            criterion = nn.CrossEntropyLoss()
            
            optimizer_name = hyperparams.get('optimizer', 'SGD')
            lr = hyperparams.get('lr0', 0.001)
            
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=hyperparams.get('weight_decay', 0.0001)
                )
            else:  # SGD
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=hyperparams.get('momentum', 0.9),
                    weight_decay=hyperparams.get('weight_decay', 0.0001)
                )
            
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hyperparams.get('step_size', 7),
                gamma=hyperparams.get('gamma', 0.1)
            )
            
            # Entrenamiento
            self.monitor.start()
            train_start = time.time()
            
            epochs = hyperparams.get('epochs', 30)
            best_acc = 0.0
            best_model_wts = None
            
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            self.logger.info(f"\n🚀 Iniciando entrenamiento ({epochs} épocas)...")
            sys.stdout.flush()
            
            for epoch in range(epochs):
                self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
                self.logger.info("-" * 40)
                
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()
                    
                    running_loss = 0.0
                    running_corrects = 0
                    
                    # Progress bar
                    pbar = tqdm(dataloaders[phase], desc=f"{phase}", leave=False)
                    
                    for inputs, labels in pbar:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        optimizer.zero_grad()
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            # ResNet50 siempre retorna Tensor (no como InceptionV3)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
                    if phase == 'train':
                        scheduler.step()
                    
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    
                    history[f'{phase}_loss'].append(epoch_loss)
                    history[f'{phase}_acc'].append(float(epoch_acc))
                    
                    self.logger.info(f"{phase.capitalize()}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model.state_dict().copy()
                        self.logger.info(f"   ⭐ Nuevo mejor modelo!")
            
            train_time = time.time() - train_start
            
            # Guardar mejor modelo
            weights_dir = combo_dir / 'weights'
            weights_dir.mkdir(exist_ok=True)
            
            if best_model_wts:
                torch.save(best_model_wts, weights_dir / 'best.pt')
            
            torch.save(model.state_dict(), weights_dir / 'last.pt')
            
            # Métricas
            metrics = {
                'combination_id': combo_id,
                'model': 'ResNet50',
                'hyperparameters': hyperparams,
                'training_time_seconds': train_time,
                'training_time_hours': train_time / 3600,
                'metrics': {
                    'best_val_accuracy': float(best_acc),
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_train_acc': history['train_acc'][-1],
                    'final_val_acc': history['val_acc'][-1],
                },
                'history': history,
                'resources': self.monitor.get_metrics(),
                'paths': {
                    'best_weights': str(weights_dir / 'best.pt'),
                    'results_dir': str(combo_dir)
                },
                'success': True,
                'error': None
            }
            
            # Actualizar mejor combinación
            if metrics['metrics']['best_val_accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['metrics']['best_val_accuracy']
                self.best_combination = metrics
            
            # Guardar métricas
            save_json(metrics, combo_dir / 'metrics.json')
            save_json(history, combo_dir / 'history.json')
            
            # Guardar gráfica de training
            self.plot_training_history(history, combo_dir)
            
            self.logger.info(f"\n✅ Combinación {combo_id} completada")
            self.logger.info(f"   Best Val Accuracy: {float(best_acc):.4f}")
            self.logger.info(f"   Tiempo: {train_time/60:.2f} minutos")
            
        except Exception as e:
            self.logger.error(f"❌ Error en combinación {combo_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            metrics = {
                'combination_id': combo_id,
                'model': 'ResNet50',
                'hyperparameters': hyperparams,
                'success': False,
                'error': str(e)
            }
        
        return metrics
    
    def plot_training_history(self, history, save_dir):
        """Grafica el historial de entrenamiento"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss
            ax1.plot(history['train_loss'], label='Train')
            ax1.plot(history['val_loss'], label='Val')
            ax1.set_title('Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            ax2.plot(history['train_acc'], label='Train')
            ax2.plot(history['val_acc'], label='Val')
            ax2.set_title('Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"No se pudo crear gráfica: {e}")
    
    def cleanup_gpu_after_combination(self):
        """Liberar memoria GPU después de cada combinación"""
        import gc
        
        self.logger.info("🧹 Liberando memoria GPU...")
        sys.stdout.flush()
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.info(f"   GPU {i}: {mem_allocated:.2f}GB allocated, "
                               f"{mem_reserved:.2f}GB reserved")
        
        self.logger.info("✅ GPU limpia")
        sys.stdout.flush()
    
    def save_results(self):
        """Guarda todos los resultados"""
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
            self.logger.info(f"   Best Val Accuracy: {self.best_combination['metrics']['best_val_accuracy']:.4f}")
            self.logger.info(f"   Hiperparámetros:")
            for key, value in self.best_combination['hyperparameters'].items():
                self.logger.info(f"      {key}: {value}")
        
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
        description='Grid Search para ResNet50 Clasificación'
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'both'],
                       default='both', help='Modo de ejecución')
    parser.add_argument('--experiment_name', type=str, default='grid_search_resnet50',
                       help='Nombre del experimento')
    parser.add_argument('--hyperparam', action='append', nargs='+',
                       help='Hiperparámetros en formato key=val1,val2,val3')
    parser.add_argument('--image-limit', type=int, default=0,
                       help='Límite de imágenes por categoría (0 = todas)')
    parser.add_argument('--base-dir', type=str, default='/workspace',
                       help='Directorio base')
    
    args = parser.parse_args()
    
    # Configurar paths
    config = Config()
    config.BASE_DIR = Path(args.base_dir)
    config.DATASET_DIR = config.BASE_DIR / "DSM"
    config.IMAGES_DIR = config.DATASET_DIR / "images"
    config.OUTPUT_DIR = config.BASE_DIR / "outputs"
    config.MODELS_DIR = config.BASE_DIR / "models/phase2_classification/resnet50"
    
    # Setup logging
    logger, log_file = setup_logging(args.experiment_name)
    
    logger.info("="*80)
    logger.info("🚀 GRID SEARCH PARA RESNET50 CLASIFICACIÓN")
    logger.info("="*80)
    logger.info(f"\nExperimento: {args.experiment_name}")
    logger.info(f"Modelo: ResNet50")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Límite de imágenes: {args.image_limit}")
    
    # Parsear hiperparámetros
    hyperparams_grid = dict(config.DEFAULT_HYPERPARAMS)
    
    if args.hyperparam:
        logger.info(f"\n📝 Hiperparámetros configurados:")
        
        # Detectar Multi-GPU
        for param_list in args.hyperparam:
            for param in param_list:
                if '=' in param:
                    key, value = param.split('=', 1)
                    if key == 'device' and ',' in value:
                        logger.info(f"\n✅ Multi-GPU detectado: device={value}")
                        logger.info(f"   Las GPUs trabajarán en PARALELO usando DataParallel")
                        logger.info(f"   Batch se dividirá entre las GPUs\n")
                        sys.stdout.flush()
        
        for param_list in args.hyperparam:
            for param in param_list:
                if '=' in param:
                    key, values_str = param.split('=', 1)
                    values = parse_hyperparam_list(values_str, param_name=key)
                    hyperparams_grid[key] = values
                    logger.info(f"   {key}: {values}")
    
    # Calcular combinaciones
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
    
    # Entrenar cada combinación
    if args.mode in ['train', 'both']:
        for i, combo in enumerate(tqdm(combinations, desc="Grid Search"), 1):
            result = grid_search.train_combination(
                combo_id=i,
                hyperparams=combo,
                data_path=data_path,
                num_classes=config.NUM_CLASSES,
                class_names=config.CLASSES
            )
            grid_search.results.append(result)
            
            # Limpiar GPU
            grid_search.cleanup_gpu_after_combination()
        
        # Guardar resultados
        grid_search.save_results()
    
    logger.info("\n✅ Grid Search completado exitosamente")
    logger.info(f"📁 Resultados en: {grid_search.experiment_dir}")
    logger.info(f"📄 Log file: {log_file}\n")
    
    # Limpiar recursos
    logger.info("🧹 Cerrando procesos y liberando recursos...")
    sys.stdout.flush()
    
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Recursos liberados correctamente\n")
    sys.stdout.flush()
    
    sys.exit(0)


if __name__ == "__main__":
    main()
