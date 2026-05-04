"""
FASE 3: Fine-tuning Qwen2-VL para Clasificación VLM
====================================================
Plataforma: RunPod (2× RTX A6000, 125GB RAM, 32 vCPU)
Dataset: DSM VLM (ShareGPT format)
Versión: SIN cuantización 4-bit (más compatible)
"""

import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
import warnings
import gc

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class Qwen2VLDataset(Dataset):
    """Dataset para Qwen2-VL"""
    
    def __init__(self, data_path, processor, sample_limit=None):
        self.processor = processor
        
        print(f"\n📥 Cargando dataset: {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if sample_limit and len(self.data) > sample_limit:
            self.data = self.data[:sample_limit]
        
        print(f"📊 Total de muestras: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Cargar imagen
            image_path = item['image']
            if not image_path.startswith('/'):
                image_path = f"/workspace/{image_path}"
            image = Image.open(image_path).convert('RGB')
            
            # Construir mensajes
            messages = []
            for conv in item['conversations']:
                role = "user" if conv['from'] == "human" else "assistant"
                content_value = conv['value'].strip()
                content_list = []
                
                if role == "user" and "<image>" in content_value:
                    content_list.append({"type": "image", "image": image})
                    text_part = content_value.replace("<image>", "").strip()
                    if text_part:
                        content_list.append({"type": "text", "text": text_part})
                else:
                    content_list.append({"type": "text", "text": content_value})
                
                messages.append({"role": role, "content": content_list})
            
            # Procesar
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            image.close()
            
            result = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['input_ids'].squeeze(0).clone()
            }
            
            if 'pixel_values' in inputs:
                result['pixel_values'] = inputs['pixel_values'].squeeze(0)
            if 'image_grid_thw' in inputs:
                result['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
            
            return result
            
        except Exception:
            return self._fallback()
    
    def _fallback(self):
        dummy = Image.new('RGB', (224, 224), color='gray')
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": dummy}, {"type": "text", "text": "Describe."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Test."}]}
        ]
        try:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(text=[text], images=[dummy], return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['input_ids'].squeeze(0).clone(),
                'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else torch.zeros((1,3,224,224)),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else torch.tensor([1,16,16])
            }
        except:
            return {
                'input_ids': torch.zeros(128, dtype=torch.long),
                'attention_mask': torch.ones(128, dtype=torch.long),
                'labels': torch.full((128,), -100, dtype=torch.long),
                'pixel_values': torch.zeros((1, 3, 224, 224)),
                'image_grid_thw': torch.tensor([1, 16, 16], dtype=torch.long)
            }

@dataclass
class SimpleCollator:
    def __call__(self, features: List[Dict]):
        if len(features) != 1:
            raise ValueError(f"Requiere batch_size=1, recibió {len(features)}")
        return {k: v.unsqueeze(0) for k, v in features[0].items()}

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning Qwen2-VL')
    
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/workspace/models/phase3_vlm/qwen2-vl')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--sample_limit', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 QWEN2-VL FINE-TUNING")
    print("="*80)
    print(f"Experimento: {args.experiment_name}")
    print(f"Modelo: {args.base_model}")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Cargando processor y modelo...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    # Cargar modelo SIN cuantización
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Configurar LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Modelo cargado")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {mem_alloc:.2f}GB / {mem_reserved:.2f}GB")
    
    train_dataset = Qwen2VLDataset(args.train_data, processor, args.sample_limit)
    val_dataset = Qwen2VLDataset(args.val_data, processor, args.sample_limit)
    
    print(f"\n📊 Datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    data_collator = SimpleCollator()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_torch",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print(f"\n🎯 Iniciando entrenamiento...")
    print(f"   Batch efectivo: {args.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else args.gradient_accumulation_steps}")
    
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    print(f"\n✓ Completado en {train_time/3600:.2f} horas")
    
    print("💾 Guardando modelo...")
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")
    
    metrics = {
        'experiment': args.experiment_name,
        'train_time_hours': train_time / 3600,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Resultados en: {output_dir}")
    clear_memory()

if __name__ == "__main__":
    main()
