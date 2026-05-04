"""
FASE 3: Fine-tuning LLaVA-NeXT para Clasificación VLM
======================================================
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
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
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

class LLaVADataset(Dataset):
    """Dataset para LLaVA-NeXT con pre-filtrado"""
    
    def __init__(self, data_path, processor, max_tokens=2048, sample_limit=None):
        self.processor = processor
        self.max_tokens = max_tokens
        
        print(f"\n📥 Cargando dataset: {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        if sample_limit and len(raw_data) > sample_limit:
            raw_data = raw_data[:sample_limit]
        
        print(f"📊 Total de muestras: {len(raw_data)}")
        
        # Pre-filtrar muestras largas
        self.valid_samples = []
        print(f"🔍 Pre-filtrando muestras (max {max_tokens} tokens)...")
        
        for item in tqdm(raw_data, desc="Filtrado"):
            try:
                text = self._build_text(item['conversations'])
                image_path = item['image']
                if not image_path.startswith('/'):
                    image_path = f"/workspace/{image_path}"
                image = Image.open(image_path).convert('RGB')
                
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                )
                
                num_tokens = inputs['input_ids'].shape[1]
                image.close()
                
                if num_tokens <= self.max_tokens:
                    self.valid_samples.append(item)
                
            except Exception:
                continue
        
        print(f"✓ {len(self.valid_samples)}/{len(raw_data)} muestras válidas ({len(self.valid_samples)*100//len(raw_data)}%)")
    
    def _build_text(self, conversations):
        text = ""
        for conv in conversations:
            role = "user" if conv['from'] == "human" else "assistant"
            content = conv['value'].strip()
            if role == "user":
                text += f"USER: {content}\n"
            else:
                text += f"ASSISTANT: {content}\n"
        return text
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        
        try:
            text = self._build_text(item['conversations'])
            image_path = item['image']
            if not image_path.startswith('/'):
                image_path = f"/workspace/{image_path}"
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            
            image.close()
            
            result = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['input_ids'].squeeze(0).clone()
            }
            
            if 'pixel_values' in inputs:
                result['pixel_values'] = inputs['pixel_values'].squeeze(0)
            
            if 'image_sizes' in inputs:
                result['image_sizes'] = inputs['image_sizes'].squeeze(0)
            
            return result
            
        except Exception:
            return self._get_fallback()
    
    def _get_fallback(self):
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        
        try:
            text = "USER: <image>\nDescribe this image.\nASSISTANT: This is a test image.\n"
            inputs = self.processor(
                text=text,
                images=dummy_image,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            
            result = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['input_ids'].squeeze(0).clone()
            }
            
            if 'pixel_values' in inputs:
                result['pixel_values'] = inputs['pixel_values'].squeeze(0)
            
            if 'image_sizes' in inputs:
                result['image_sizes'] = inputs['image_sizes'].squeeze(0)
            
            return result
            
        except Exception:
            return {
                'input_ids': torch.zeros(256, dtype=torch.long),
                'attention_mask': torch.ones(256, dtype=torch.long),
                'labels': torch.full((256,), -100, dtype=torch.long),
                'pixel_values': torch.zeros((3, 224, 224)),
                'image_sizes': torch.tensor([224, 224])
            }

@dataclass
class SimpleCollator:
    def __call__(self, features: List[Dict]):
        if len(features) != 1:
            raise ValueError(f"Requiere batch_size=1, recibió {len(features)}")
        return {k: v.unsqueeze(0) for k, v in features[0].items()}

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning LLaVA-NeXT')
    
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/workspace/models/phase3_vlm/llava-next')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--base_model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf')
    parser.add_argument('--max_tokens', type=int, default=2048)
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
    print("🚀 LLaVA-NeXT FINE-TUNING")
    print("="*80)
    print(f"Experimento: {args.experiment_name}")
    print(f"Modelo: {args.base_model}")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Cargando processor y modelo...")
    processor = LlavaNextProcessor.from_pretrained(args.base_model)
    
    # Cargar modelo SIN cuantización
    model = LlavaNextForConditionalGeneration.from_pretrained(
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
    
    train_dataset = LLaVADataset(args.train_data, processor, args.max_tokens, args.sample_limit)
    val_dataset = LLaVADataset(args.val_data, processor, args.max_tokens, args.sample_limit)
    
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
