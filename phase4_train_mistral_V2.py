"""
FASE 4: Fine-tuning Mistral-7B-Instruct
=======================================
Plataforma: RunPod (2× RTX A4500, 62GB RAM, 16 vCPU)
Dataset: DSM LLM (Alpaca format)
Versión: SIN cuantización 4-bit (más compatible)
"""

import argparse
import json
import time
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
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

class AlpacaDataset(Dataset):
    """Dataset para formato Alpaca"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"\n📥 Cargando dataset: {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"📊 Total de muestras: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            
            encodings = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as e:
            print(f"⚠️  Error en muestra {idx}: {e}")
            dummy_text = "### Instruction:\nDescribe this.\n\n### Response:\nThis is a test."
            encodings = self.tokenizer(
                dummy_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": encodings["input_ids"].squeeze(0)
            }

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning Mistral-7B')
    
    parser.add_argument('--train_data', type=str, default='/workspace/outputs/llm_train_alpaca.json')
    parser.add_argument('--val_data', type=str, default='/workspace/outputs/llm_val_alpaca.json')
    parser.add_argument('--output_dir', type=str, default='/workspace/models/phase4_llm/mistral-7b')
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=1)
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 MISTRAL-7B FINE-TUNING")
    print("="*80)
    print(f"Experimento: {args.experiment_name}")
    print(f"Modelo: {args.base_model}")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Cargar modelo SIN cuantización
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Modelo cargado")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {mem_alloc:.2f}GB / {mem_reserved:.2f}GB")
    
    train_dataset = AlpacaDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = AlpacaDataset(args.val_data, tokenizer, args.max_length)
    
    print(f"\n📊 Datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print(f"\n🎯 Iniciando entrenamiento...")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_effective = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
    print(f"   Batch efectivo: {batch_effective}")
    
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    print(f"\n✓ Completado en {train_time/3600:.2f} horas")
    
    print("💾 Guardando modelo...")
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
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
