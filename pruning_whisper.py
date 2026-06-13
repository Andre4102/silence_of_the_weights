import os
import re
import json
import unicodedata
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor
)
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import autocast

from whisper_utils import load_model
from custom_attention import LayerWiseWhisperConfig
import config_whisper as config
from lora import LoRAWrapper, merge_lora_weights
from utils import (set_seed, 
                   capture_encoder_layer_names, 
                   get_layers, 
                   get_layer_weights,
                   count_parameters,
                   log_sparsity,
                   log_heatmap,
                   build_warmup_scheduler,
                   update_config_from_model)
from structured_pruning_utils_fisher_rope import (get_enum, 
                                                  PruningStrategy, 
                                                  ThresholdStrategy, 
                                                  ImportanceStrategy, 
                                                  structured_prune_model as prune_model, 
                                                  FisherConfig, 
                                                  PruningChanges,
                                                  get_model_sparsity_stats)
from whisper_utils import merge_datasets
from whisper_eval import evaluate_all_languages


def one_epoch_train(model, processor, dataloader, max_grad_norm, gradient_accumulation_steps, optimizer, scheduler, epoch, device):
    total_training_loss = 0.0
    model.train()
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training, loss: {total_training_loss}")):
        optimizer.zero_grad()

        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        batch = processor(batch)
        output = model(batch)

        loss = output.loss
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            #Use set_to_none=True
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        total_training_loss += loss.item()

def compute_distillation_loss(
    student_logits,        # (B, T, vocab)
    teacher_logits,        # (B, T, vocab)
    labels,                # (B, T) ground truth token ids
    temperature=2.0,
    alpha=0.5              # weight between CE and KL loss
):
    """
    Combines:
      - Cross-entropy loss with ground truth labels (standard ASR loss)
      - KL divergence loss with teacher soft targets (distillation)

    alpha=1.0 => pure distillation, alpha=0.0 => pure CE with labels
    """
    # --- Standard CE loss with hard labels ---
    # logits: (B, T, vocab) -> (B*T, vocab); labels: (B, T) -> (B*T,)
    B, T, V = student_logits.shape
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100
    )

    # --- KL Divergence with teacher soft targets ---
    # Scale logits by temperature, then compute soft distributions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)   # (B, T, V)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)        # (B, T, V)

    # Mask padding positions (where label == -100)
    mask = (labels != -100).unsqueeze(-1).float()                              # (B, T, 1)
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none'
    )  # (B, T, V)
    kl_loss = (kl_loss * mask).sum() / mask.sum().clamp(min=1)

    # Scale KL loss by T^2 (standard practice) and combine
    distill_loss = (temperature ** 2) * kl_loss
    total_loss = alpha * distill_loss + (1.0 - alpha) * ce_loss

    return total_loss, ce_loss.detach(), distill_loss.detach()

def one_epoch_train_distill(
    student_model,
    teacher_model,
    processor,
    dataloader,
    max_grad_norm,
    gradient_accumulation_steps,
    optimizer,
    scheduler,
    epoch,
    device,
    temperature=2.0,
    alpha=0.5,
    writer=None,
    global_step=0
):
    student_model.train()
    teacher_model.eval()  # teacher always frozen

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Distillation")):
        # Collate fn already produced processed tensors — move to device directly
        input_features = batch["input_features"].to(device, non_blocking=True)   # (B, 80, T)
        labels = batch["labels"].to(device, non_blocking=True)            # (B, T)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        # Whisper expects decoder_input_ids = labels shifted right.
        # The tokenizer already prepends the language/task/SOT tokens,
        # so we just replace -100 padding with pad_token_id then drop the last token.
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = processor.tokenizer.pad_token_id
        decoder_input_ids = decoder_input_ids[:, :-1].contiguous()   # (B, T-1)
        labels_for_loss   = labels[:, 1:].contiguous()                # (B, T-1), shifted

        # --- Student forward (fp16) ---
        with autocast('cuda', dtype=torch.bfloat16):
            student_out = student_model(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels_for_loss,
            )
        student_logits = student_out.logits   # (B, T-1, vocab)

        # --- Teacher forward (fp16, no grad) ---
        with torch.no_grad():
            teacher_out = teacher_model(
                input_features=input_features.half(),
                attention_mask=attention_mask.half() if attention_mask is not None else None,
                decoder_input_ids=decoder_input_ids,
            )
            # Cast teacher logits back to fp32 for the loss computation
            teacher_logits = teacher_out.logits.float()   # (B, T-1, vocab)

        # Align sequence lengths in case of mismatch (student may be smaller after pruning)
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        labels_clipped = labels_for_loss[:, :min_len]

        loss, ce_loss, kl_loss = compute_distillation_loss(
            student_logits, teacher_logits, labels_clipped,
            temperature=temperature,
            alpha=alpha
        )

        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        total_ce   += ce_loss.item()
        total_kl   += kl_loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if writer is not None:
                writer.add_scalar("Train/loss_total", total_loss / (step + 1), global_step)
                writer.add_scalar("Train/loss_ce",    total_ce   / (step + 1), global_step)
                writer.add_scalar("Train/loss_kl",    total_kl   / (step + 1), global_step)
            global_step += 1

    return total_loss / len(dataloader), global_step

def main(pruning_strategy, threshold_strategy, importance_strategy, epochs, lr, optim_name, results_root, num_iterations=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = config.MODEL_NAME
    batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps

    iterations = config.num_iterations if num_iterations is None else num_iterations
    prune_amount = config.prune_amount_per_iteration

    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
    cache_root = getattr(config, "subset_data_path", None)

    save_dir = os.path.join(results_root, f'{threshold_strategy.value}_{pruning_strategy.value}_{importance_strategy.value}')
    # Set random seeds
    set_seed(config.seed)
    os.makedirs(save_dir, exist_ok=True)

    optimizer_state = None 
    scheduler_state = None
    step = 0
    start_iteration = 0
    writer = SummaryWriter(log_dir=os.path.join(save_dir, f"tensorboard_logs/"))

    paths = {
        "librispeech": config.LIBRISPEECH_ROOT,
        "mls": config.MLS_ROOT,
        "commonvoice": config.COMMON_VOICE_ROOT,
        "fleurs": config.FLEURS_ROOT,
        "covost": config.COVOST_ROOT,
        "voxpopuli": config.VOXPOPULI_ROOT,
    }
    
    model, processor = load_model(model_name)
    train_loader = merge_datasets(paths, processor, batch_size, True)
    fisher_loader = merge_datasets(paths, processor, 1, False)
    
    fisher_config = FisherConfig(config.fisher_num_samples, 
                                 config.fisher_damping, 
                                 config.fisher_use_diagonal) if config.importance_strategy == ImportanceStrategy.FISHER_INFORMATION else None
    changes = PruningChanges()

    if not hasattr(model, "_encoder_layer_names"):
        model._encoder_layer_names = capture_encoder_layer_names(model)

    attention_layers = get_layers(model)
    # print(attention_layers)

    # Log heatmaps before pruning
    for i, sa in enumerate(attention_layers):
        #log sparsity of initial model
        writer.add_scalar(f"Sparsity/Layer_{i}_in_proj_weight", 0, 0)
        #log the attention heatmaps of the initial model
        tag = f'Attention_Heatmaps/Layer_{i}_in_proj_weight'
        log_heatmap(writer, sa, tag=tag, iteration=0, order = config.order, strategy_name=pruning_strategy.value)
        _, _, _, _, embed_dim, num_heads, head_dims= get_layer_weights(sa)
        
        num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
        head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
        #log model stats (only needed if structural pruning)
        changes.original_config[i] = {
            'embed_dim': embed_dim,
            'num_heads_q': num_heads_q,
            'num_heads_k': num_heads_k,
            'num_heads_v': num_heads_v,
            'num_heads_o': num_heads_o,
            'head_dim_q': head_dim_q,
            'head_dim_k': head_dim_k,
            'head_dim_v': head_dim_v,
            'head_dim_o': head_dim_o
        }

    # inputs = torch.randn(1, 3, 224, 224).to(device)

    # flops = FlopCountAnalysis(model, inputs)
    # print(f"Vision FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    # writer.add_scalar('flops (GFLOPS)', flops.total() / 1e9, 0)
    if not config.USE_LORA:
        teacher_model, _ = load_model(config.teacher_path)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        # Cast to fp16 to halve teacher VRAM footprint
        teacher_model = teacher_model.half()
    model.to(device)
    for it in range(start_iteration, iterations, 1):
        #prune the model
        model = prune_model(model, prune_amount=prune_amount, pruning_strategy=pruning_strategy, 
                            threshold_strategy=threshold_strategy, importance_strategy=importance_strategy, 
                            fisher_data_loader= fisher_loader, fisher_config=fisher_config,
                            order=config.order, device=device, teacher=teacher_model, processor=processor)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        param_count = count_parameters(model)
        ckpt_dir = os.path.join(save_dir, f"model_{param_count//1_000_000}M_params")
        # debug_pruning(model, test_loader, device)
        model.to(device)
        
        # Collect all self-attention layers
        attention_layers = get_layers(model)
        # Log heatmaps after pruning
        for i, sa in enumerate(attention_layers):
            tag = f'Attention_Heatmaps/Layer_{i}_in_proj_weight'
            log_heatmap(writer, sa, tag=tag, iteration=it+1, order = config.order, strategy_name=pruning_strategy.value)
        
        final_stats = get_model_sparsity_stats(model, changes, detailed=True)
        log_sparsity(writer, final_stats['layers'], it+1)
        sparsity = final_stats['overall']['sparsity']

        # results = evaluate_all_languages(model, processor, device)

        # inputs = torch.randn(1, 3, 224, 224).to(device)

        # flops = FlopCountAnalysis(model, inputs)
        # print(f"Vision FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        # writer.add_scalar('flops (GFLOPS)', flops.total() / 1e9, sparsity*100)
        if config.USE_LORA:
            model = LoRAWrapper(model, rank=getattr(config, "LORA_RANK", 16), alpha=getattr(config, "LORA_ALPHA", 16.0)).to(device)
            
            params = [p for _, p in model.named_parameters() if p.requires_grad]
            base_params = [p for _, p in model.named_parameters() if not p.requires_grad]

            # Optional: safety assert
            assert all(not bp.requires_grad for bp in base_params), "Base weights must be frozen for LoRA FT"
            # optimizer = optim.AdamW(lora_params, lr=getattr(config, "LORA_LR", 5e-4), weight_decay=0.0)

        else:
            # standard full-model training
            for p in model.parameters():
                p.requires_grad = True
            params = model.parameters()
        
        if optim_name == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(params, lr, weight_decay=1e-4)

        scheduler = build_warmup_scheduler(optimizer, epochs, len(train_loader), gradient_accumulation_steps, config.warmup_steps)

        for epoch in range(epochs):
            global_step = 0  # or track across epochs
            if config.USE_LORA:
                one_epoch_train(model, processor, train_loader, max_grad_norm, gradient_accumulation_steps, optimizer, scheduler, epoch, device)
            else:
                avg_loss, global_step = one_epoch_train_distill(
                    student_model=model,
                    teacher_model=teacher_model,
                    processor=processor,
                    dataloader=train_loader,
                    max_grad_norm=max_grad_norm,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    device=device,
                    temperature=getattr(config, "distill_temperature", 2.0),
                    alpha=getattr(config, "distill_alpha", 0.5),
                    writer=writer,
                    global_step=global_step
                )
                print(f"Epoch {epoch+1} | avg loss: {avg_loss:.4f}")

        if isinstance(model, LoRAWrapper):
        #Remove LoRA, evaluate and start again
            model = merge_lora_weights(model)
        #load the best weights from the original model
        save_checkpoint(model, processor, ckpt_dir, iteration = it)
        # Persist attention-block sparsity (overall + per-layer) next to the
        # checkpoint so the later eval/benchmark stage can match WER/FLOPs/latency
        # to the exact attention sparsity without recomputing the pruning.
        with open(os.path.join(ckpt_dir, "sparsity.json"), "w") as f:
            json.dump({
                "iteration": it + 1,
                "param_count": param_count,
                "attention_sparsity": sparsity,
                "stats": final_stats,
            }, f, indent=2)
        model, processor, _, _, _, _ = load_checkpoint(ckpt_dir, device=device, load_lora=False)

        #absorb lora if added
        # results = evaluate_all_languages(model, processor, device)

def save_checkpoint(model, processor, model_path,  iteration, optimizer=None, scheduler=None, step=None):
    # update_config
    if hasattr(model, "module"):
        model_to_save = model.module
    else:
        model_to_save = model

    os.makedirs(model_path, exist_ok=True)
    
    if config.USE_LORA and isinstance(model_to_save, LoRAWrapper):
        model_config = model_to_save.model.config
        # update it from actual pruned model    
        model_config = update_config_from_model(model_to_save.model, model_config)
        model_to_save.model.config = model_config
    else:
        model_config = model_to_save.config
        # update it from actual pruned model    
        model_config = update_config_from_model(model_to_save, model_config)
        model_to_save.config = model_config
    
    ckpt_dir = os.path.join(model_path, f"checkpoint-{step}") if step is not None else model_path

    # cleanup old checkpoints
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
    for old_ckpt in checkpoints:
        old_path = os.path.join(model_path, old_ckpt)
        print(f"Removing old checkpoint: {old_path}")
        os.system(f"rm -rf {old_path}")

    os.makedirs(ckpt_dir, exist_ok=True)
    #save tokenizer and model
    processor.save_pretrained(ckpt_dir)

    if isinstance(model_to_save, LoRAWrapper):
        # Save base model (no LoRA)
        model_to_save.model.save_pretrained(ckpt_dir)
        # Save LoRA separately
        lora_path = os.path.join(ckpt_dir, "lora_weights.pt")
        model_to_save.save_lora_weights(lora_path)

        # Save training state
        torch.save({
            "step": step,
            "iteration": iteration,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        }, os.path.join(ckpt_dir, "training_state.bin"))
    else:
        #training is done, save the model (should already be merged)
        model_to_save.save_pretrained(ckpt_dir)
        torch.save({"iteration": iteration}, os.path.join(ckpt_dir, "training_state.bin"))

    print(f"Checkpoint saved to {ckpt_dir}")
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
def load_checkpoint(model_path, device, load_lora=False):
    base_model, processor = load_model(model_path)
    if load_lora:
        model = LoRAWrapper(base_model, rank=config.LORA_RANK, alpha=config.LORA_ALPHA)
        lora_path = os.path.join(model_path, "lora_weights.pt")
        if os.path.exists(lora_path):
            model.load_lora_weights(lora_path)
            print(f"Loaded LoRA weights from {lora_path}")
    else:
        model = base_model  # vanilla
    if not hasattr(model, "_encoder_layer_names"):
        model._encoder_layer_names = capture_encoder_layer_names(model)
    model.to(device)

    # Load optimizer/scheduler if present
    training_state_path = os.path.join(model_path, "training_state.bin")
    optimizer_state, scheduler_state, step = None, None, 0
    if os.path.exists(training_state_path):
        state = torch.load(training_state_path, map_location=device)
        optimizer_state = state.get("optimizer", None)
        scheduler_state = state.get("scheduler", None)
        step = state.get("step", 0)
        iteration = state.get("iteration", 0)

    return model, processor, optimizer_state, scheduler_state, step, iteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning_strategy", default=config.pruning_strategy)
    parser.add_argument("--threshold_strategy", default=config.threshold_strategy)
    parser.add_argument("--importance_strategy", default=config.importance_strategy)
    parser.add_argument("--epochs", default=config.num_epochs, type=int)
    parser.add_argument("--lr", default=config.learning_rate, type=float)
    parser.add_argument("--optim", default='sgd')
    parser.add_argument("--results_root", default=config.results_root)
    parser.add_argument("--num_iterations", default=None, type=int,
                        help="Override config.num_iterations (e.g. 1 for a quick smoke test)")

    args = parser.parse_args()

    # Convert strings to Enum objects
    pruning_strategy = get_enum(PruningStrategy, args.pruning_strategy) if isinstance(args.pruning_strategy, str) else args.pruning_strategy 
    threshold_strategy = get_enum(ThresholdStrategy, args.threshold_strategy) if isinstance(args.threshold_strategy, str) else args.threshold_strategy
    importance_strategy = get_enum(ImportanceStrategy, args.importance_strategy) if isinstance(args.importance_strategy, str) else args.importance_strategy
    main(pruning_strategy, threshold_strategy, importance_strategy, args.epochs, args.lr, args.optim, args.results_root, args.num_iterations)