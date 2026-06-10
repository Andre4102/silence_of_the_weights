"""
Advanced Structural Fisher Information Analysis (Unified & VRAM Optimized)
========================================================================
Computes advanced structured Fisher metrics (Token-Weighted, Bilinear, Subspace, 
and Functional) for structural units in LLaMA/Vicuna.

Memory Optimized: 
- Parameter metrics process layer-by-layer with CPU offloading.
- Functional metric runs entirely in activation space (Zero Weight Gradients).
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_from_disk
from tqdm import tqdm
import gc

def parse_args():
    p = argparse.ArgumentParser(description="Advanced Structural Fisher Analysis")
    p.add_argument("--model", default="lmsys/vicuna-7b-v1.5")
    p.add_argument("--metric", default="functional", choices=["token_weighted", "bilinear", "subspace", "functional"], help="Main metric to deeply analyze.")
    p.add_argument("--ablation", action="store_true", help="If set, runs Standard Loss backward pass to compare baselines.")
    p.add_argument("--subspace_k", type=int, default=1, help="Rank k for Structural Subspace")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1, help="Increase to 4 or 8 for massive speedups!")
    p.add_argument("--output_dir", default="./advanced_fisher_results")
    p.add_argument("--dataset", default="/data/pruning/data/sharegpt_unfiltered_tokenized")
    return p.parse_args()

class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_samples=None):
        self.dataset = load_from_disk(path).select(range(max_samples)) if max_samples else load_from_disk(path)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"input_ids": torch.tensor(item["input_ids"], dtype=torch.long), 
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long)}

def build_dataloader(dataset, tokenizer, batch_size):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# =========================================================================
# HELPER: Factory function to completely prevent Python closure memory leaks
# =========================================================================
def get_backward_hook(name, layer_idx, act_detached, is_attn, num_heads, hd, tracker_scores):
    def backward_hook(grad):
        with torch.no_grad():
            taylor = act_detached * grad.detach()
            if is_attn:
                B, T, _ = taylor.shape
                taylor = taylor.view(B, T, num_heads, hd)
            # Sum over sequence, square, sum over batch
            fisher = taylor.sum(dim=1).pow(2).sum(dim=0)
            tracker_scores[name][layer_idx] += fisher
    return backward_hook

def accumulate_metrics(model, dataloader, args, device):
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    hd = model.config.hidden_size // num_heads

    ablation_scores = {
        '1st_order_std': defaultdict(lambda: defaultdict(float)),
        '2nd_order_diag_std': defaultdict(lambda: defaultdict(float)),
        '2nd_order_full_std': defaultdict(lambda: defaultdict(float)),
        '2nd_order_diag_seq': defaultdict(lambda: defaultdict(float)),
        '2nd_order_full_seq': defaultdict(lambda: defaultdict(float)),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # FUNCTIONAL METRIC BRANCH (VRAM Optimized, Hook-Based)
    # ──────────────────────────────────────────────────────────────────────────
    if args.metric == "functional":
        print(f"🚀 Running in Functional mode with Batch Size {args.batch_size}")
        for param in model.parameters():
            param.requires_grad = False
        model.model.embed_tokens.weight.requires_grad = True

        tracker_scores = {
            'q': defaultdict(lambda: torch.zeros((num_heads, hd), device=device)),
            'k': defaultdict(lambda: torch.zeros((num_heads, hd), device=device)),
            'v': defaultdict(lambda: torch.zeros((num_heads, hd), device=device)),
            'o': defaultdict(lambda: torch.zeros((num_heads, hd), device=device)),
            'gate': defaultdict(lambda: torch.zeros(model.config.intermediate_size, device=device)),
            'up':   defaultdict(lambda: torch.zeros(model.config.intermediate_size, device=device)),
            'down': defaultdict(lambda: torch.zeros(model.config.intermediate_size, device=device)),
        }
        hooks = []

        def get_hook(name, layer_idx, is_input=False, is_attn=False):
            def forward_hook(module, input, output):
                target = input[0] if is_input else output
                if target.requires_grad:
                    # Detach immediately. Creates a new tensor without graph history.
                    act_detached = target.detach()
                    # Use factory to prevent `target` from being captured in a closure
                    hook = get_backward_hook(name, layer_idx, act_detached, is_attn, num_heads, hd, tracker_scores)
                    target.register_hook(hook)
            return forward_hook

        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(get_hook('q', i, False, True)))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(get_hook('k', i, False, True)))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(get_hook('v', i, False, True)))
            hooks.append(layer.self_attn.o_proj.register_forward_hook(get_hook('o', i, True, True)))

            hooks.append(layer.mlp.gate_proj.register_forward_hook(get_hook('gate', i, False, False)))
            hooks.append(layer.mlp.up_proj.register_forward_hook(get_hook('up', i, False, False)))
            hooks.append(layer.mlp.down_proj.register_forward_hook(get_hook('down', i, True, False)))

        model.train()
        n_samples_processed = 0
        
        for batch in tqdm(dataloader, desc="Computing Functional Fisher"):
            if n_samples_processed >= args.n_samples: break
            
            bsz = batch["input_ids"].size(0)
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)

            outputs = model(**batch)
            logits_shift = outputs.logits[..., :-1, :].contiguous()
            targets = batch["input_ids"][..., 1:].contiguous()
            
            loss_t = torch.nn.functional.cross_entropy(
                logits_shift.view(-1, logits_shift.size(-1)), targets.view(-1), reduction="none"
            ).view(targets.shape)
            
            mask = batch["attention_mask"][..., 1:].contiguous()
            probs = torch.softmax(logits_shift, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            w = entropy * mask
            
            # Loss is inherently mean() over batch size. 
            loss = ((w / (w.sum(dim=-1, keepdim=True) + 1e-8)) * (loss_t * mask)).sum(dim=-1).mean()
            
            # We multiply by `bsz` so the gradient returned by backward is the TRUE 
            # unscaled gradient, ensuring math stays invariant to batch size changes.
            (loss * bsz).backward()
            
            n_samples_processed += bsz
            
            # 🧹 Aggressive Garbage Collection to prevent any lingering memory
            del outputs, logits_shift, targets, loss_t, mask, probs, entropy, w, loss, batch
            torch.cuda.empty_cache()

        for h in hooks: h.remove()

        final_scores = { 'qk_channel': {}, 'vo_channel': {}, 'mlp_neuron': {}, 'head': {} }
        for i in range(num_layers):
            q, k, v, o = [tracker_scores[x][i].cpu() / n_samples_processed for x in ['q','k','v','o']]
            gate, up, down = [tracker_scores[x][i].cpu() / n_samples_processed for x in ['gate','up','down']]

            final_scores['qk_channel'][i] = (q + k).numpy()
            final_scores['vo_channel'][i] = (v + o).numpy()
            final_scores['head'][i] = (q + k + v + o).sum(dim=-1).numpy()
            final_scores['mlp_neuron'][i] = (gate + up + down).numpy()

        if args.ablation:
            print("⚠️ Note: Standard ablation curves are disabled in 'functional' mode because weight gradients are frozen.")

        return final_scores, ablation_scores

    # ──────────────────────────────────────────────────────────────────────────
    # PARAMETER METRICS BRANCH (Token Weighted, Bilinear, Subspace)
    # ──────────────────────────────────────────────────────────────────────────
    deep_scores = {
        'mlp_neuron': defaultdict(lambda: torch.zeros(model.config.intermediate_size, device=device)),
        'qk_channel': defaultdict(lambda: torch.zeros(num_heads * hd, device=device)),
        'vo_channel': defaultdict(lambda: torch.zeros(num_heads * hd, device=device)),
        'head':       defaultdict(lambda: torch.zeros(num_heads, device=device))
    }
    
    E_g_std = defaultdict(lambda: 0) 
    subspace_U = None
    subspace_lr = 1e-3
    model.train()
    n_batches = 0
    
    for batch in tqdm(dataloader, desc=f"Computing {args.metric} Metrics"):
        # We track batches here to maintain your previous math implementation
        if (n_batches * args.batch_size) >= args.n_samples: break
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if args.ablation:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**batch)
                outputs.loss.backward()
                
            for i, layer in enumerate(model.model.layers):
                w_q = layer.self_attn.q_proj.weight.detach().float().view(num_heads, hd, -1)
                g_q = layer.self_attn.q_proj.weight.grad.float().view(num_heads, hd, -1)
                E_g_std[(i, 'q')] += g_q.cpu()
                gw_q = (g_q * w_q).sum(dim=-1)
                del w_q, g_q 
                
                w_k = layer.self_attn.k_proj.weight.detach().float().view(num_heads, hd, -1)
                g_k = layer.self_attn.k_proj.weight.grad.float().view(num_heads, hd, -1)
                E_g_std[(i, 'k')] += g_k.cpu()
                gw_k = (g_k * w_k).sum(dim=-1)
                del w_k, g_k

                w_v = layer.self_attn.v_proj.weight.detach().float().view(num_heads, hd, -1)
                g_v = layer.self_attn.v_proj.weight.grad.float().view(num_heads, hd, -1)
                E_g_std[(i, 'v')] += g_v.cpu()
                gw_v = (g_v * w_v).sum(dim=-1)
                del w_v, g_v

                w_o = layer.self_attn.o_proj.weight.detach().float().view(-1, num_heads, hd)
                g_o = layer.self_attn.o_proj.weight.grad.float().view(-1, num_heads, hd)
                E_g_std[(i, 'o')] += g_o.cpu()
                gw_o = (g_o * w_o).sum(dim=0)
                del w_o, g_o

                w_gate = layer.mlp.gate_proj.weight.detach().float()
                g_gate = layer.mlp.gate_proj.weight.grad.float()
                E_g_std[(i, 'gate')] += g_gate.cpu()
                gw_gate = (g_gate * w_gate).sum(dim=-1)
                del w_gate, g_gate

                w_up = layer.mlp.up_proj.weight.detach().float()
                g_up = layer.mlp.up_proj.weight.grad.float()
                E_g_std[(i, 'up')] += g_up.cpu()
                gw_up = (g_up * w_up).sum(dim=-1)
                del w_up, g_up

                w_down = layer.mlp.down_proj.weight.detach().float()
                g_down = layer.mlp.down_proj.weight.grad.float()
                E_g_std[(i, 'down')] += g_down.cpu()
                gw_down = (g_down * w_down).sum(dim=0)
                del w_down, g_down

                ablation_scores['2nd_order_diag_std']['qk'][i] += (gw_q.pow(2) + gw_k.pow(2)).flatten()
                ablation_scores['2nd_order_diag_std']['vo'][i] += (gw_v.pow(2) + gw_o.pow(2)).flatten()
                ablation_scores['2nd_order_full_std']['qk'][i] += (gw_q + gw_k).pow(2).flatten()
                ablation_scores['2nd_order_full_std']['vo'][i] += (gw_v + gw_o).pow(2).flatten()
                ablation_scores['2nd_order_diag_std']['mlp'][i] += (gw_gate.pow(2) + gw_up.pow(2) + gw_down.pow(2))
                ablation_scores['2nd_order_full_std']['mlp'][i] += (gw_gate + gw_up + gw_down).pow(2)
                del gw_q, gw_k, gw_v, gw_o, gw_gate, gw_up, gw_down

            del outputs

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(**batch)
            logits = outputs.logits
            targets = batch["input_ids"][..., 1:].contiguous()
            logits_shift = logits[..., :-1, :].contiguous()
            loss_t = torch.nn.functional.cross_entropy(logits_shift.view(-1, logits_shift.size(-1)), targets.view(-1), reduction="none").view(targets.shape)
            probs = torch.softmax(logits_shift, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            mask = batch["attention_mask"][..., 1:].contiguous()
            w = entropy * mask
            loss_t = loss_t * mask
            loss = ( (w / (w.sum(dim=-1, keepdim=True) + 1e-8)) * loss_t ).sum(dim=-1).mean()
            loss.backward()

        struct_vecs = []
        for i, layer in enumerate(model.model.layers):
            w_q = layer.self_attn.q_proj.weight.detach().float().view(num_heads, hd, -1)
            g_q = layer.self_attn.q_proj.weight.grad.float().view(num_heads, hd, -1)
            gw_q = (g_q * w_q).sum(dim=-1)
            del w_q, g_q
            
            w_k = layer.self_attn.k_proj.weight.detach().float().view(num_heads, hd, -1)
            g_k = layer.self_attn.k_proj.weight.grad.float().view(num_heads, hd, -1)
            gw_k = (g_k * w_k).sum(dim=-1)
            del w_k, g_k
            
            w_v = layer.self_attn.v_proj.weight.detach().float().view(num_heads, hd, -1)
            g_v = layer.self_attn.v_proj.weight.grad.float().view(num_heads, hd, -1)
            gw_v = (g_v * w_v).sum(dim=-1)
            del w_v, g_v
            
            w_o = layer.self_attn.o_proj.weight.detach().float().view(-1, num_heads, hd)
            g_o = layer.self_attn.o_proj.weight.grad.float().view(-1, num_heads, hd)
            gw_o = (g_o * w_o).sum(dim=0)
            del w_o, g_o
            
            w_gate = layer.mlp.gate_proj.weight.detach().float()
            g_gate = layer.mlp.gate_proj.weight.grad.float()
            gw_gate = (g_gate * w_gate).sum(dim=-1)
            del w_gate, g_gate
            
            w_up = layer.mlp.up_proj.weight.detach().float()
            g_up = layer.mlp.up_proj.weight.grad.float()
            gw_up = (g_up * w_up).sum(dim=-1)
            del w_up, g_up
            
            w_down = layer.mlp.down_proj.weight.detach().float()
            g_down = layer.mlp.down_proj.weight.grad.float()
            gw_down = (g_down * w_down).sum(dim=0)
            del w_down, g_down

            if args.metric == "token_weighted":
                deep_scores['qk_channel'][i] += (gw_q.pow(2) + gw_k.pow(2)).flatten()
                deep_scores['vo_channel'][i] += (gw_v.pow(2) + gw_o.pow(2)).flatten()
                deep_scores['head'][i] += (gw_q.pow(2) + gw_k.pow(2) + gw_v.pow(2) + gw_o.pow(2)).sum(dim=-1)
                deep_scores['mlp_neuron'][i] += (gw_gate.pow(2) + gw_up.pow(2) + gw_down.pow(2))
            else:
                joint_qk = (gw_q + gw_k).pow(2)
                joint_vo = (gw_v + gw_o).pow(2)
                joint_mlp = (gw_gate + gw_up + gw_down).pow(2)
                
                deep_scores['qk_channel'][i] += joint_qk.flatten()
                deep_scores['vo_channel'][i] += joint_vo.flatten()
                deep_scores['head'][i] += joint_qk.sum(dim=-1) + joint_vo.sum(dim=-1)
                deep_scores['mlp_neuron'][i] += joint_mlp
            
            if args.metric == "subspace":
                struct_vecs.append((gw_q + gw_k).flatten())
                struct_vecs.append((gw_v + gw_o).flatten())
                struct_vecs.append((gw_gate + gw_up + gw_down).flatten())
                
            if args.ablation:
                ablation_scores['2nd_order_diag_seq']['qk'][i] += (gw_q.pow(2) + gw_k.pow(2)).flatten()
                ablation_scores['2nd_order_diag_seq']['vo'][i] += (gw_v.pow(2) + gw_o.pow(2)).flatten()
                ablation_scores['2nd_order_full_seq']['qk'][i] += (gw_q + gw_k).pow(2).flatten()
                ablation_scores['2nd_order_full_seq']['vo'][i] += (gw_v + gw_o).pow(2).flatten()
                ablation_scores['2nd_order_diag_seq']['mlp'][i] += (gw_gate.pow(2) + gw_up.pow(2) + gw_down.pow(2))
                ablation_scores['2nd_order_full_seq']['mlp'][i] += (gw_gate + gw_up + gw_down).pow(2)
                
            del gw_q, gw_k, gw_v, gw_o, gw_gate, gw_up, gw_down

        if args.metric == "subspace":
            V = torch.cat(struct_vecs) 
            if subspace_U is None:
                subspace_U = torch.randn(V.numel(), args.subspace_k, device=device)
                subspace_U, _ = torch.linalg.qr(subspace_U, mode='reduced')
            
            V_norm = V / (V.norm() + 1e-8)
            proj = torch.matmul(V_norm, subspace_U)
            subspace_U += subspace_lr * torch.outer(V_norm, proj)
            if n_batches % 10 == 0:
                subspace_U, _ = torch.linalg.qr(subspace_U, mode='reduced')

        n_batches += 1
        
        # Aggressive memory clear
        del outputs, logits_shift, targets, loss_t, mask, probs, entropy, w, loss, batch
        torch.cuda.empty_cache()

    final_scores = {k: {} for k in deep_scores.keys()}
    vec_idx = 0
    for i in range(num_layers):
        for stype in ['qk_channel', 'vo_channel', 'mlp_neuron', 'head']:
            score_tensor = deep_scores[stype][i] / max(1, n_batches)
            
            if args.metric == "subspace" and stype != "head":
                numel = score_tensor.numel()
                U_chunk = subspace_U[vec_idx : vec_idx + numel]
                proj_energy = torch.linalg.matrix_norm(U_chunk, dim=1).pow(2)
                score_tensor += proj_energy * score_tensor.mean() 
                vec_idx += numel
                
            final_scores[stype][i] = score_tensor.cpu().numpy()

    if args.ablation:
        for i in range(num_layers):
            w_q = model.model.layers[i].self_attn.q_proj.weight.detach().float().view(num_heads, hd, -1)
            gw_q = ((E_g_std[(i, 'q')].to(device) / n_batches) * w_q).sum(dim=-1)
            w_k = model.model.layers[i].self_attn.k_proj.weight.detach().float().view(num_heads, hd, -1)
            gw_k = ((E_g_std[(i, 'k')].to(device) / n_batches) * w_k).sum(dim=-1)
            ablation_scores['1st_order_std']['qk'][i] = torch.abs(gw_q + gw_k).flatten().cpu().numpy()
            del w_q, gw_q, w_k, gw_k

            w_v = model.model.layers[i].self_attn.v_proj.weight.detach().float().view(num_heads, hd, -1)
            gw_v = ((E_g_std[(i, 'v')].to(device) / n_batches) * w_v).sum(dim=-1)
            w_o = model.model.layers[i].self_attn.o_proj.weight.detach().float().view(-1, num_heads, hd)
            gw_o = ((E_g_std[(i, 'o')].to(device) / n_batches) * w_o).sum(dim=0)
            ablation_scores['1st_order_std']['vo'][i] = torch.abs(gw_v + gw_o).flatten().cpu().numpy()
            del w_v, gw_v, w_o, gw_o
            
            w_gate = model.model.layers[i].mlp.gate_proj.weight.detach().float()
            gw_gate = ((E_g_std[(i, 'gate')].to(device) / n_batches) * w_gate).sum(dim=-1)
            w_up = model.model.layers[i].mlp.up_proj.weight.detach().float()
            gw_up = ((E_g_std[(i, 'up')].to(device) / n_batches) * w_up).sum(dim=-1)
            w_down = model.model.layers[i].mlp.down_proj.weight.detach().float()
            gw_down = ((E_g_std[(i, 'down')].to(device) / n_batches) * w_down).sum(dim=0)
            ablation_scores['1st_order_std']['mlp'][i] = torch.abs(gw_gate + gw_up + gw_down).cpu().numpy()
            del w_gate, gw_gate, w_up, gw_up, w_down, gw_down

            for m_name in['2nd_order_diag_std', '2nd_order_full_std', '2nd_order_diag_seq', '2nd_order_full_seq']:
                ablation_scores[m_name]['qk'][i] = (ablation_scores[m_name]['qk'][i] / n_batches).cpu().numpy()
                ablation_scores[m_name]['vo'][i] = (ablation_scores[m_name]['vo'][i] / n_batches).cpu().numpy()
                ablation_scores[m_name]['mlp'][i] = (ablation_scores[m_name]['mlp'][i] / n_batches).cpu().numpy()

    return final_scores, ablation_scores


# ──────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ──────────────────────────────────────────────────────────────────────────────
def plot_structural_distributions(scores, out_dir, metric_name):
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans"})
    
    plt.figure(figsize=(10, 6))
    for stype, layer_dict in scores.items():
        all_vals = np.concatenate(list(layer_dict.values()))
        pos = all_vals[all_vals > 0]
        if len(pos) == 0: continue
        logs = np.log10(pos + 1e-30)
        counts, bins = np.histogram(logs, bins=100)
        plt.plot(0.5 * (bins[:-1] + bins[1:]), counts / counts.sum(), label=f"{stype}", linewidth=2)
        
    plt.xlabel(f"log10({metric_name} Score)")
    plt.ylabel("Density")
    plt.title(f"Structural Importance Density ({metric_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "01_structural_density.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for stype, layer_dict in scores.items():
        all_vals = np.concatenate(list(layer_dict.values()))
        pos = np.sort(all_vals[all_vals > 0])
        if len(pos) == 0: continue
        percentiles = np.linspace(0, 100, len(pos))
        plt.plot(percentiles, np.cumsum(pos[::-1])[::-1] / pos.sum() * 100, label=f"{stype}", linewidth=2)

    plt.plot([0, 100], [100, 0], 'k--', alpha=0.5, label="Random Pruning")
    plt.xlabel("% of Structures Pruned")
    plt.ylabel(f"% of Information Retained")
    plt.title(f"Structural Retention Curve ({metric_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "02_structural_retention.png", dpi=150)
    plt.close()

    n_layers = len(scores['mlp_neuron'])
    for m, (stype, layer_dict) in enumerate(scores.items()):
        all_vals = np.concatenate(list(layer_dict.values()))
        pos_vals = all_vals[all_vals > 0]
        if len(pos_vals) == 0: continue
        
        log_min, log_max = np.log10(pos_vals.min()), np.log10(pos_vals.max())
        bins = np.linspace(log_min, log_max, 101)
        heat = np.zeros((n_layers, 100))
        
        for i in range(n_layers):
            pos_layer = layer_dict[i][layer_dict[i] > 0]
            if len(pos_layer) == 0: continue
            hist, _ = np.histogram(np.log10(pos_layer), bins=bins)
            heat[i] = hist / (hist.sum() + 1e-12)
            
        plt.figure(figsize=(12, 8))
        plt.imshow(heat, aspect="auto", origin="lower", extent=[log_min, log_max, -0.5, n_layers - 0.5], cmap="viridis")
        plt.yticks(range(n_layers),[f"Layer {l}" for l in range(n_layers)], fontsize=8)
        plt.xlabel(f"log10({metric_name} Score)")
        plt.title(f"{stype.replace('_', ' ').title()} Importance by Layer")
        plt.colorbar(label="Density")
        plt.savefig(out_dir / f"0{m+3}_{stype}_heatmap.png", dpi=150)
        plt.close()

def plot_ablation_curves(metrics, out_dir):
    metric_labels = {
        '1st_order_std': "1st Order Taylor (Baseline)",
        '2nd_order_diag_std': "2nd Order Diag (Vanilla Fisher)",
        '2nd_order_full_std': "2nd Order Full Block (Exact Hessian)",
        '2nd_order_diag_seq': "Seq-Coupled Diag Fisher",
        '2nd_order_full_seq': "Seq-Coupled Full Block (Ours)"
    }
    for struct in ['qk', 'vo', 'mlp']:
        plt.figure(figsize=(10, 6))
        for m_name, label in metric_labels.items():
            if len(metrics[m_name][struct]) == 0: continue
            all_vals = np.concatenate([metrics[m_name][struct][i] for i in range(len(metrics[m_name][struct]))])
            pos = np.sort(all_vals[all_vals > 0])
            if len(pos) == 0: continue
            percentiles = np.linspace(0, 100, len(pos))
            plt.plot(percentiles, np.cumsum(pos[::-1])[::-1] / pos.sum() * 100, label=label, linewidth=2)
            
        plt.plot([0, 100], [100, 0], 'k:', alpha=0.5, label="Random Pruning")
        plt.xlabel(f"% of {struct.upper()} Pruned")
        plt.ylabel("Information Retained (%)")
        plt.title(f"Ablation Study: Parameter vs Context Coupling ({struct.upper()})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out_dir / f"ablation_{struct}.png", dpi=150)
        plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.output_dir) / args.metric
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device)
    
    clm_dataset = CLMDataset(args.dataset, max_samples=args.n_samples)
    dataloader = build_dataloader(clm_dataset, tokenizer, args.batch_size)
    
    print(f"\nPhase 1: Accumulating {args.metric} Gradients (Ablation: {args.ablation})...")
    final_scores, ablation_scores = accumulate_metrics(model, dataloader, args, device)
    
    print("\nPhase 2: Generating Deep-Dive Plots...")
    plot_structural_distributions(final_scores, out_dir, args.metric)
    
    if args.ablation and args.metric != "functional":
        print("\nPhase 3: Generating Ablation Comparison Plots...")
        plot_ablation_curves(ablation_scores, out_dir)

    print(f"✅ All plots saved to {out_dir}/")

if __name__ == "__main__":
    main()