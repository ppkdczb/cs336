from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import time
import wandb

from cs336_basics.transformers.utils import (
    transformer_lm,
    cross_entropy_loss,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)

from cs336_basics.bpe.tokenizer import tokenizer

@dataclass
class TrainConfig:
    train_tokens_path: str = "/home/ppkdczb/study/assignment1-basics/data/tinystories_train.bin" # 训练数据路径
    valid_tokens_path: str | None = "/home/ppkdczb/study/assignment1-basics/data/tinystories_valid.bin" # 验证数据路径（可选）
    vocab_size: int = 10000 # 词表大小
    context_length: int = 256 # 上下文长度
    d_model: int = 512 # embedding 维度
    num_layers: int = 4 # transformer block 的层数
    num_heads: int = 16 # multi-head attention 中 head 的数量
    d_ff: int = 1344 # feedforward 网络中隐藏层的维度
    rope_theta: float = 10000.0 # RoPE 中的 θ 参数
    batch_size: int = 32 # 每个训练批次的样本数量
    AdamW_beta: tuple[float, float] = (0.9, 0.999) # AdamW 优化器的 β 参数
    max_iters: int = 20000 # 最大训练迭代次数
    max_lr: float = 3e-4 # 学习率的初始值
    min_lr: float = 3e-5 # 学习率的最小值
    warmup_iters: int = 1000 # 学习率预热的迭代次数
    grad_clip: float = 1.0 # 梯度裁剪的阈值
    weight_decay: float = 0.1 # 权重衰减系数
    log_every: int = 20 # 每隔多少迭代记录一次训练日志
    eval_every: int = 500 # 每隔多少迭代评估一次模型性能
    eval_batches: int = 20 # 评估时使用的批次数量
    save_every: int = 1000 # 每隔多少迭代保存一次模型检查点
    checkpoint_path: str | None = "./checkpoints/checkpoint.pt" #"checkpoint.pt" # 需要读取的模型检查点的保存路径
    save_checkpoint_path: str = "./checkpoints/" # 需要保存的模型检查点的路径
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备
    use_wandb: bool = True
    wandb_project: str = "cs336_transformer_lm"
    wandb_name: str | None = None
    wandb_mode: str = "online" # "online", "offline", or "disabled"

def build_model(config: TrainConfig) -> transformer_lm:
    """
    构建一个 transformer_lm 模型实例。

    参数:
        config (TrainConfig): 包含模型超参数的配置对象。

    返回:
        transformer_lm: 根据配置构建的语言模型实例。
    """
    return transformer_lm(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)


def evaluate(model: transformer_lm, dataset, cfg: TrainConfig) -> float:
    """
    在验证集上评估模型的性能，计算平均交叉熵损失。

    参数:
        model (transformer_lm): 需要评估的语言模型实例。
        dataset: 验证数据集，形状为 (num_tokens,) 的一维张量。
        cfg (TrainConfig): 包含评估配置的对象。
    """
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(dataset, cfg.batch_size, cfg.context_length, cfg.device)
            logits = model(x)
            loss1 = cross_entropy_loss(logits, y)
            loss += loss1
    loss = loss / cfg.eval_batches
    model.train()
    return loss

def train(model: transformer_lm, cfg: TrainConfig):
    train_path = Path(cfg.train_tokens_path)
    valid_path = Path(cfg.valid_tokens_path) if cfg.valid_tokens_path else None
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_path, dtype=np.uint16, mode="r") if valid_path else None
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=cfg.AdamW_beta,
        weight_decay=cfg.weight_decay,
    )
    iteration = 0
    run = None
    if cfg.use_wandb:
        run_name = cfg.wandb_name or f"train_{int(time.time())}"
        run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            mode=cfg.wandb_mode,
            config={
                "train_tokens_path": cfg.train_tokens_path,
                "valid_tokens_path": cfg.valid_tokens_path,
                "vocab_size": cfg.vocab_size,
                "context_length": cfg.context_length,
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "d_ff": cfg.d_ff,
                "rope_theta": cfg.rope_theta,
                "batch_size": cfg.batch_size,
                "adamw_beta": cfg.AdamW_beta,
                "max_iters": cfg.max_iters,
                "max_lr": cfg.max_lr,
                "min_lr": cfg.min_lr,
                "warmup_iters": cfg.warmup_iters,
                "grad_clip": cfg.grad_clip,
                "weight_decay": cfg.weight_decay,
                "eval_batches": cfg.eval_batches,
                "device": cfg.device,
            },
        )
    if cfg.checkpoint_path is not None and Path(cfg.checkpoint_path).exists():
        iteration = load_checkpoint(cfg.checkpoint_path, model, optimizer) + 1
        print(f"Resuming training from iteration {iteration}...")
    model.train()
    start_time = time.time()
    for iter in range(iteration, cfg.max_iters):
        lr = get_lr_cosine_schedule(iter, cfg.max_lr, cfg.min_lr, cfg.warmup_iters, cfg.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr # question, 为什么要单独设置学习率，直接设置 optimizer.lr 不行吗？ answer: 不要改defaults
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()
        elapsed = time.time() - start_time
        tokens_processed = (iter + 1) * cfg.batch_size * cfg.context_length
        if iter % cfg.log_every == 0:
            print(
                f"Iter {iter}: loss = {loss.item():.4f}, lr = {lr:.2e}, "
                f"tokens_processed = {tokens_processed}"
            )
            if run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/wallclock_time": elapsed,
                        "train/tokens_processed": tokens_processed,
                    },
                    step=iter,
                )
        if valid_data is not None and iter % cfg.eval_every == 0:
            val_loss = evaluate(model, valid_data, cfg).item()
            print(f"Iter {iter}: validation loss = {val_loss:.4f}")
            if run is not None:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/wallclock_time": elapsed,
                        "val/tokens_processed": tokens_processed,
                    },
                    step=iter,
                )
        if iter % cfg.save_every == 0:
            #check
            if not Path(cfg.save_checkpoint_path).exists():
                Path(cfg.save_checkpoint_path).mkdir(parents=True, exist_ok=True)
            point_path = Path(cfg.save_checkpoint_path)/f"checkpoint.pt"
            save_checkpoint(model, optimizer, iter, point_path)
    
    # 最后保存一次模型
    if not Path(cfg.save_checkpoint_path).exists():
        Path(cfg.save_checkpoint_path).mkdir(parents=True, exist_ok=True)
    final_point_path = Path(cfg.save_checkpoint_path)/f"checkpoint.pt"
    save_checkpoint(model, optimizer, cfg.max_iters, final_point_path)
    if run is not None:
        wandb.finish()

if __name__ == "__main__":
    config = TrainConfig()
    model = build_model(config)
    train(model, config)
    print("Training completed.")