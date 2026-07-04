"""
使用 MicroDiT checkpoint 生成图像的推理脚本。

用法示例:
    # 使用4通道模型生成512x512图像
    python generate.py \
        --ckpt_path ./trained_models/MicroDiTXL_mask_0_res_512_finetune/latest-rank0.pt \
        --in_channels 4 \
        --latent_res 64 \
        --pos_interp_scale 2.0 \
        --prompts "A photo of a cat wearing a hat" "A beautiful sunset over mountains" \
        --num_images 4 \
        --output_dir ./generated_images

    # 使用256x256模型
    python generate.py \
        --ckpt_path ./trained_models/MicroDiTXL_mask_0_res_256_finetune/latest-rank0.pt \
        --in_channels 4 \
        --latent_res 32 \
        --pos_interp_scale 1.0 \
        --prompts "A cute robot" \
        --num_images 8

    # 从文件读取prompts
    python generate.py \
        --ckpt_path ./ckpts/dit_4_channel_37M_real_and_synthetic_data.pt \
        --in_channels 4 \
        --latent_res 64 \
        --pos_interp_scale 2.0 \
        --prompt_file prompts.txt \
        --num_images 4
"""

import argparse
import math
import os
from typing import List, Optional

import torch
from torchvision.utils import save_image, make_grid

from micro_diffusion.models.model import create_latent_diffusion


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 MicroDiT checkpoint 生成图像"
    )
    
    # 模型参数
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="checkpoint 文件路径 (.pt 文件)"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        choices=[4, 16],
        help="VAE 的通道数: SDXL-VAE=4, Ostris-VAE=16 (默认: 4)"
    )
    parser.add_argument(
        "--latent_res",
        type=int,
        default=64,
        help="latent 空间分辨率: 256x256图像用32, 512x512用64 (默认: 64)"
    )
    parser.add_argument(
        "--pos_interp_scale",
        type=float,
        default=2.0,
        help="位置编码插值比例: 256分辨率用1.0, 512用2.0 (默认: 2.0)"
    )
    parser.add_argument(
        "--dit_arch",
        type=str,
        default="MicroDiT_XL_2",
        help="DiT 模型架构名称 (默认: MicroDiT_XL_2)"
    )
    parser.add_argument(
        "--vae_name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="VAE 模型名称 (默认: stabilityai/stable-diffusion-xl-base-1.0)"
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
        help="文本编码器名称 (默认: openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型数据类型 (默认: float16, 如果硬件支持 bfloat16 可选 bfloat16)"
    )
    
    # 生成参数
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="生成图像的文本提示列表"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="包含 prompts 的文本文件路径 (每行一个prompt)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="每个 prompt 生成的图像数量 (默认: 4)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="采样步数，越大质量越好但速度越慢 (默认: 30)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance 强度，越大越遵循prompt (默认: 5.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子，用于可复现生成 (默认: 2024)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="输出图像保存目录 (默认: ./generated_images)"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        default=True,
        help="是否保存单张图像 (默认: True)"
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        default=True,
        help="是否保存网格图 (默认: True)"
    )
    parser.add_argument(
        "--grid_nrow",
        type=int,
        default=4,
        help="网格图每行图像数量 (默认: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (默认: cuda)"
    )
    
    return parser.parse_args()


def load_prompts(args) -> List[str]:
    """从命令行参数或文件加载 prompts。"""
    prompts = []
    
    if args.prompts:
        prompts.extend(args.prompts)
    
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt 文件不存在: {args.prompt_file}")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # 跳过空行和注释
                    prompts.append(line)
    
    if not prompts:
        raise ValueError("必须提供 --prompts 或 --prompt_file 参数")
    
    return prompts


def main():
    """主函数：加载模型并生成图像。"""
    args = parse_args()
    
    # ============================================================
    # 1. 加载 prompts
    # ============================================================
    prompts = load_prompts(args)
    print(f"📝 加载了 {len(prompts)} 个 prompts:")
    for i, p in enumerate(prompts):
        print(f"   [{i+1}] {p}")
    
    # ============================================================
    # 2. 创建模型
    # ============================================================
    print("\n🔧 创建模型...")
    print(f"   - 架构: {args.dit_arch}")
    print(f"   - VAE: {args.vae_name}")
    print(f"   - 通道数: {args.in_channels}")
    print(f"   - Latent 分辨率: {args.latent_res}x{args.latent_res}")
    print(f"   - 位置编码缩放: {args.pos_interp_scale}")
    print(f"   - 数据类型: {args.dtype}")
    
    model = create_latent_diffusion(
        vae_name=args.vae_name,
        text_encoder_name=args.text_encoder_name,
        dit_arch=args.dit_arch,
        latent_res=args.latent_res,
        in_channels=args.in_channels,
        pos_interp_scale=args.pos_interp_scale,
        dtype=args.dtype,
        precomputed_latents=True,
    ).to(args.device)
    
    # ============================================================
    # 3. 加载 checkpoint（支持多种格式，自动检测分辨率）
    # ============================================================
    print(f"\n📦 加载 checkpoint: {args.ckpt_path}")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {args.ckpt_path}")
    
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    
    # ---- 探测 checkpoint 内部结构 ----
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Composer 格式: {"state_dict": {...}, "optimizers": ..., ...}
        raw_state = ckpt["state_dict"]
        ckpt_format = "composer"
    elif isinstance(ckpt, dict) and any(k.startswith("blocks.") or k.startswith("x_embedder.") for k in ckpt.keys()):
        # 纯 DiT 权重（HuggingFace 预训练格式），key 如 "blocks.0.attn..."
        raw_state = ckpt
        ckpt_format = "dit_only"
    elif isinstance(ckpt, dict) and any(k.startswith("dit.") for k in ckpt.keys()):
        # 完整模型权重但不在 state_dict 嵌套中，key 如 "dit.blocks.0..."
        raw_state = ckpt
        ckpt_format = "full_model"
    else:
        raw_state = ckpt
        ckpt_format = "unknown"
    print(f"   格式: {ckpt_format}")

    # ---- 通过 pos_embed 自动检测 checkpoint 对应的 latent_res ----
    def _detect_latent_res(state):
        """通过 pos_embed 形状推断 latent_res。pos_embed shape = [1, num_patches, dim]。"""
        for key in ["pos_embed", "dit.pos_embed"]:
            if key in state:
                num_patches = state[key].shape[1]
                return int(math.isqrt(num_patches) * 2)  # patch_size=2
        return None

    detected_res = _detect_latent_res(raw_state)
    if detected_res and detected_res != args.latent_res:
        print(f"\n⚠️  检测到 checkpoint latent_res={detected_res} (对应 {detected_res*8}×{detected_res*8})")
        print(f"   当前参数: latent_res={args.latent_res} (对应 {args.latent_res*8}×{args.latent_res*8})")
        print(f"   自动修正: latent_res={detected_res}, pos_interp_scale={detected_res/32:.1f}")
        args.latent_res = detected_res
        args.pos_interp_scale = detected_res / 32.0
        # 用正确参数重建模型
        model = create_latent_diffusion(
            vae_name=args.vae_name,
            text_encoder_name=args.text_encoder_name,
            dit_arch=args.dit_arch,
            latent_res=args.latent_res,
            in_channels=args.in_channels,
            pos_interp_scale=args.pos_interp_scale,
            dtype=args.dtype,
            precomputed_latents=True,
        ).to(args.device)

    # ---- 加载权重，根据格式选择不同的加载策略 ----
    if ckpt_format == "dit_only":
        # 纯 DiT 权重，key 无前缀，直接加载到 model.dit
        print("   加载纯 DiT 权重 → model.dit.load_state_dict()")
        model.dit.load_state_dict(raw_state, strict=True)
    elif ckpt_format in ("composer", "full_model"):
        # 完整模型权重，key 带 "dit." 前缀，用 model.load_state_dict
        # strict=False 因为 VAE/text_encoder 由 create_latent_diffusion 预加载
        print("   加载完整模型权重 → model.load_state_dict(strict=False)")
        missing, unexpected = model.load_state_dict(raw_state, strict=False)
        # 只报告 DiT 相关的 missing/unexpected
        dit_missing = [k for k in missing if k.startswith("dit.")]
        dit_unexpected = [k for k in unexpected if k.startswith("dit.")]
        if dit_missing:
            print(f"   ⚠️  DiT 缺失键: {len(dit_missing)} 个 (前5个: {dit_missing[:5]})")
        if dit_unexpected:
            print(f"   ⚠️  DiT 多余键: {len(dit_unexpected)} 个 (前5个: {dit_unexpected[:5]})")
    else:
        # 尝试两种方式加载
        print("   未知格式，尝试 model.dit.load_state_dict()")
        try:
            model.dit.load_state_dict(raw_state, strict=True)
        except RuntimeError:
            print("   失败，尝试 model.load_state_dict(strict=False)")
            model.load_state_dict(raw_state, strict=False)
    
    model.eval()
    print("   ✅ 模型加载完成")
    
    # ============================================================
    # 4. 生成图像
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_images = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n🎨 生成 prompt [{prompt_idx + 1}/{len(prompts)}]: {prompt}")
        print(f"   生成 {args.num_images} 张图像, 步数={args.num_inference_steps}, CFG={args.guidance_scale}")
        
        with torch.no_grad():
            # 注意：不使用 autocast！EDM 采样器内部已精确管理精度：
            # - 采样循环使用 float64（避免数值不稳定）
            # - 模型前向使用 float32（通过 .to(torch.float32) 显式转换）
            # 外部 autocast 会覆盖这些精度控制，导致 NaN → 全黑图像
            images = model.generate(
                prompt=[prompt] * args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
        
        # 诊断：打印生成图像的统计信息
        print(f"   像素范围: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}")
        
        all_images.append(images)
        
        # 保存单张图像
        if args.save_individual:
            for img_idx, img in enumerate(images):
                # 生成安全的文件名
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:50].strip()
                safe_prompt = safe_prompt.replace(" ", "_")
                filename = f"{safe_prompt}_seed{args.seed}_{img_idx:02d}.png"
                filepath = os.path.join(args.output_dir, filename)
                save_image(img, filepath)
                print(f"   💾 保存: {filepath}")
    
    # ============================================================
    # 5. 保存网格图
    # ============================================================
    if args.save_grid and all_images:
        # 将所有图像合并到一个列表
        all_imgs_flat = []
        for imgs in all_images:
            all_imgs_flat.extend([img.cpu() for img in imgs])
        
        # 创建网格图
        grid = make_grid(all_imgs_flat, nrow=args.grid_nrow, padding=2, normalize=True)
        grid_path = os.path.join(args.output_dir, "grid_all.png")
        save_image(grid, grid_path)
        print(f"\n🖼️  网格图已保存: {grid_path}")
    
    print(f"\n✅ 生成完成! 所有图像保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
