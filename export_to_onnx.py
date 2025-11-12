import os
import argparse
import types
import torch
import torch.nn.functional as F

def _ensure_stub_modules():
    try:
        import xformers  # noqa: F401
    except Exception:
        xformers = types.ModuleType("xformers")
        ops = types.ModuleType("ops")
        fmha = types.ModuleType("fmha")
        class _DummyBias:  # 仅占位
            pass
        fmha.attn_bias = _DummyBias
        xformers.ops = ops
        xformers.ops.fmha = fmha
        import sys
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = ops
        sys.modules["xformers.ops.fmha"] = fmha


def _activate_export_safe_patch(transformer_module, model):
    # 2.1 禁用 gradient checkpoint（transformer 模块里通常以全局名称引用）
    def _no_checkpoint(func, *args, **kwargs):
        return func(*args)
    transformer_module.checkpoint = _no_checkpoint

    # 2.2 替换自定义 _fmha：把注意力实现成 MatMul+Softmax 路径，避免不受支持的算子
    def _safe_fmha(q, k, v, p, bias, training: bool):
        # 期望输入 [B, L, H, D]
        B, L, H, D = q.shape
        q_ = q.permute(0, 2, 1, 3).reshape(B * H, L, D)
        k_ = k.permute(0, 2, 1, 3).reshape(B * H, L, D)
        v_ = v.permute(0, 2, 1, 3).reshape(B * H, L, D)
        scale = D ** -0.5
        scores = torch.matmul(q_, k_.transpose(1, 2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_)
        if training and p and p > 0:
            out = F.dropout(out, p=p, training=True)
        out = out.reshape(B, H, L, D).permute(0, 2, 1, 3)  # -> [B, L, H, D]
        return out
    transformer_module._fmha = _safe_fmha

    # 2.3 将 RoPE 缓存从 half 升到 float32（若存在），避免 CPU 导出 dtype 限制
    for name, module in model.named_modules():
        if hasattr(module, "rotary_emb"):
            rope = getattr(module, "rotary_emb")
            for attr in ("cos_cached", "sin_cached"):
                if hasattr(rope, attr):
                    buf = getattr(rope, attr)
                    if isinstance(buf, torch.Tensor):
                        try:
                            setattr(rope, attr, buf.float())
                        except Exception:
                            pass


def _strip_state_dict_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and (k.startswith("_orig_mod.") or k.startswith("module.")):
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            if k.startswith("module."):
                k = k[len("module."):]
        new_state[k] = v
    return new_state


def main():
    _ensure_stub_modules()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help=".pth 路径（支持包含 model_state_dict 的字典或纯 state_dict）")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="导出 ONNX 路径")
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--output_dim", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--dynamic_seq", action="store_true", help="将序列长度轴也设为动态维")
    parser.add_argument("--strict_load", type=int, default=1, help="1=严格加载；0=非严格（跳过不匹配项）")
    args = parser.parse_args()

    # 延后导入，确保 stub 已生效
    import transformer_GQA_Teacher as T
    _orig_gqa_init = T.GroupedQueryAttention.__init__

    def _gqa_init_monkey(self, d_model, num_heads, num_groups, dropout=T.DROPOUT, max_position_embeddings=None):
        if max_position_embeddings is None:
            max_position_embeddings = args.max_len  # 动态使用导出时指定的长度
        return _orig_gqa_init(self, d_model, num_heads, num_groups, dropout, max_position_embeddings)

    T.GroupedQueryAttention.__init__ = _gqa_init_monkey
    T.MAX_LENGTH = args.max_len

    # 构建模型（需与训练时超参一致）
    model = T.TransformerModel(
        output_dim=args.output_dim,
        max_length=args.max_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_groups=args.num_groups,
    )

    # 加载 checkpoint/state_dict，并剥前缀
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        # 常见两种键名
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # 可能直接就是 state dict
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)} if all(isinstance(k, str) for k in ckpt.keys()) else ckpt
    else:
        state = ckpt

    state = _strip_state_dict_prefixes(state)
    remap_cnt = sum(1 for k in state.keys() if k.startswith("_orig_mod.") or k.startswith("module."))
    if remap_cnt:
        print(f"[INFO] state_dict keys remapped: {remap_cnt}")

    try:
        model.load_state_dict(state, strict=bool(args.strict_load))
    except Exception as e:
        print("[ERROR] load_state_dict failed with strict=", bool(args.strict_load))
        # 给出一些诊断信息
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state.keys())
        missing = sorted(list(model_keys - ckpt_keys))[:20]
        unexpected = sorted(list(ckpt_keys - model_keys))[:20]
        print("  missing (first 20):", missing)
        print("  unexpected (first 20):", unexpected)
        raise e

    model.eval()

    # 启用导出安全补丁
    _activate_export_safe_patch(T, model)

    # 自动推断输入特征维度（线性投影层 in_features）
    try:
        in_dim = model.input_projection.in_features  # 你的教师模型里通常存在这一层
    except Exception:
        in_dim = 4  # 兜底
    print(f"[INFO] inferred input_dim = {in_dim}")

    # 准备 dummy 输入（float32 更通用）
    dummy = torch.randn(1, args.max_len, in_dim, dtype=torch.float32)

    # Dry-run，确认前向可通
    with torch.no_grad():
        out = model(dummy)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            emissions, mask = out
        else:
            raise RuntimeError("model.forward 预期返回 (emissions, mask)")
    print(f"[OK] Dry run passed. emissions={tuple(emissions.shape)}, mask={tuple(mask.shape)}")

    # 动态轴设置
    dynamic_axes = {
        "input": {0: "batch_size"},
        "emissions": {0: "batch_size"},
        "mask": {0: "batch_size"},
    }
    if args.dynamic_seq:
        dynamic_axes["input"][1] = "seq_len"
        dynamic_axes["emissions"][1] = "seq_len"
        dynamic_axes["mask"][1] = "seq_len"

    # 导出
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["emissions", "mask"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[DONE] ONNX saved to: {os.path.abspath(args.onnx)}")


if __name__ == "__main__":
    main()


#python export_to_onnx.py --ckpt best_transformer_model.pth --onnx teacher_gqa.onnx --d_model 64 --num_heads 8 --num_layers 4 --num_groups 2 --output_dim 16 --max_len 2000 --opset 20