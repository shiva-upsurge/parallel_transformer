import torch
from collections import defaultdict


def print_trainable_parameters(model):
    """
    Prints detailed information about the model's parameters.
    """
    trainable_params = 0
    all_param = 0
    dtype_counts = defaultdict(int)
    param_details = []
    layer_trainability = defaultdict(
        lambda: {"trainable": 0, "non_trainable": 0})
    total_memory = 0  # Approximate memory usage

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            layer_trainability[name.split(
                ".")[0]]["trainable"] += param.numel()
        else:
            layer_trainability[name.split(
                ".")[0]]["non_trainable"] += param.numel()

        # Count dtype occurrences
        dtype_counts[param.dtype] += param.numel()

        # Approximate memory usage (bytes = numel * dtype size)
        total_memory += param.numel() * param.element_size()

        # Collect parameter details
        param_details.append(
            f"{name}: shape={param.size()} trainable={param.requires_grad} dtype={param.dtype}"
        )

    dtype_summary = ", ".join(
        f"{dtype}: {count:,}" for dtype, count in dtype_counts.items()
    )
    frozen_layers = sum(
        1 for layer, counts in layer_trainability.items() if counts["trainable"] == 0)

    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )
    print(f"Parameter dtypes: {dtype_summary}")
    print(f"Approx. memory usage: {total_memory / (1024 ** 3):,.2f} GB")
    print(f"Frozen layers: {frozen_layers}/{len(layer_trainability)}")
    print("\nLayer-wise parameter trainability:")
    for layer, counts in layer_trainability.items():
        trainable = counts["trainable"]
        non_trainable = counts["non_trainable"]
        print(f"{layer}: trainable={trainable:,}, non_trainable={non_trainable:,}")
    print("\nDetailed parameter info:")
    for detail in param_details:
        print(detail)
