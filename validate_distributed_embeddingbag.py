# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Launch this script on a maching with 2 GPUs by running CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node 2 validate_distributed_embeddingbag.py
"""

import torch
import os
from functools import lru_cache, partial, reduce
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from lingua.product_key.colwise_embedding_bag import ColwiseEmbeddingBag, xFormerEmbeddingBag

from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
    Partial,
)


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()

def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0

@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1

def allclose(a, b):
    return ((a - b).abs() < 1e-3).all()


if torch.cuda.device_count() > 1:
    torch.cuda.set_device(get_local_rank())
torch.distributed.init_process_group(init_method="env://", backend="nccl")

def get_embeddingbag():
    torch.manual_seed(42)
    B = 256
    K = 1024 #1024**2
    bag_size = 32
    dim =  128 #4096
    eb = xFormerEmbeddingBag(K, dim).to("cuda").to(torch.float32)
    indices = torch.randint(0, K, [B, bag_size], device='cuda')
    per_sample_weights = torch.randn(indices.shape, requires_grad=True, dtype=torch.float32, device='cuda')
    gradient = torch.randn([B, dim], dtype=torch.float32, device='cuda')
    return eb, indices, per_sample_weights, gradient




# No parallelization
def no_mp():
    eb, indices, per_sample_weights, gradient = get_embeddingbag()
    out = eb(indices, per_sample_weights)
    out.backward(gradient)

    out = out.detach()
    # We need to average the gradients on the parameters instead of summing them
    out_grad = eb.weight.grad.detach() / get_world_size()
    per_sample_weights_grad = per_sample_weights.grad.detach()
    return out, out_grad, per_sample_weights_grad




# parallel mp 1
def mp(mp_size):
    eb, indices, per_sample_weights, gradient = get_embeddingbag()
    n_gpu = get_world_size()

    memory_mesh = init_device_mesh("cuda", mesh_shape=(n_gpu // mp_size, mp_size), mesh_dim_names=["dp_replicate", "mp_size"])

    parallelize_module(
        eb,
        memory_mesh["mp_size"],
        ColwiseEmbeddingBag(),
    )

    fsdp_config = dict(
        mp_policy=(
            MixedPrecisionPolicy(
                # param_dtype=torch.bfloat16,
                # reduce_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        ),
        mesh=memory_mesh['dp_replicate'],
    )

    eb = fully_shard(
        eb, **fsdp_config, reshard_after_forward=False
    )

    # Split accross the first dimension
    dim_0 = indices.shape[0]
    dim_0_start =( dim_0 // n_gpu) * get_local_rank()
    dim_0_end= ( dim_0 // n_gpu) * (get_local_rank() + 1)

    split_indices = indices[dim_0_start:dim_0_end]
    split_per_sample_weights = per_sample_weights[dim_0_start:dim_0_end].clone().detach().requires_grad_(True)
    split_gradient = gradient[dim_0_start:dim_0_end]

    # Run the embedding bag with the plit indices and scores
    out = eb(split_indices, split_per_sample_weights)
    out.backward(split_gradient)

    out = out.detach()
    out_grad = eb.weight.grad
    per_sample_grad = split_per_sample_weights.grad.detach()

    # Reconstruct the  gradients
    per_sample_grad =  DTensor.from_local(
        per_sample_grad, memory_mesh, (Shard(0),Shard(0),), run_check=False
    )
    per_sample_grad = per_sample_grad.full_tensor()
    out_grad = out_grad.full_tensor()
    # Reconstruct the outputs
    out = DTensor.from_local(
        out, memory_mesh, (Shard(0),Shard(0),), run_check=False
    ).full_tensor()

    return out, out_grad, per_sample_grad


no_mp_out, no_mp_out_grad, no_mp_per_sample_weights_grad = no_mp()
mp1_out, mp1_out_grad, mp1_out_per_sample_weights_grad = mp(1)
mp2_out, mp2_out_grad, mp2_out_per_sample_weights_grad = mp(2)



assert allclose(no_mp_out, mp1_out)
assert allclose(no_mp_out_grad, mp1_out_grad)
assert allclose(no_mp_per_sample_weights_grad, mp1_out_per_sample_weights_grad)

assert allclose(no_mp_out, mp2_out)
assert allclose(no_mp_out_grad, mp2_out_grad)
assert allclose(no_mp_per_sample_weights_grad, mp2_out_per_sample_weights_grad) 

torch.distributed.destroy_process_group()
print("OK")
