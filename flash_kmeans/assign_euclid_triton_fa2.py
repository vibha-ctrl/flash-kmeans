import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# =============================================================================
# FA2-style tuning configs
# Key insight: Use BLOCK_N that divides evenly by (num_warps × 32) for clean
# warp-level work distribution. Each warp handles BLOCK_N / num_warps rows.
# =============================================================================

_FA2_TUNE_CONFIGS = [
    # BLOCK_N should be divisible by num_warps for clean split-Q
    # With 4 warps: BLOCK_N=128 → 32 rows per warp
    # With 8 warps: BLOCK_N=256 → 32 rows per warp
    triton.Config({"BLOCK_N": 128, "BLOCK_K": BK}, num_stages=ns, num_warps=4)
    for BK in [64, 128, 256]
    for ns in [2, 3, 4]
] + [
    triton.Config({"BLOCK_N": 256, "BLOCK_K": BK}, num_stages=ns, num_warps=8)
    for BK in [64, 128, 256]
    for ns in [2, 3, 4]
] + [
    # Smaller tiles for higher occupancy
    triton.Config({"BLOCK_N": 64, "BLOCK_K": BK}, num_stages=ns, num_warps=4)
    for BK in [64, 128]
    for ns in [2, 3]
]


# =============================================================================
# FA2-Style Euclidean Assignment Kernel
# =============================================================================

@triton.autotune(_FA2_TUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _euclid_assign_fa2_kernel(
    x_ptr,                 # [B, N, D]
    c_ptr,                 # [B, K, D]
    x_sq_ptr,              # [B, N]
    c_sq_ptr,              # [B, K]
    out_ptr,               # [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FA2-style Split-Q: Each warp handles different points, shares centroids.
    
    FA2 optimizations:
    1. Split-Q partitioning: Points (Q) divided across warps, centroids (K) shared
       - Each warp computes COMPLETE result for its points
       - No inter-warp communication/sync needed for reduction
    2. BLOCK_N chosen to divide evenly by num_warps
       - 4 warps × 32 threads = 128 rows handled independently
       - 8 warps × 32 threads = 256 rows handled independently
    3. All warps load same centroid tile → maximizes L2 cache hits
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # FA2 Split-Q: Load points tile (BLOCK_N rows)
    # Each warp will handle BLOCK_N/num_warps rows independently
    # The tl.dot will compute all (BLOCK_N, BLOCK_K) elements,
    # and the per-row min/argmin has no cross-row dependencies
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # Load x_sq (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Initialize per-point best distance/index
    # Each row is independent → perfect for split-Q
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # FA2: Iterate over centroids
    # All warps load the SAME centroid tile (shared K/V)
    # This maximizes cache reuse across warps
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile (D, BLOCK_K) - SHARED across all warps
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Load c_sq (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Matmul: (BLOCK_N, D) @ (D, BLOCK_K) → (BLOCK_N, BLOCK_K)
        # Each warp computes its rows independently (split-Q behavior)
        cross = tl.dot(x_tile, c_tile).to(tl.float32)

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        # Per-row reduction - NO cross-row communication needed
        # This is the key FA2 benefit: each warp's rows are independent
        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results - each row writes independently
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# =============================================================================
# FA2-Style Cosine Assignment Kernel
# =============================================================================

@triton.autotune(_FA2_TUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _cosine_assign_fa2_kernel(
    x_ptr,                 # [B, N, D] - should be L2 normalized
    c_ptr,                 # [B, K, D] - should be L2 normalized
    out_ptr,               # [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FA2-style cosine similarity with Split-Q partitioning."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Load points tile (split across warps)
    offs_d = tl.arange(0, D)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # For cosine, find maximum similarity
    best_sim = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # Iterate over centroids (shared across warps)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile - shared across warps
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Cosine similarity = dot product for normalized vectors
        sim = tl.dot(x_tile, c_tile).to(tl.float32)
        sim = tl.where(k_mask[None, :], sim, -3.4e38)

        # Per-row max - no cross-row dependencies
        curr_max = tl.max(sim, axis=1)
        curr_idx = tl.argmax(sim, axis=1)

        update = curr_max > best_sim
        best_sim = tl.where(update, curr_max, best_sim)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# =============================================================================
# Python Wrappers
# =============================================================================

def euclid_assign_fa2(x: torch.Tensor, centroids: torch.Tensor, x_sq: torch.Tensor,
                      out: torch.Tensor = None, c_sq: torch.Tensor = None) -> torch.Tensor:
    """FA2-style nearest-centroid assignment (Euclidean distance).
    
    Args:
        x         : (B, N, D) points
        centroids : (B, K, D) centroids
        x_sq      : (B, N) pre-computed ||x||^2
        out       : (B, N) optional output buffer
        c_sq      : (B, K) optional pre-computed ||c||^2
    
    Returns:
        cluster_ids: (B, N) int32
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda
    assert centroids.dtype == x.dtype

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)
    assert x_sq.shape == (B, N)

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _euclid_assign_fa2_kernel[grid](
        x, centroids, x_sq, c_sq, out,
        B, N, K, D,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_xsq_b, stride_xsq_n,
        stride_csq_b, stride_csq_k,
        stride_out_b, stride_out_n,
    )
    return out


def cosine_assign_fa2(x: torch.Tensor, centroids: torch.Tensor,
                      out: torch.Tensor = None) -> torch.Tensor:
    """FA2-style nearest-centroid assignment (cosine similarity).
    
    Args:
        x         : (B, N, D) L2-normalized points
        centroids : (B, K, D) L2-normalized centroids
        out       : (B, N) optional output buffer
    
    Returns:
        cluster_ids: (B, N) int32
    """
    assert x.is_cuda and centroids.is_cuda
    assert centroids.dtype == x.dtype

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _cosine_assign_fa2_kernel[grid](
        x, centroids, out,
        B, N, K, D,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_out_b, stride_out_n,
    )
    return out


# ---------------------------------------------------------------
# Quick correctness & performance check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128
    K = 1000
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    cent = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)

    # Reference
    dist = (
        x_sq.unsqueeze(-1) + (cent.to(torch.float32) ** 2).sum(-1).unsqueeze(1) - 2.0 * torch.einsum("bnd,bkd->bnk", x, cent).to(torch.float32)
    ).clamp_min_(0.0)
    ref_ids = dist.argmin(dim=-1)

    tri_ids = euclid_assign_fa2(x, cent, x_sq, out)

    print("Correct:", torch.equal(ref_ids.cpu(), tri_ids.cpu()))


    dist_cos = torch.einsum("bnd,bkd->bnk", x.to(torch.float32), cent.to(torch.float32))
    ref_ids_cos = dist_cos.argmax(dim=-1)
    tri_ids_cos = cosine_assign_fa2(x, cent, out)

    print("Cosine Correct:", torch.equal(ref_ids_cos.cpu(), tri_ids_cos.cpu()))

    # Simple timing
    repeats = 20
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        euclid_assign_fa2(x, cent, x_sq, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton FA2: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids[10, 69344]=}, {tri_ids[10, 69344]=}, {dist[10, 69344, ref_ids[10, 69344]]=}, {dist[10, 69344, tri_ids[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids, tri_ids.to(ref_ids.dtype))
    except Exception as e:
        print("Assertion failed:", e)

    start.record()
    for _ in range(repeats):
        cosine_assign_fa2(x, cent, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton FA2 Cosine: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids_cos[10, 69344]=}, {tri_ids_cos[10, 69344]=}, {dist_cos[10, 69344, ref_ids_cos[10, 69344]]=}, {dist_cos[10, 69344, tri_ids_cos[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids_cos, tri_ids_cos.to(ref_ids_cos.dtype))
    except Exception as e:
        print("Assertion failed:", e)
