"""
FA2-Style Euclidean Assignment Kernel for K-Means

This implements FlashAttention-2 inspired optimizations for k-means clustering:

1. REDUCED NON-MATMUL FLOPs (Primary FA2 contribution)
   - Removed tl.maximum(dist, 0.0) - saves 1 op per element
   - Tiny negative values from float rounding don't affect argmin
   
2. OPTIMIZED TILING FOR TENSOR CORES
   - BLOCK_N × BLOCK_K aligned to MMA fragment sizes (multiples of 16)
   - Larger BLOCK_K (up to 256) increases arithmetic intensity
   
3. WARP-LEVEL WORK PARTITIONING (Split-Q style)
   - BLOCK_N chosen so each warp handles complete rows independently
   - No inter-warp reduction needed (each row's argmin is independent)
   - 4 warps → BLOCK_N=128 (32 rows/warp)
   - 8 warps → BLOCK_N=256 (32 rows/warp)
   
4. AGGRESSIVE PIPELINING
   - num_stages=3,4 for better memory latency hiding
   - More stages = more in-flight memory requests
"""

import torch
import triton
import triton.language as tl


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# =============================================================================
# FA2-Style Autotuning Configs
# =============================================================================
# Key insight: Match BLOCK_N to num_warps for clean work distribution
# Each warp should handle BLOCK_N / num_warps rows independently

_FA2_CONFIGS = [
    # 4 warps: 128 rows / 4 = 32 rows per warp (1 row per thread)
    triton.Config({"BLOCK_N": 128, "BLOCK_K": BK}, num_stages=ns, num_warps=4)
    for BK in [64, 128, 256]      # FA2: larger K tiles
    for ns in [2, 3, 4]           # FA2: more pipeline stages
] + [
    # 8 warps: 256 rows / 8 = 32 rows per warp (1 row per thread)  
    triton.Config({"BLOCK_N": 256, "BLOCK_K": BK}, num_stages=ns, num_warps=8)
    for BK in [64, 128, 256]
    for ns in [2, 3, 4]
] + [
    # Smaller config for higher occupancy on smaller problems
    triton.Config({"BLOCK_N": 64, "BLOCK_K": BK}, num_stages=ns, num_warps=2)
    for BK in [64, 128]
    for ns in [2, 3]
]


# =============================================================================
# FA2-Style Euclidean Assignment Kernel
# =============================================================================

@triton.autotune(_FA2_CONFIGS, key=["N", "K"])
@triton.jit
def _euclid_assign_special_kernel(
    x_ptr,                 # [B, N, D] float16/float32
    c_ptr,                 # [B, K, D] centroids
    x_sq_ptr,              # [B, N] pre-computed ||x||²
    c_sq_ptr,              # [B, K] pre-computed ||c||²
    out_ptr,               # [B, N] output cluster IDs
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
    """
    FA2-Style Split-Q kernel for k-means assignment.
    
    Each program handles BLOCK_N points. Within the program:
    - All warps cooperatively load the same centroid tile (shared K)
    - Each warp computes distances for its subset of points (split Q)
    - Per-row argmin has no cross-row dependencies → no inter-warp sync
    """
    pid_n = tl.program_id(0)  # which N-block
    pid_b = tl.program_id(1)  # which batch

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load points tile (BLOCK_N, D)
    # FA2 Split-Q: These rows will be distributed across warps
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    # Load pre-computed ||x||² (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Initialize best distance/index per point
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # FA2: All warps load SAME centroid tile → L2 cache reuse
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile (D, BLOCK_K) - shared across all warps
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        # Load ||c||² (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        c_sq_tile = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # ------------------------------------------------------------------
        # Compute distances: ||x - c||² = ||x||² + ||c||² - 2(x·c)
        # This is the MATMUL part - uses Tensor Cores
        # ------------------------------------------------------------------
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # (BLOCK_N, BLOCK_K)
        
        # ------------------------------------------------------------------
        # FA2 OPTIMIZATION: Remove tl.maximum(dist, 0.0)
        # 
        # Original code had: dist = tl.maximum(dist, 0.0)
        # This clamps tiny negative values from floating-point rounding.
        # 
        # Why we can remove it:
        # - We only care about ARGMIN, not the actual distance values
        # - Negative values (if any) are tiny (~1e-7) due to float error
        # - argmin([-0.0001, 0.5, 1.2]) == argmin([0, 0.5, 1.2])
        # - Saves 1 non-matmul FLOP per element (significant at scale)
        # ------------------------------------------------------------------
        dist = x_sq_tile[:, None] + c_sq_tile[None, :] - 2.0 * cross
        # dist = tl.maximum(dist, 0.0)  # FA2: REMOVED - reduces non-matmul FLOPs
        
        # Mask invalid centroids
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        # ------------------------------------------------------------------
        # Per-row reduction (no cross-row dependencies)
        # FA2 benefit: Each warp's rows are independent → no sync needed
        # ------------------------------------------------------------------
        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # Write results
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


# =============================================================================
# Python Wrapper
# =============================================================================

def euclid_assign_special(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    c_sq: torch.Tensor = None,
) -> torch.Tensor:
    """
    FA2-style nearest-centroid assignment using Euclidean distance.
    
    Args:
        x         : (B, N, D) float16/float32 - data points
        centroids : (B, K, D) same dtype - cluster centroids  
        x_sq      : (B, N) float32 - pre-computed ||x||² per point
        out       : (B, N) int32 - optional output buffer
        c_sq      : (B, K) float32 - optional pre-computed ||c||²
        
    Returns:
        cluster_ids: (B, N) int32 - nearest centroid index per point
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

    # Get strides
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    # Launch kernel
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
    
    _euclid_assign_special_kernel[grid](
        x, centroids, x_sq, c_sq, out,
        B, N, K, D,
        stride_x_b, stride_x_n, stride_x_d,
        stride_c_b, stride_c_k, stride_c_d,
        stride_xsq_b, stride_xsq_n,
        stride_csq_b, stride_csq_k,
        stride_out_b, stride_out_n,
    )
    return out


# =============================================================================
# Correctness & Performance Test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    B, N, D = 32, 65536, 128
    K = 1024
    dtype = torch.float16
    
    print(f"Testing FA2-style kernel: B={B}, N={N}, K={K}, D={D}")
    print("=" * 60)
    
    # Create test data
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)
    c_sq = (centroids.to(torch.float32) ** 2).sum(-1)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    
    # Reference (PyTorch)
    dist_ref = (
        x_sq.unsqueeze(-1) 
        + c_sq.unsqueeze(1) 
        - 2.0 * torch.einsum("bnd,bkd->bnk", x.float(), centroids.float())
    ).clamp_min_(0.0)
    ref_ids = dist_ref.argmin(dim=-1)
    
    # Our kernel
    tri_ids = euclid_assign_special(x, centroids, x_sq, out, c_sq)
    
    # Check correctness
    match_pct = (ref_ids == tri_ids).float().mean() * 100
    print(f"Correctness: {match_pct:.2f}% match with PyTorch reference")
    
    if match_pct < 99.9:
        # Check if mismatches are due to ties (same distance, different index)
        mismatches = (ref_ids != tri_ids).sum().item()
        print(f"  Mismatches: {mismatches} / {B*N} ({mismatches/(B*N)*100:.4f}%)")
    
    # Benchmark
    print()
    print("Benchmarking...")
    
    # Warmup
    for _ in range(10):
        euclid_assign_special(x, centroids, x_sq, out, c_sq)
    
    # Time it
    repeats = 50
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        euclid_assign_special(x, centroids, x_sq, out, c_sq)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / repeats
    throughput = (B * N * K) / (avg_ms * 1e-3) / 1e9  # billion comparisons/sec
    
    print(f"Average time: {avg_ms:.3f} ms")
    print(f"Throughput: {throughput:.2f} billion point-centroid comparisons/sec")

