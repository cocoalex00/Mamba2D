#include <torch/extension.h>

template <typename scalar_t>
__global__ void wf_fwd(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> hs,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> deltaT,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> deltaL,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> BT,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> BL,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> AT,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> AL,
    const int diag,
    const int diag_len,
    const int i_max
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= diag_len) return;
    
    const int i = i_max - t;
    const int j = diag - i;
    
    const int batch = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const int emb = hs.size(3);
    const int state = hs.size(4);

    if (k >= batch * emb * state) return;

    const int b = k / (emb * state);
    const int k_rem = k % (emb * state);
    const int e = k_rem / state;
    const int n = k_rem % state;

    if ((i < 0) || (i >= h) || (j < 0) || (j >= w)) return;

    // Fused computation: load inputs and compute gates in registers
    scalar_t val_dt = deltaT[b][i][j][e];
    scalar_t val_dl = deltaL[b][i][j][e];
    scalar_t val_x  = x[b][i][j][e];
    scalar_t val_BT = BT[b][i][j][n];
    scalar_t val_BL = BL[b][i][j][n];
    scalar_t val_AT = AT[e][n];
    scalar_t val_AL = AL[e][n];

    // Discretized gates and input terms
    scalar_t val_deltaAT = exp(val_dt * val_AT);
    scalar_t val_deltaAL = exp(val_dl * val_AL);
    scalar_t val_BXT     = val_dt * val_BT * val_x;
    scalar_t val_BXL     = val_dl * val_BL * val_x;

    // --------------------------------------------------------
    // STANDARD WAVEFRONT RECURRENCE
    // --------------------------------------------------------
    if ((i == 0) && (j == 0)) {
        hs[b][i][j][e][n] = val_BXL; // (0,0): L-direction only by convention (matches Python mask)
    } else if (i == 0) {
        hs[b][i][j][e][n] = val_deltaAL * hs[b][i][j-1][e][n] + val_BXL;
    } else if (j == 0) {
        hs[b][i][j][e][n] = val_deltaAT * hs[b][i-1][j][e][n] + val_BXT;
    } else {
        hs[b][i][j][e][n] = 0.5f * ( // Average T and L contributions
            val_deltaAL * hs[b][i][j-1][e][n] + val_BXL +
            val_deltaAT * hs[b][i-1][j][e][n] + val_BXT
        );
    }
}


template <typename scalar_t>
__global__ void wf_bwd(
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> hs,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> deltaT,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> deltaL,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> AT,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> AL,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> dDAT,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> dDAL,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> omega,
    const int diag,
    const int diag_len,
    const int i_max
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= diag_len) return;
    const int i = i_max - t;
    const int j = diag - i;
    
    const int batch = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const int emb = hs.size(3);
    const int state = hs.size(4);

    if (k >= batch * emb * state) return;
    const int b = k / (emb * state);
    const int k_rem = k % (emb * state);
    const int e = k_rem / state;
    const int n = k_rem % state;
    if ((i < 0) || (i >= h) || (j < 0) || (j >= w)) return;

    // Rolling buffer: omega alternates between two slots per diagonal
    int next_slot = (diag + 1) % 2;
    int curr_slot = (diag) % 2;
    scalar_t omega_right = (j < w - 1) ? omega[next_slot][b][i][e][n] : (scalar_t)0.0f;
    scalar_t omega_bottom = (i < h - 1) ? omega[next_slot][b][i+1][e][n] : (scalar_t)0.0f;

    scalar_t grad_out = grad_output[b][i][j][e][n];
    scalar_t current_omega = 0.0f;
    scalar_t current_dBX = 0.0f;

    // Gradient accumulation — 4 cases for boundary neighbor access
    if ((i == h-1) && (j == w-1)) {
        current_dBX = grad_out;
        current_omega = grad_out;
    } else if (i == h-1) {
         // Bottom row: right neighbor only
         scalar_t gate_AL = exp(deltaL[b][i][j+1][e] * AL[e][n]);
         scalar_t d_next = grad_output[b][i][j+1][e][n];
         current_dBX = grad_out + gate_AL * d_next;
         current_omega = grad_out + gate_AL * omega_right;
    } else if (j == w-1) {
         // Right column: bottom neighbor only
         scalar_t gate_AT = exp(deltaT[b][i+1][j][e] * AT[e][n]);
         scalar_t d_next = grad_output[b][i+1][j][e][n];
         current_dBX = grad_out + gate_AT * d_next;
         current_omega = grad_out + gate_AT * omega_bottom;
    } else {
         // Interior: both right and bottom neighbors
         scalar_t gate_AL = exp(deltaL[b][i][j+1][e] * AL[e][n]);
         scalar_t gate_AT = exp(deltaT[b][i+1][j][e] * AT[e][n]);
         scalar_t d_next_L = grad_output[b][i][j+1][e][n];
         scalar_t d_next_T = grad_output[b][i+1][j][e][n];
         
         current_dBX = grad_out + gate_AT * d_next_T + gate_AL * d_next_L;
         
         scalar_t mask_dAT = (j == 0) ? 2.0f : 1.0f;
         scalar_t mask_dAL = (i == 0) ? 2.0f : 1.0f;
         current_omega = grad_out + mask_dAT * gate_AT * omega_bottom + mask_dAL * gate_AL * omega_right;
    }
    
    // 0.5: gradient of forward's interior averaging (hs = 0.5*(T+L))
    current_omega *= 0.5f;
    if (i > 0 && j > 0) current_dBX *= 0.5f;

    grad_output[b][i][j][e][n] = current_dBX;
    omega[curr_slot][b][i][e][n] = current_omega;

    // Gate gradients
    scalar_t mask_dAT_out = (j == 0) ? 2.0f : 1.0f;
    dDAT[b][i][j][e][n] = (i > 0) ? (mask_dAT_out * current_omega * hs[b][i - 1][j][e][n]) : (scalar_t)0.0f;
    
    scalar_t mask_dAL_out = (i == 0) ? 2.0f : 1.0f;
    dDAL[b][i][j][e][n] = (j > 0) ? (mask_dAL_out * current_omega * hs[b][i][j - 1][e][n]) : (scalar_t)0.0f;
}

void wf_fwd_launcher(torch::Tensor& hs,
                     const torch::Tensor& x,
                     const torch::Tensor& deltaT,
                     const torch::Tensor& deltaL,
                     const torch::Tensor& BT,
                     const torch::Tensor& BL,
                     const torch::Tensor& AT,
                     const torch::Tensor& AL
                    )
{
    const int b = hs.size(0);
    const int H = hs.size(1);
    const int W = hs.size(2);
    const int E = hs.size(3);
    const int N = hs.size(4);

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, hs.scalar_type(),
                                   "wf_fwd_launcher", [&] () {
        const int max_threads = 1024;

        // k = i + j ranges from 0 to H + W - 2
        for (int k = 0; k < H + W - 1; ++k) {
            // Compute diag_len and i_max for this diagonal
            const int i_min = std::max(0, k - (W - 1));
            const int i_max = std::min(k, H - 1);
            const int diag_len = std::max(0, i_max - i_min + 1);
            
            if (diag_len <= 0) continue;

            // y-dimension: one thread per (i,j) along diag
            threadsPerBlock.y = std::min(diag_len, 16);
            threadsPerBlock.x = max_threads / threadsPerBlock.y;

            const int total_k = b * E * N;
            blocksPerGrid.x = (total_k + threadsPerBlock.x - 1) / threadsPerBlock.x;
            blocksPerGrid.y = (diag_len + threadsPerBlock.y - 1) / threadsPerBlock.y;

            wf_fwd<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
                hs.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                x.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                deltaT.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                deltaL.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                BT.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                BL.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                AT.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                AL.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                k,
                diag_len,
                i_max
            );
        }
    });
}

void wf_bwd_launcher(const torch::Tensor& hs,
                     const torch::Tensor& x,
                     const torch::Tensor& deltaT,
                     const torch::Tensor& deltaL,
                     const torch::Tensor& AT,
                     const torch::Tensor& AL,
                     torch::Tensor& grad_output,
                     torch::Tensor& dDAT,
                     torch::Tensor& dDAL,
                     torch::Tensor& omega)
{
    const int b = hs.size(0);
    const int H = hs.size(1);
    const int W = hs.size(2);
    const int E = hs.size(3);
    const int N = hs.size(4);

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, hs.scalar_type(), "wf_bwd_launcher", [&] () {
        const int max_threads = 1024;

        // Backward iteration: k goes from H+W-2 down to 0
        for (int k = H + W - 2; k >= 0; --k) {
            // Compute diag_len and i_max for this diagonal
            const int i_min = std::max(0, k - (W - 1));
            const int i_max = std::min(k, H - 1);
            const int diag_len = std::max(0, i_max - i_min + 1);
            
            if (diag_len <= 0) continue;
            
            threadsPerBlock.y = std::min(diag_len, 16);
            threadsPerBlock.x = max_threads / threadsPerBlock.y;

            const int total_k = b * E * N;
            blocksPerGrid.x = (total_k + threadsPerBlock.x - 1) / threadsPerBlock.x;
            blocksPerGrid.y = (diag_len + threadsPerBlock.y - 1) / threadsPerBlock.y;

            wf_bwd<<<blocksPerGrid, threadsPerBlock>>>(
                hs.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                x.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                deltaT.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                deltaL.packed_accessor64<scalar_t,4,torch::RestrictPtrTraits>(),
                AT.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                AL.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                grad_output.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                dDAT.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                dDAL.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                omega.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
                k,
                diag_len,
                i_max
            );
        }
    });
}