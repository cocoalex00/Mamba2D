#include "stdio.h"
#include <torch/extension.h>

template <typename scalar_t>
__global__ void wf_fwd(
        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> hs,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> deltaAT,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> deltaAL,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> BXT,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> BXL,
        const int diag,
        const int offset
        ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y * blockDim.y + threadIdx.y + offset;

    const int i = diag - d;
    const int j = d;

    const int batch = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const int emb = hs.size(3);
    const int state = hs.size(4);

    const int n = k % state;
    const int e = (k / state) % emb;
    const int b = (k / state / emb) % batch;

    // Early exit if indices are OOB
    if ((i < 0) || (i >= h) || (j < 0) || (j >= w) || (k >= batch * emb * state)) {
        return;
    }

    if (i+j == diag) {
        if ((i == 0) && (j == 0)) {
            hs[b][i][j][e][n] = BXL[b][i][j][e][n];
        } else if (i == 0) {
            // Top row
            hs[b][i][j][e][n] = deltaAL[b][i][j][e][n] * hs[b][i][j-1][e][n] + BXL[b][i][j][e][n];
        } else if (j == 0) {
            // Left col
            hs[b][i][j][e][n] = deltaAT[b][i][j][e][n] * hs[b][i-1][j][e][n] + BXT[b][i][j][e][n];
        } else {
            // Default case
            hs[b][i][j][e][n] = 0.5 * (deltaAL[b][i][j][e][n] * hs[b][i][j-1][e][n] + BXL[b][i][j][e][n] + deltaAT[b][i][j][e][n] * hs[b][i-1][j][e][n] + BXT[b][i][j][e][n]);
        }
    }
}

template <typename scalar_t>
__global__ void wf_bwd(
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> hs,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> deltaAT,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> deltaAL,
        const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output,
        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dDAT,
        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dDAL,
        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> omega,
        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dBX,
        const int diag,
        const int offset
        ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y * blockDim.y + threadIdx.y + offset;

    const int i = diag - d;
    const int j = d;

    const int batch = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const int emb = hs.size(3);
    const int state = hs.size(4);

    const int n = k % state;
    const int e = (k / state) % emb;
    const int b = (k / state / emb) % batch;

    // Early exit if indices are OOB
    if ((i < 0) || (i >= h) || (j < 0) || (j >= w) || (k >= batch * emb * state)) {
        return;
    }

    if (i+j == diag) {
        scalar_t grad_out = grad_output[b][i][j][e][n];

        if ((i == h-1) && (j == w-1)) {
            dBX[b][i][j][e][n] = grad_out;
            omega[b][i][j][e][n] *= grad_out;
        } else if (i == h-1) {
            // Bottom row
            dBX[b][i][j][e][n] = grad_out + deltaAL[b][i][j+1][e][n] * dBX[b][i][j+1][e][n];
            omega[b][i][j][e][n] *= grad_out + (deltaAL[b][i][j+1][e][n] * omega[b][i][j+1][e][n]);
        } else if (j == w-1) {
            // Right col
            dBX[b][i][j][e][n] = grad_out + deltaAT[b][i+1][j][e][n] * dBX[b][i+1][j][e][n];
            omega[b][i][j][e][n] *= grad_out + (deltaAT[b][i+1][j][e][n] * omega[b][i+1][j][e][n]);
        } else {
            // Default case
            dBX[b][i][j][e][n] = grad_out + (deltaAT[b][i+1][j][e][n] * dBX[b][i+1][j][e][n]) + (deltaAL[b][i][j+1][e][n] * dBX[b][i][j+1][e][n]);
            omega[b][i][j][e][n] *= grad_out + (dDAT[b][i][j][e][n] * deltaAT[b][i+1][j][e][n] * omega[b][i+1][j][e][n] + dDAL[b][i][j][e][n] * deltaAL[b][i][j+1][e][n] * omega[b][i][j+1][e][n]);
        }

        // Handling out-of-bounds cases
        dDAT[b][i][j][e][n] = (i == 0) ? (scalar_t)0 : dDAT[b][i][j][e][n] * omega[b][i][j][e][n] * hs[b][i - 1][j][e][n];
        dDAL[b][i][j][e][n] = (j == 0) ? (scalar_t)0 : dDAL[b][i][j][e][n] * omega[b][i][j][e][n] * hs[b][i][j - 1][e][n];

        // dBX top/left row/col fudge
        if ((i > 0) and (j > 0)){
            dBX[b][i][j][e][n] /= 2;
        }
    }
}

void wf_fwd_launcher(torch::Tensor& hs,
                     const torch::Tensor& deltaAT,
                     const torch::Tensor& deltaAL,
                     const torch::Tensor& BXT,
                     const torch::Tensor& BXL) 
{
    
    // Extract dim sizes
    const uint b = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const uint e = hs.size(3);
    const uint n = hs.size(4);

    // Calculate set threads per block and blocks per grid 
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;
    int diff = std::abs(h-w);
    int diag_len = 0;
    int max_diag_len = std::min(h,w);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, hs.scalar_type(), "wf_fwd_launcher", [&] () {
        for (int diag = 0; diag < (h+w-1); diag++) {

            // Dynamically scale threads on diag
            if (diag < max_diag_len) {
                diag_len++;
            } else {
                if (diff > 0) {
                    diff--;
                } else {
                    diag_len--;
                }
            }
            
            threadsPerBlock.y = std::min(diag_len, 16);
            threadsPerBlock.x = 1024/threadsPerBlock.y;

            // https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
            blocksPerGrid.x = ((b*e*n)  / threadsPerBlock.x) + ((b*e*n) % threadsPerBlock.x != 0);
            blocksPerGrid.y = (diag_len / threadsPerBlock.y) + (diag_len % threadsPerBlock.y != 0);

            wf_fwd<<<blocksPerGrid, threadsPerBlock>>>(
                hs.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                deltaAT.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                deltaAL.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                BXT.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                BXL.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                diag,
                // Compensate with offset as diags are now zero indexed
                std::max(0,(diag-max_diag_len+1)));
        }
    });
}

void wf_bwd_launcher(const torch::Tensor& hs,
                     const torch::Tensor& deltaAT,
                     const torch::Tensor& deltaAL,
                     const torch::Tensor& grad_output,
                     torch::Tensor& dDAT,
                     torch::Tensor& dDAL,
                     torch::Tensor& omega,
                     torch::Tensor& dBX)
{
    
    // Extract dim sizes
    const uint b = hs.size(0);
    const int h = hs.size(1);
    const int w = hs.size(2);
    const uint e = hs.size(3);
    const uint n = hs.size(4);

    // Calculate set threads per block and blocks per grid 
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;
    int diff = std::abs(h-w);
    int diag_len = 0;
    int max_diag_len = std::min(h,w);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, hs.scalar_type(), "wf_bwd_launcher", [&] () {
        for (int diag = (h+w-1)-1; diag >= 0; diag--) {

            // Dynamically scale threads on diag
            if (diag_len < max_diag_len) {
                diag_len++;
            } else {
                if (diff > 0) {
                    diff--;
                } else {
                    diag_len--;
                }
            }
            
            threadsPerBlock.y = std::min(diag_len, 16);
            threadsPerBlock.x = 1024/threadsPerBlock.y;

            // https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
            blocksPerGrid.x = ((b*e*n)  / threadsPerBlock.x) + ((b*e*n) % threadsPerBlock.x != 0);
            blocksPerGrid.y = (diag_len / threadsPerBlock.y) + (diag_len % threadsPerBlock.y != 0);

            wf_bwd<<<blocksPerGrid, threadsPerBlock>>>(
                hs.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                deltaAT.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                deltaAL.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                dDAT.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                dDAL.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                omega.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                dBX.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                diag,
                // Compensate with offset as diags are now zero indexed
                std::max(0,(diag-max_diag_len+1))
                );
        }
    });
}
