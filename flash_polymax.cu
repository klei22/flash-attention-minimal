#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                              const int Tc, const int Tr, const int Bc, const int Br, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);

    // Define shared memory for Q,K,V tiles
    extern __shared__ float shared_mem[];
    float* Qi = shared_mem;
    float* Kj = Qi + (Bc * d);
    float* Vj = Kj + (Bc * d);

    for (int j = 0; j < Tc; ++j) {
        // Load K, V tiles into shared memory
        int global_kv_idx = qkv_offset + (j * Bc * d) + tx;
        if(tx < d) { // Ensure we do not read out of bounds
            Kj[tx] = K[global_kv_idx];
            Vj[tx] = V[global_kv_idx];
        }
        __syncthreads();

        for (int i = 0; i < Tr; ++i) {
            // Load Q tile into shared memory
            int global_q_idx = qkv_offset + (i * Br * d) + tx;
            if(tx < d) { // Ensure we do not read out of bounds
                Qi[tx] = Q[global_q_idx];
            }

            // Compute dot product
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += Qi[k] * Kj[k];
            }
            
            // Apply the piecewise function
            float result;
            if (sum < -128) {
                result = 0.0f;
            } else if (sum <= 0) {
                result = (sum / 128.0f) + 1.0f;
            } else { // sum > 0
                result = (sum * sum) + 1.0f;
            }

            // Multiply the piecewise result with V and write back
            for (int k = 0; k < d; ++k) {
                O[global_q_idx + k] += result * Vj[k];
            }
        }
        __syncthreads();
    }
}



torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; // Block size for columns
    const int Br = 32; // Block size for rows

    const int B = Q.size(0); // Batch size
    const int nh = Q.size(1); // Number of heads
    const int N = Q.size(2); // Sequence length
    const int d = Q.size(3); // Feature dimension

    const int Tc = (N + Bc - 1) / Bc; // Tiles along columns
    const int Tr = (N + Br - 1) / Br; // Tiles along rows

    auto options = torch::TensorOptions().device(torch::kCUDA);
    auto O = torch::empty_like(Q, options);

    dim3 grid(B, nh);
    dim3 block(std::min(N, Bc));

    // Shared memory size calculation needs to account for Q, K, and V tiles
    size_t shared_mem_size = 3 * Bc * d * sizeof(float);

    forward_kernel<<<grid, block, shared_mem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, O.data_ptr<float>()
    );

    return O;
}

