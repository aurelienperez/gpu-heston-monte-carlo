/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define POW_TRAJ 18
#define N_TRAJ (1 << POW_TRAJ)  // number of trajectories per triplet
#define MAX_TRIPLETS 1000 // upper bound on number of valid (kappa, theta, sigma) triplets

struct HestonParam{
    float kappa;
    float theta;
    float sigma;
};

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


__device__ float rgamma_sup1(curandState *state, float alpha) {
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    float z, u, x, v;
    while (true) {
        z = curand_normal(state);
        u = curand_uniform(state);
        x = 1.0f + c * z;
        v = x * x * x;
        if (z > -1.0f / c && logf(u) < (0.5f * z * z + d - d * v + d * logf(v)))
            return d * v;
    }
}

__device__ float rgamma_inf1(curandState *state, float alpha) {
    float u = curand_uniform(state);
    return rgamma_sup1(state, alpha + 1.0f) * powf(u, 1.0f / alpha);
}

// Gamma distribution sampling (Marsaglia-Tsang)
__device__ float rgamma(curandState *state, float alpha) {
    if (alpha < 1.0f) {
        return rgamma_inf1(state, alpha);
    }
    else{
        return rgamma_sup1(state, alpha);
    }
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(0, idx, 0, &state[idx]);
}

// Exact scheme kernel (Broadie-Kaya)
__global__ void MC_Heston_Exact_kernel(float S_0, float v_0, float r, float rho, float sqrt_dt,
     float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (idx >= total_threads) return;
    int triplet_idx = idx >> POW_TRAJ; // idx / 2^POW_TRAJ  
    // idx          0 1 2 3 4 ... 2^18 - 1  2^18 2^18+1 ...
    // triplet_idx  0 0 0 0 0 ...     0       1     1     ...   
    if (triplet_idx >= n_triplets) return;

    __shared__ HestonParam p_shared;
    if (threadIdx.x == 0) p_shared = d_params[triplet_idx];
    __syncthreads();

    HestonParam p = p_shared;
    float kappa = p.kappa, theta = p.theta, sigma = p.sigma;
    float sigma2 = sigma * sigma;
    float dt = sqrt_dt * sqrt_dt;
    float d = 2.0f * kappa * theta / sigma2;
    float v = v_0, v_next, vI = 0.0f, S = S_0;
    curandState localState = state[idx];

    for (int i = 0; i < N; i++) {
        float lambda = (2.0f * kappa * expf(-kappa * dt) * v) / (sigma2 * (1.0f - expf(-kappa * dt)));
        int N_pois = curand_poisson(&localState, lambda);
        float gamma = rgamma(&localState, d + N_pois);
        v_next = (sigma2 * (1.0f - expf(-kappa * dt)) / (2.0f * kappa)) * gamma;
        vI += 0.5f * dt * (v_next + v);
        v = v_next;
    }
    float m = -0.5f * vI + (rho / sigma) * (v - v_0 - kappa * theta + kappa * vI);
    float Sigma = sqrtf((1.0f - rho * rho) * vI);
    float2 G = curand_normal2(&localState);
    S = S_0 * expf(m + Sigma * G.x);

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    // reduction
    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	} 

}

// Almost Exact Scheme kernel (Haastrecht-Pelsser)
__global__ void MC_Heston_Almost_Exact_kernel(float S_0, float v_0, float r, float rho, float sqrt_dt,
     float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (idx >= total_threads) return;
    int triplet_idx = idx >> POW_TRAJ;
    if (triplet_idx >= n_triplets) return;

    __shared__ HestonParam p_shared;
    if (threadIdx.x == 0) p_shared = d_params[triplet_idx];
    __syncthreads();

    HestonParam p = p_shared;
    float kappa = p.kappa, theta = p.theta, sigma = p.sigma;
    float sigma2 = sigma * sigma;
    float dt = sqrt_dt * sqrt_dt;
    float d = 2.0f * kappa * theta / sigma2;
    float S = logf(S_0);
    float v = v_0;
    float v_next;
    float rhosigma = rho / sigma;
    float k0 = (-rhosigma * kappa * theta) * dt;
    float k1 = (rhosigma * kappa  - 0.5f) * dt - rhosigma;
    float k2 = rhosigma;
    curandState localState = state[idx];

    for (int i = 0; i < N; i++){
        float2 G = curand_normal2(&localState);
        float lambda = (2.0f * kappa * expf(-kappa * dt) * v) / (sigma2 * (1.0f - expf(-kappa * dt)));
        int N_pois = curand_poisson(&localState, lambda);
        float gamma = rgamma(&localState, d + N_pois);
        v_next = (sigma2 * (1.0f - expf(-kappa * dt)) / (2.0f * kappa)) * gamma;
        S += k0 + k1 * v + k2 * v_next + sqrtf((1.0f - rho*rho) * v) * sqrt_dt * (rho * G.x + sqrtf(1.0f - rho*rho) * G.y);
        v = v_next;
    }
    S = expf(S);

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;
    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    
    // reduction
    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	}

}

// Euler Scheme kernel 
__global__ void MC_Heston_Euler_kernel(float S_0, float v_0, float r, float rho, float sqrt_dt,
     float K, int N, curandState *state, float *d_sum, float *d_sum2, int n_triplets, HestonParam *d_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_triplets * N_TRAJ;
    if (idx >= total_threads) return;
    int triplet_idx = idx >> POW_TRAJ;
    if (triplet_idx >= n_triplets) return;

    __shared__ HestonParam p_shared;
    if (threadIdx.x == 0) p_shared = d_params[triplet_idx];
    __syncthreads();

    HestonParam p = p_shared;
    float kappa = p.kappa, theta = p.theta, sigma = p.sigma;
    float dt = sqrt_dt * sqrt_dt;
    float S = S_0;
    float v = v_0, v_next;
    curandState localState = state[idx];

    extern __shared__ float A[];
    float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

    for (int i = 0; i < N; i++){
        float2 G = curand_normal2(&localState);
        v_next = fmaxf(v + kappa*(theta - v)*dt + sigma * sqrtf(v) * sqrt_dt * G.x, 0.0f);
        S += r * S * dt + sqrtf(v) * S * sqrt_dt * (rho * G.x + sqrtf(1.0f - rho*rho) * G.y);
        v = v_next;
    }
    R1s[threadIdx.x] = expf(-r * dt * N) * fmaxf(0.0f, S-K)/N_TRAJ ;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * N_TRAJ;

    // reduction
    __syncthreads();
	int i = blockDim.x / 2;
	while (i!=0){
		if (threadIdx.x < i){
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;	
	}
	
	if (threadIdx.x == 0){
		atomicAdd(&d_sum[triplet_idx], R1s[0]);
		atomicAdd(&d_sum2[triplet_idx], R2s[0]);
	}

}

// Generate valid Heston parameter triplets
int generateTriplets(HestonParam *h_params) {
    float kappa_vals[10] = {0.1f,1.2f,2.3f,3.4f,4.5f,5.6f,6.7f,7.8f,8.9f,10.0f};
    float theta_vals[10] = {0.01f,0.0644f,0.1189f,0.1733f,0.2278f,0.2822f,0.3367f,0.3911f,0.4456f,0.5f};
    float sigma_vals[10] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f};
    int idx = 0;
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            for (int k = 0; k < 10; k++){
                float kappa = kappa_vals[i];
                float theta = theta_vals[j];
                float sigma = sigma_vals[k];
                if (20.0f * kappa * theta > sigma * sigma) {
                    h_params[idx++] = (HestonParam){kappa, theta, sigma};
                }
            }
        }
    }
    return idx;
}

int main(void) {
    HestonParam *h_params = (HestonParam*)malloc(MAX_TRIPLETS * sizeof(HestonParam));
    int n_triplets = generateTriplets(h_params);
    printf("Number of valid triplets = %d\n", n_triplets);
    printf("Number of trajectories per triplet = %d\n", N_TRAJ);
    int total_threads = n_triplets * N_TRAJ;
    int threadsPerBlock = 1024;
    int numBlocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock; // same as ceil(total_threads/threadsPerBlock)
    float T = 1.0f;
    float S_0 = 1.0f;
    float v_0 = 0.1f;
    float r = 0.0f;
    float rho = -0.5f;
    float K = S_0;
    int N = 1000;
    float sqrt_dt = sqrtf(T / (float)N);
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, total_threads * sizeof(curandState)));
    init_curand_state_k<<<numBlocks, threadsPerBlock>>>(d_states);

    HestonParam *d_params;
    testCUDA(cudaMalloc(&d_params, n_triplets * sizeof(HestonParam)));
    testCUDA(cudaMemcpy(d_params, h_params, n_triplets * sizeof(HestonParam), cudaMemcpyHostToDevice));

    float *d_sum, *d_sum2;
    testCUDA(cudaMallocManaged(&d_sum, n_triplets * sizeof(float)));
    testCUDA(cudaMallocManaged(&d_sum2, n_triplets * sizeof(float)));
    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }

    cudaEvent_t start, stop;

    float time_exact, time_almost, time_euler;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MC_Heston_Exact_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, rho, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_exact, start, stop);
    cudaDeviceSynchronize();

    // write csv
    FILE *fpt;
    fpt = fopen("MC_Heston_Exact.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params[i].kappa, h_params[i].theta, h_params[i].sigma, rho, price, err, time_exact/n_triplets);
    }
    fclose(fpt);

    

    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }

    cudaEventRecord(start, 0);

    MC_Heston_Almost_Exact_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, rho, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_almost, start, stop);
    cudaDeviceSynchronize();

    // write csv
    fpt = fopen("MC_Heston_Almost_Exact.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params[i].kappa, h_params[i].theta, h_params[i].sigma, rho, price, err, time_almost/n_triplets);
    }
    fclose(fpt);

    for (int i = 0; i < n_triplets; i++){
        d_sum[i] = 0.0f;
        d_sum2[i] = 0.0f;
    }

    cudaEventRecord(start, 0);

    MC_Heston_Euler_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(S_0, v_0, r, rho, sqrt_dt, K, N, d_states, d_sum, d_sum2, n_triplets, d_params);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_euler, start, stop);
    cudaDeviceSynchronize();

    // write csv
    fpt = fopen("MC_Heston_Euler.csv", "w+");
    fprintf(fpt, "triplet,kappa,theta,sigma,rho,price,err,exec_time\n");
    for (int i = 0; i < n_triplets; i++){
        float price = d_sum[i];
        float err = 1.96 * sqrt((double)(1.0f / (N_TRAJ - 1)) * (N_TRAJ*d_sum2[i] - (d_sum[i] * d_sum[i])))/sqrt((double)N_TRAJ);
        fprintf(fpt, "%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.2f\n", i, h_params[i].kappa, h_params[i].theta, h_params[i].sigma, rho, price, err, time_euler/n_triplets);
    }
    fclose(fpt);

    free(h_params);
    cudaFree(d_states);
    cudaFree(d_params);
    cudaFree(d_sum);
    cudaFree(d_sum2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
