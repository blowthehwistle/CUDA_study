#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16
// ==========================================
// 1. Naive Kernel
// ==========================================

__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, 
                                             int M, int N, int K) {
    // x가 열(Col/N), y가 행(Row/M)을 담당
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // 범위 체크 (M행 N열)
    if(row >= M || col >= N)
        return;
    
    float tmp = 0.0f;
    // K차원을 따라 내적 수행
    for(int i = 0; i < K; i++) {
        // A는 [row][i], B는 [i][col]
        tmp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = tmp;
}

// ==========================================
// 2. Global Memory Coalescing
// ==========================================

__global__ void MEM_Coalescing_matmul_kernel(const float* A, const float* B, float* C, 
                                             int M, int N, int K) {
    // x가 열(Col/N), y가 행(Row/M)을 담당
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 범위 체크 (M행 N열)
    if(row >= M || col >= N)
        return;
    
    float tmp = 0.0f;
    // K차원을 따라 내적 수행
    for(int i = 0; i < K; i++) {
        // A는 [row][i], B는 [i][col]
        tmp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = tmp;
}


// ==========================================
// 3. Shared Memory Tiling
// ==========================================


// 표준 행렬 곱셈: C(M x N) = A(M x K) * B(K x N)
__global__ void shared_tiling_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int alpha = 1;
    int beta = 1;
    // 범위 체크 (M행 N열)
    if(cRow >= M || cCol >= N)
        return;
    
    A += cRow * BLOCK_SIZE * K; // 아래 방향으로 이동 (내 담당 row 찾기 위해)
    B += cCol * BLOCK_SIZE; // 오른쪽으로 이동 (내 담당 col 찾기 위해)
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; //아래방향으로 + 오른쪽 방향으로 갈 칸 수


    float tmp = 0.0f;
    for(int bkIdx=0; bkIdx<K; bkIdx+=BLOCK_SIZE){   // 타일 단위 움직임 
        As[threadRow*BLOCK_SIZE+threadCol] = A[threadRow*K+threadCol];  //타일 단위 복사 (병렬)
        Bs[threadRow*BLOCK_SIZE+threadCol] = B[threadRow*N+threadCol];
        __syncthreads();

        A += BLOCK_SIZE;        // 오른쪽 방향으로 이동
        B += BLOCK_SIZE * N;    // 아래 방향으로 이동 (짝 맞춰 계산)
        
        //inner tile calculation
        for(int dotIdx=0; dotIdx<BLOCK_SIZE; ++dotIdx){ // 타일 내 Thread 단위 움직임
            tmp += As[threadRow*BLOCK_SIZE+dotIdx] * Bs[dotIdx*BLOCK_SIZE+threadCol];
        }

        __syncthreads();

        C[threadRow*N+threadCol] = alpha*tmp + beta*C[threadRow*N+threadCol];
    }

}

// ==========================================
// 4. 1D Blocktiling for Calculating multiple results per Thread
// ==========================================


// 표준 행렬 곱셈: C(M x N) = A(M x K) * B(K x N)
__global__ void blocktiling_1D_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float threadResults[TM] = {0.0};    //TM size만한 배열 선언 (in Register)
    
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int alpha = 1;
    int beta = 1;
    // 범위 체크 (M행 N열)
    if(cRow >= M || cCol >= N)
        return;
    
    A += cRow * BLOCK_SIZE * K; // 아래 방향으로 이동 (내 담당 row 찾기 위해)
    B += cCol * BLOCK_SIZE; // 오른쪽으로 이동 (내 담당 col 찾기 위해)
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; //아래방향으로 + 오른쪽 방향으로 갈 칸 수


    float tmp = 0.0f;
    for(int bkIdx=0; bkIdx<K; bkIdx+=BLOCK_SIZE){   // 타일 단위 움직임 
        As[threadRow*BLOCK_SIZE+threadCol] = A[threadRow*K+threadCol];  //타일 단위 복사 (병렬)
        Bs[threadRow*BLOCK_SIZE+threadCol] = B[threadRow*N+threadCol];
        __syncthreads();

        A += BK;        // 오른쪽 방향으로 이동
        B += BK*N;    // 아래 방향으로 이동 (짝 맞춰 계산)
        
        for(uint dotIdx=0; dotIdx<BK; ++dotIdx){
            float Btmp = Bs[dotIdx * BN + threadCol];   //TM번 재활용될 B의 값 하나 (그림의 주황색 점)
            for(uint resIdx = 0; resIdx < TM; ++resIdx){
                //threadRow * TM : 시작 index
                // row * 가로길이 + col
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        for(int dotIdx=0; dotIdx<BLOCK_SIZE; ++dotIdx){ // 타일 내 Thread 단위 움직임
            tmp += As[threadRow*BLOCK_SIZE+dotIdx] * Bs[dotIdx*BLOCK_SIZE+threadCol];
        }
    }
    __syncthreads();

}


// 호스트에서 호출하는 래퍼 함수
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    // 그리드 계산: (N / 16, M / 16)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    shared_tiling_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    // 에러 체크 및 동기화
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// ==========================================
// 2. 유틸리티 함수 (초기화 및 검증)
// ==========================================

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const float* A, const float* B, const float* C, int M, int N, int K) {
    // 결과 검증은 시간이 오래 걸리므로 일부 샘플만 확인하거나 작은 행렬에서만 수행
    // 여기서는 0번 행, 0번 열의 값 하나만 간단히 비교 (정밀 검증 아님)
    float val = 0.0f;
    for(int i=0; i<K; i++) {
        val += A[0 * K + i] * B[i * N + 0];
    }
    
    // 오차 범위 확인
    if (fabs(C[0] - val) > 1e-3) {
        printf("Verification Failed! Host: %f, Device: %f\n", val, C[0]);
        return false;
    }
    return true;
}

// ==========================================
// 3. Main 함수
// ==========================================

int main() {
    int choice;
    int M, N, K;

    std::cout << "Select Matrix Size:\n";
    std::cout << "1. Small (32 x 32) - for Correctness Check\n";
    std::cout << "2. Large (4096 x 4096) - for Performance Check\n";
    std::cout << "Enter choice (1 or 2): ";
    std::cin >> choice;

    if (choice == 1) {
        M = 32; N = 32; K = 32;
    } else if (choice == 2) {
        M = 4096; N = 4096; K = 4096;
    } else {
        std::cout << "Invalid choice. Exiting.\n";
        return -1;
    }

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::cout << "\nInitializing Data (" << M << "x" << N << ")..." << std::endl;

    // 호스트 메모리 할당
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // 데이터 초기화
    srand(2025);
    randomInit(h_A, M * K);
    randomInit(h_B, K * N);

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 데이터 복사 (Host -> Device)
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // CUDA Event를 이용한 시간 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Executing Kernel..." << std::endl;
    
    cudaEventRecord(start);
    
    // 커널 실행
    solve(d_A, d_B, d_C, M, N, K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 결과 복사 (Device -> Host)
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "Done!" << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // 간단 검증
    if (choice == 1) {
        if (verifyResult(h_A, h_B, h_C, M, N, K)) {
            std::cout << "Result Verification: PASSED" << std::endl;
        }
    } else {
        std::cout << "(Skipping full verification for large matrix)" << std::endl;
        std::cout << "C[0] value check: " << h_C[0] << std::endl;
    }

    // 메모리 해제
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}