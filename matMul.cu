#include<iostream>
#include<cuda.h>

using namespace std;

#define N 64
#define TILE 16

__global__ void matrixMul(float *A, float *B, float *C){
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if(row < N && col < N){
        float sum = 0;
        for(int k = 0; k < N;k++){
          sum += A[row * N + k] * B[k * N + col];
        }

        C[row*N + col] = sum;
    }
}

int main(){
    float *A, *B, *C;
    float h_A[N*N], h_B[N*N], h_C[N*N];

    for(int i=0;i<N*N;i++){
      h_A[i] = i;
      h_B[i] = 1.0f;
    }

    cudaMalloc(&A, N*N*sizeof(float));
    cudaMalloc(&B, N*N*sizeof(float));
    cudaMalloc(&C, N*N*sizeof(float));

    cudaMemcpy(A, h_A, N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, N*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N/TILE, N/TILE);

    matrixMul<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<16;i++){
      for(int j=0;j<16;j++){
        cout<<h_C[i*N+j]<<" ";
      }
      cout<<endl;
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}