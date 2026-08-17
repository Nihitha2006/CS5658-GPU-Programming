#include <iostream>
#include <cuda_runtime.h>

__device__ int barrierCount = 0;
__device__ volatile int globalFlag = 0;

__device__ void globalBarrier(){

    __shared__ int blockFlag;
    __syncthreads();

    if(threadIdx.x == 0){
        blockFlag = 1 - globalFlag;

        int oldCount = atomicAdd(&barrierCount,1);

        if(oldCount == gridDim.x - 1){
            barrierCount = 0;
            globalFlag = blockFlag;
        }
    }

    __syncthreads();

    while(globalFlag != blockFlag){

    }

    __syncthreads();
}

__global__ void testBarrier(){
    int tid = threadIdx.x;
    int bId = blockIdx.x;

    printf("Before barrier: Block %d, Thread %d\n", bId, tid);

    globalBarrier();

    printf("After barrier: Block %d, Thread %d\n", bId, tid);
}

int main(int argc, char* argv[]){
    if(argc != 3){
        std::cout << "Usage: ./globalBarrier <numBlocks> <threadsPerBlock>\n";
        return 1;
    }

    int numBlocks = std::stoi(argv[1]);
    int TPB = std::stoi(argv[2]);

    testBarrier<<<numBlocks,TPB>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if(error != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    return 0;
}