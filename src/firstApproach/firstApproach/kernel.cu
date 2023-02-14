
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>

#define MAXCHILD 5
#define NODECOUNT 156
#define DEFNODECOUNT 81
#define FINDINGVALUE 77     //0-80

struct nodeOfGraph {
    int Value;              //hodnota uzlu
    int ChildCount;         //pocet nasledujicich prvku (Pri teto implementaci je nepotrebny ale pro jine pristupy)
    int IndexPrevNode;
    bool SetNode;
};


cudaError_t BFSWithCude(nodeOfGraph* ArrNode, nodeOfGraph* SearchNode, int LevelCount);

__global__ void BFSSearchKernel(nodeOfGraph* dev_ArrNodes, nodeOfGraph* dev_SearchNode, int startIndex)
{
    int i = threadIdx.x + startIndex;
    if (dev_ArrNodes[i].Value == FINDINGVALUE) {
        dev_SearchNode->ChildCount = dev_ArrNodes[i].ChildCount;
        dev_SearchNode->IndexPrevNode = dev_ArrNodes[i].IndexPrevNode;
        dev_SearchNode->SetNode = dev_ArrNodes[i].SetNode;
        dev_SearchNode->Value = dev_ArrNodes[i].Value;
    }
}

int main()
{
    //vytvoreni referencnich dat, hledat budeme cisla od 0 do 80
    const int ArrValue[DEFNODECOUNT] = { 0, 2, 5, 8, 4, 6, 3, 9, 1, 7, 10, 13, 11, 14, 12, 16, 17, 15, 18, 20, 19, 24, 22, 26, 23, 27, 25, 29, 28, 35, 30, 36, 31, 37, 32, 38, 33, 39, 34, 40, 45, 41, 46, 42, 47, 43, 48, 44, 49, 55, 50, 56, 51, 57, 52, 58, 53, 59, 54, 60, 65, 61, 66, 62, 67, 63, 68, 64, 69, 75, 70, 76, 71, 77, 72, 78, 73, 79, 74, 80, 21 };
    const int IndexPrevNode[DEFNODECOUNT] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30 };
    nodeOfGraph* ArrNode;
    nodeOfGraph SearchNode;
    SearchNode.Value = -1;

    ArrNode = (nodeOfGraph*)malloc(sizeof(nodeOfGraph) * NODECOUNT);

    memset(ArrNode, 0, sizeof(nodeOfGraph) * NODECOUNT);
    int ValAndPrevInedx = 0;

    //naplneni struktury daty
    for (int indexNode = 0; indexNode < NODECOUNT; indexNode++)
    {
        if (indexNode < 31 || indexNode % MAXCHILD == 1 || indexNode % MAXCHILD == 2) {
            ArrNode[indexNode].Value = ArrValue[ValAndPrevInedx];
            ArrNode[indexNode].ChildCount = 0;
            ArrNode[indexNode].IndexPrevNode = IndexPrevNode[ValAndPrevInedx];
            ArrNode[indexNode].SetNode = 1;
            ValAndPrevInedx++;
        }
        else {
            ArrNode[indexNode].Value = -1;
            ArrNode[indexNode].ChildCount = -1;
            ArrNode[indexNode].IndexPrevNode = -1;
            ArrNode[indexNode].SetNode = 0;
        }
    }

    cudaError_t cudaStatus = BFSWithCude(ArrNode, &SearchNode, 4);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "BFSWithCude failed!");
        return 1;
    }

    std::cout << "Value: " << SearchNode.Value << ", prev. index: " << SearchNode.IndexPrevNode << ", child count: " << SearchNode.ChildCount << ", set node: " << SearchNode.SetNode << std::endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t BFSWithCude(nodeOfGraph* ArrNode, nodeOfGraph* SearchNode, int LevelCount)
{
    nodeOfGraph* dev_ArrNode;
    nodeOfGraph* dev_SearchNode;
    cudaError_t cudaStatus;
    int* dev_StartIndex;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaMalloc(&dev_ArrNode, NODECOUNT * sizeof(struct nodeOfGraph));
    cudaMalloc(&dev_SearchNode, sizeof(struct nodeOfGraph));
    cudaMalloc(&dev_StartIndex, sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ArrNode, ArrNode, NODECOUNT * sizeof(struct nodeOfGraph), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        std::cerr << "Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int startIndex = 1;
    for (int i = 1; i < LevelCount; i++) {
        cudaMemcpy(dev_StartIndex, &startIndex, sizeof(int), cudaMemcpyHostToDevice);

        BFSSearchKernel << <1, (startIndex + pow(MAXCHILD, i)) - startIndex >> > (dev_ArrNode, dev_SearchNode, startIndex);

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(SearchNode, dev_SearchNode, sizeof(struct nodeOfGraph), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
            goto Error;
        }

        if (SearchNode->Value == FINDINGVALUE)
            break;

        startIndex += pow(MAXCHILD, i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Count of milliseconds: " << milliseconds << std::endl;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching BFSSearchKernel!\n", cudaStatus);
        std::cerr << "Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        goto Error;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "BFSSearchKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        std::cout << "Cuda status: " << cudaStatus << std::endl;
        goto Error;
    }

    std::cout << "All was okey\n";

Error:
    cudaFree(dev_ArrNode);
    cudaFree(dev_SearchNode);
    cudaFree(dev_StartIndex);
    return cudaStatus;
}