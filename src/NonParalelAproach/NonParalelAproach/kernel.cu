
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

#define MAXCHILD 5
#define NODECOUNT 156
#define DEFNODECOUNT 81
#define FINDINGVALUE 77     //0-80

struct nodeOfGraph {
    int Value;              //hodnota uzlu
    int ChildCount;         //pocet nasledujicich prvku (Pri teto implementaci je nepotrebny ale pro jine pristupy)
    int IndexPrevNode;
    bool SetNode;
    bool Visited;
};

int main()
{
    //vytvoreni referencnich dat, hledat budeme cisla od 0 do 80
    const int ArrValue[DEFNODECOUNT] = { 0, 2, 5, 8, 4, 6, 3, 9, 1, 7, 10, 13, 11, 14, 12, 16, 17, 15, 18, 20, 19, 24, 22, 26, 23, 27, 25, 29, 28, 35, 30, 36, 31, 37, 32, 38, 33, 39, 34, 40, 45, 41, 46, 42, 47, 43, 48, 44, 49, 55, 50, 56, 51, 57, 52, 58, 53, 59, 54, 60, 65, 61, 66, 62, 67, 63, 68, 64, 69, 75, 70, 76, 71, 77, 72, 78, 73, 79, 74, 80, 21 };
    const int IndexPrevNode[DEFNODECOUNT] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30 };
    nodeOfGraph* ArrNode = (nodeOfGraph*)malloc(sizeof(struct nodeOfGraph) * NODECOUNT);
    nodeOfGraph* SearchNode = (nodeOfGraph*)malloc(sizeof(struct nodeOfGraph));

    int ValAndPrevInedx = 0;
    //naplneni struktury daty
    for (int indexNode = 0; indexNode < NODECOUNT; indexNode++)
    {
        if (indexNode < 31 || indexNode % MAXCHILD == 1 || indexNode % MAXCHILD == 2) {
            
            ArrNode[indexNode].Value = ArrValue[ValAndPrevInedx];
            ArrNode[indexNode].ChildCount = (indexNode < 31) ? 5 : 2;
            ArrNode[indexNode].IndexPrevNode = IndexPrevNode[ValAndPrevInedx];
            ArrNode[indexNode].SetNode = 1;
            ArrNode[indexNode].Visited = 0;
            ValAndPrevInedx++;
        }
        else {
            ArrNode[indexNode].Value = -1;
            ArrNode[indexNode].ChildCount = -1;
            ArrNode[indexNode].IndexPrevNode = -1;
            ArrNode[indexNode].SetNode = 0;
            ArrNode[indexNode].Visited = 0;
        }
    }


    //BFS
    vector<int> VisitedNodes;
    vector<int> QueueNodes;

    //vlozime koren
    QueueNodes.push_back(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (QueueNodes.size())
    {
        int index = QueueNodes.front();

        if (!ArrNode[index].Visited) 
        {
            if (ArrNode[index].Value == FINDINGVALUE) 
            {
                SearchNode->Value = ArrNode[index].Value;
                SearchNode->ChildCount = ArrNode[index].ChildCount;
                SearchNode->IndexPrevNode = ArrNode[index].IndexPrevNode;
                SearchNode->SetNode = ArrNode[index].SetNode;
                SearchNode->Visited = ArrNode[index].Visited;
                break;
            }
            else
            {
                   
                for (int i = 1; i <= ArrNode[index].ChildCount; i++)
                {
                    QueueNodes.push_back((index * MAXCHILD) + i);
                }
            }

            ArrNode[index].Visited = 1;
            VisitedNodes.push_back(index);
        }

        QueueNodes.erase(QueueNodes.begin());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Count of milliseconds: " << milliseconds << std::endl;

    std::cout << "Value: " << SearchNode->Value << ", prev. index: " << SearchNode->IndexPrevNode << ", child count: " << SearchNode->ChildCount << ", set node: " << SearchNode->SetNode << ", visited node: " << SearchNode->Visited << std::endl;

    free(ArrNode);
    free(SearchNode);

    return 0;
}
