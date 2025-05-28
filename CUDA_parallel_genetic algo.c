#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <limits.h>
#include "crossover_operators.h"
#define MAX_CITIES 10000
#define BLOCK_SIZE 256
#define MAX_GENERATIONS 10000
#define CONSECUTIVE_THRESHOLD 100

_device_ void swap(int *array, int a, int b) {
    int temp = array[a];
    array[a] = array[b];
    array[b] = temp;
}

_device_ int d_rand(int seed) {
    return (seed * 1103515245 + 12345) & 0x7FFFFFFF;
}

// Tournament selection: each thread selects one parent index
_global_ void selectionKernel(int *fitness, int *selectedParentIdx, int populationSize, int tournamentSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    int best = d_rand(idx) % populationSize;
    int bestFitness = fitness[best];
    for (int i = 1; i < tournamentSize; i++) {
        int competitor = d_rand(idx + i * 123) % populationSize;
        int competitorFitness = fitness[competitor];
        if (competitorFitness < bestFitness) {
            best = competitor;
            bestFitness = competitorFitness;
        }
    }
    selectedParentIdx[idx] = best;
}

// Ordered Crossover: each thread creates one child
_global_ void crossoverKernel(int *population, int *selectedParentIdx, int *nextGeneration, int populationSize, int numCities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    int parent1_idx = selectedParentIdx[idx];
    int parent2_idx = selectedParentIdx[(idx+1)%populationSize];

    int *parent1 = &population[parent1_idx * numCities];
    int *parent2 = &population[parent2_idx * numCities];
    int *child = &nextGeneration[idx * numCities];

    int point1 = d_rand(idx) % numCities;
    int point2 = d_rand(idx + 1) % numCities;
    if (point1 > point2) {
        int temp = point1;
        point1 = point2;
        point2 = temp;
    }

    // Copy segment from parent1
    bool used[MAX_CITIES] = {0};
    for (int j = point1; j <= point2; j++) {
        child[j] = parent1[j];
        used[child[j]] = true;
    }

    // Fill remaining from parent2
    int p2_idx = 0;
    for (int j = 0; j < numCities; j++) {
        if (j >= point1 && j <= point2) continue;
        while (used[parent2[p2_idx]]) p2_idx++;
        child[j] = parent2[p2_idx];
        used[child[j]] = true;
        p2_idx++;
    }
}

_global_ void mutationKernel(int *population, int populationSize, int numCities, float mutationRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        int offset = idx * numCities;
        unsigned int seed = (idx + 1) * clock64();
        for (int j = 0; j < numCities; j++) {
            if ((d_rand(seed + j) % 1000) < (mutationRate * 1000)) {
                int a = d_rand(seed + 2*j) % numCities;
                int b = d_rand(seed + 3*j) % numCities;
                if (a != b) swap(population + offset, a, b);
            }
        }
    }
}

_global_ void initializePopulationKernel(int *population, int populationSize, int numCities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        int offset = idx * numCities;
        for(int j=0; j<numCities; j++)
            population[offset + j] = j;
        
        unsigned int seed = (idx + 1) * clock64();
        for(int j=numCities-1; j>0; j--) {
            int r = d_rand(seed) % (j + 1);
            swap(population + offset, j, r);
            seed = r + clock();
        }
    }
}

_global_ void calculateFitnessKernel(int *population, int *fitness, int populationSize, int numCities, int *distanceMatrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        int *chromosome = &population[idx * numCities];
        int distance = 0;
        for (int i = 0; i < numCities - 1; i++) {
            int from = chromosome[i];
            int to = chromosome[i+1];
            distance += distanceMatrix[from * numCities + to];
        }
        int last = chromosome[numCities-1];
        int first = chromosome[0];
        distance += distanceMatrix[last * numCities + first];
        fitness[idx] = distance;
    }
}

_global_ void findBestFitnessKernel(int *fitness, int populationSize, int *bestIndex) {
    _shared_ int sharedBest[BLOCK_SIZE];
    _shared_ int sharedIndex[BLOCK_SIZE];

    int idx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + idx;
    int best = INT_MAX;
    int bestIdx = -1;

    while(globalIdx < populationSize) {
        if(fitness[globalIdx] < best) {
            best = fitness[globalIdx];
            bestIdx = globalIdx;
        }
        globalIdx += gridDim.x * blockDim.x;
    }

    sharedBest[idx] = best;
    sharedIndex[idx] = bestIdx;
    __syncthreads();

    for(int s=blockDim.x/2; s>0; s>>=1) {
        if(idx < s) {
            if(sharedBest[idx + s] < sharedBest[idx]) {
                sharedBest[idx] = sharedBest[idx + s];
                sharedIndex[idx] = sharedIndex[idx + s];
            }
        }
        __syncthreads();
    }

    if(idx == 0) {
        bestIndex[blockIdx.x] = sharedIndex[0];
    }
}

_global_ void copyPopulationKernel(int *src, int *dst, int numIndividuals, int numCities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndividuals * numCities) {
        dst[idx] = src[idx];
    }
}

int main() {
    srand(time(NULL));
    int populationSize = 2000;
    int numCities = 10000;
    printf("Enter number of cities: ");
    scanf("%d", &numCities);

    int *distanceMatrix, *d_distanceMatrix;
    distanceMatrix = (int*)malloc(numCities * numCities * sizeof(int));
    cudaMalloc(&d_distanceMatrix, numCities * numCities * sizeof(int));
    for(int i=0; i<numCities; i++) {
        for(int j=0; j<numCities; j++) {
            distanceMatrix[i * numCities + j] = (i == j) ? 0 : rand()%100 + 1;
        }
    }
    cudaMemcpy(d_distanceMatrix, distanceMatrix, numCities * numCities * sizeof(int), cudaMemcpyHostToDevice);

    printf("Enter population size: ");
    scanf("%d", &populationSize);

    size_t populationBytes = populationSize * numCities * sizeof(int);
    size_t fitnessBytes = populationSize * sizeof(int);

    int *d_population, *d_nextGeneration, *d_fitness, *d_bestIndices, *d_selectedParents;
    int h_fitness = (int)malloc(fitnessBytes);
    int h_bestIndices = (int)malloc(sizeof(int));

    cudaMalloc(&d_population, populationBytes);
    cudaMalloc(&d_nextGeneration, populationBytes);
    cudaMalloc(&d_fitness, fitnessBytes);
    cudaMalloc(&d_bestIndices, sizeof(int));
    cudaMalloc(&d_selectedParents, populationSize * sizeof(int));

    dim3 blocks((populationSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initialize population
    initializePopulationKernel<<<blocks, BLOCK_SIZE>>>(d_population, populationSize, numCities);
    cudaDeviceSynchronize();

    int consecutiveCount = 0;
    int globalBest = INT_MAX;
    int currentBest;

    float mutationRate = 0.02f; // 2% mutation rate

    clock_t start = clock();

    for(int gen=0; gen<MAX_GENERATIONS; gen++) {
        // Calculate fitness
        calculateFitnessKernel<<<blocks, BLOCK_SIZE>>>(d_population, d_fitness, populationSize, numCities, d_distanceMatrix);

        // Selection (tournament)
        selectionKernel<<<blocks, BLOCK_SIZE>>>(d_fitness, d_selectedParents, populationSize, 5);

        // Crossover (ordered)
        crossoverKernel<<<blocks, BLOCK_SIZE>>>(d_population, d_selectedParents, d_nextGeneration, populationSize, numCities);

        // Mutation
        mutationKernel<<<blocks, BLOCK_SIZE>>>(d_nextGeneration, populationSize, numCities, mutationRate);

        // Copy nextGeneration to population for next iteration
        copyPopulationKernel<<<(populationSize*numCities+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            d_nextGeneration, d_population, populationSize, numCities);

        // Find global best
        findBestFitnessKernel<<<1, BLOCK_SIZE>>>(d_fitness, populationSize, d_bestIndices);
        cudaMemcpy(h_bestIndices, d_bestIndices, sizeof(int), cudaMemcpyDeviceToHost);
        int bestIdx = h_bestIndices[0];
        cudaMemcpy(&currentBest, &d_fitness[bestIdx], sizeof(int), cudaMemcpyDeviceToHost);

        if(currentBest < globalBest) {
            globalBest = currentBest;
            consecutiveCount = 0;
        } else {
            consecutiveCount++;
        }

        printf("Generation %3d: Current Best: %d, Global Best: %d\n", gen+1, currentBest, globalBest);

        if(consecutiveCount >= CONSECUTIVE_THRESHOLD) {
            printf("\nNo improvement for %d generations. Stopping early.\n", CONSECUTIVE_THRESHOLD);
            break;
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nFinal Global Best Distance: %d\n", globalBest);
    printf("Execution Time: %.2f seconds\n", time_taken);

    // Cleanup
    free(distanceMatrix);
    free(h_fitness);
    free(h_bestIndices);
    cudaFree(d_distanceMatrix);
    cudaFree(d_population);
    cudaFree(d_nextGeneration);
    cudaFree(d_fitness);
    cudaFree(d_bestIndices);
    cudaFree(d_selectedParents);

    return 0;
}