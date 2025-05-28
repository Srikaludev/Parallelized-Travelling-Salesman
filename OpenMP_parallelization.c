#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include <sys/time.h>

#define MAX_CITIES 10000
#define MAX_GENERATIONS 5000
#define CONSECUTIVE_THRESHOLD 10000

int distanceMatrix[MAX_CITIES][MAX_CITIES];

// Function to swap two elements in an array
void swap(int *chromosome, int a, int b) {
    int temp = chromosome[a];
    chromosome[a] = chromosome[b];
    chromosome[b] = temp;
}

// Function to initialize a population with random permutations
void initializePopulation(int *population, int populationSize, int numCities) {
    #pragma omp parallel for
    for(int idx = 0; idx < populationSize; idx++) {
        int offset = idx * numCities;
        // Initialize with sequential order
        for(int j = 0; j < numCities; j++) {
            population[offset + j] = j;
        }

        // Fisher-Yates shuffle
        unsigned int seed = (idx + 1) * time(NULL);
        for(int j = numCities - 1; j > 0; j--) {
            int r = rand_r(&seed) % (j + 1);
            swap(population + offset, j, r);
        }
    }
}

// Calculate fitness (tour distance) for all chromosomes
void calculateFitness(int *population, int *fitness, int populationSize, int numCities) {
    #pragma omp parallel for
    for(int idx = 0; idx < populationSize; idx++) {
        int *chromosome = &population[idx * numCities];
        int distance = 0;

        // Calculate total distance of tour
        for(int i = 0; i < numCities - 1; i++) {
            distance += distanceMatrix[chromosome[i]][chromosome[i+1]];
        }
        // Add distance back to starting city
        distance += distanceMatrix[chromosome[numCities-1]][chromosome[0]];

        fitness[idx] = distance;
    }
}

// Mutation operator - swap two random cities in each chromosome
void mutation(int *population, int populationSize, int numCities) {
    #pragma omp parallel for
    for(int idx = 0; idx < populationSize; idx++) {
        int offset = idx * numCities;
        unsigned int seed = (idx + 1) * time(NULL);

        int a = rand_r(&seed) % numCities;
        seed += a;
        int b = rand_r(&seed) % numCities;

        swap(population + offset, a, b);
    }
}

// Find index of chromosome with best fitness
int findBestFitness(int *fitness, int populationSize) {
    int bestIdx = 0;
    int bestFitness = fitness[0];

    #pragma omp parallel
    {
        int localBestIdx = 0;
        int localBestFitness = fitness[0];

        #pragma omp for nowait
        for(int i = 0; i < populationSize; i++) {
            if(fitness[i] < localBestFitness) {
                localBestFitness = fitness[i];
                localBestIdx = i;
            }
        }

        #pragma omp critical
        {
            if(localBestFitness < bestFitness) {
                bestFitness = localBestFitness;
                bestIdx = localBestIdx;
            }
        }
    }

    return bestIdx;
}

// Print the best tour found
void printBestTour(int *population, int numCities, int bestIdx) {
    printf("Best tour: ");
    int *bestTour = &population[bestIdx * numCities];
    for(int i = 0; i < numCities; i++) {
        printf("%d ", bestTour[i]);
    }
    printf("\n");
}

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double start_time, end_time;
    start_time = get_time();
    srand(time(NULL));
    int populationSize = 5;
    int numCities = 10;

    printf("Enter number of cities (max %d): ", MAX_CITIES);
    scanf("%d", &numCities);

    if(numCities > MAX_CITIES || numCities < 2) {
        printf("Invalid number of cities. Using default value of 10.\n");
        numCities = 10;
    }

    // Initialize distance matrix
    printf("Generating random distance matrix...\n");
    for(int i = 0; i < numCities; i++) {
        for(int j = 0; j < numCities; j++) {
            distanceMatrix[i][j] = (i == j) ? 0 : rand() % 100 + 1;
        }
    }

    // Print distance matrix
    // printf("Distance matrix:\n");
    // for(int i = 0; i < numCities; i++) {
    //     for(int j = 0; j < numCities; j++) {
    //         printf("%3d ", distanceMatrix[i][j]);
    //     }
    //     printf("\n");
    // }

    printf("Enter population size: ");
    scanf("%d", &populationSize);

    if(populationSize < 2) {
        printf("Invalid population size. Using default value of 100.\n");
        populationSize = 100;
    }

    // Allocate memory
    int population = (int)malloc(populationSize * numCities * sizeof(int));
    int fitness = (int)malloc(populationSize * sizeof(int));

    if(!population || !fitness) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Initialize OpenMP
    omp_set_num_threads(omp_get_max_threads());
    printf("Using %d OpenMP threads\n", omp_get_max_threads());

    // Initialize population
    initializePopulation(population, populationSize, numCities);

    int consecutiveCount = 0;
    int globalBest = INT_MAX;
    int currentBest;
    int bestIdx;

    // Main evolution loop
    double loop_start_time = get_time();
    for(int gen = 0; gen < MAX_GENERATIONS; gen++) {
        calculateFitness(population, fitness, populationSize, numCities);
        bestIdx = findBestFitness(fitness, populationSize);
        currentBest = fitness[bestIdx];

        if(currentBest < globalBest) {
            globalBest = currentBest;
            consecutiveCount = 0;
        } else {
            consecutiveCount++;
        }

        printf("Generation %3d: Current Best: %4d, Global Best: %4d\n",
               gen+1, currentBest, globalBest);

        // Mutation for next generation
        mutation(population, populationSize, numCities);

        // Early termination if no improvement for several generations
        if(consecutiveCount >= CONSECUTIVE_THRESHOLD) {
            printf("No improvement for %d consecutive generations. Stopping.\n", CONSECUTIVE_THRESHOLD);
            break;
        }
    }

    double loop_end_time = get_time();
    printf("\nFinal Global Best Distance: %d\n", globalBest);
    printBestTour(population, numCities, bestIdx);

    printf("Evolution loop execution time: %.6f seconds\n", loop_end_time - loop_start_time);

    // Free memory
    free(population);
    free(fitness);

    end_time = get_time();
    printf("Total execution time: %.6f seconds\n", end_time - start_time);

    return 0;
}