#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_CITIES 10

int bestValue = 0;
int answer = 0;

int distanceMatrix[MAX_CITIES][MAX_CITIES];

void swap(int *array, int a, int b) {
    int temp = array[a];
    array[a] = array[b];
    array[b] = temp;
}

int findFitnessOf(int *chromosome, int numCities) {
    int distance = 0;
    for (int i = 0; i < numCities - 1; i++) {
        distance += distanceMatrix[chromosome[i]][chromosome[i + 1]];
    }
    distance += distanceMatrix[chromosome[numCities - 1]][chromosome[0]]; 
    return distance;
}

void initializePopulation(int populationSize, int numCities, int population[populationSize][MAX_CITIES]) {
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < numCities; j++) {
            population[i][j] = j;
        }

        for (int j = 0; j < numCities; j++) {
            int r = rand() % numCities;
            swap(population[i], j, r);
        }
    }
}

void Selection(int size, int population[size][MAX_CITIES], int selectedParent[size/2][MAX_CITIES], int numCities) {
    for (int i = 0; i < (size + 1) / 2; i++) {
        int randomIndex = rand() % size;
        for (int j = 0; j < numCities; j++) {
            selectedParent[i][j] = population[randomIndex][j];
        }
    }
}

void Crossover(int populationSize, int selectedParent[(populationSize + 1) / 2][MAX_CITIES], int nextGeneration[populationSize][MAX_CITIES], int numCities) {
    for (int i = 0; i < populationSize; i++) {
        int point1 = rand() % numCities;
        int point2 = rand() % numCities;
        while (point1 == point2) {
            point2 = rand() % numCities;
        }

        int maleIndex = rand() % ((populationSize + 1) / 2);
        int femaleIndex = rand() % ((populationSize + 1) / 2);

        int male[MAX_CITIES], female[MAX_CITIES];
        for (int j = 0; j < numCities; j++) {
            male[j] = selectedParent[maleIndex][j];
            female[j] = selectedParent[femaleIndex][j];
        }

        int child[MAX_CITIES];
        for (int j = 0; j < numCities; j++) {
            child[j] = -1;
        }

        for (int j = point1; j <= point2; j++) {
            child[j] = male[j];
        }

        int currentIndex = 0;
        for (int j = 0; j < numCities; j++) {
            if (child[j] == -1) {
                while (1) {
                    int found = 0;
                    for (int k = 0; k < numCities; k++) {
                        if (female[k] == male[currentIndex]) {
                            currentIndex = (currentIndex + 1) % numCities;
                            found = 1;
                            break;
                        }
                    }
                    if (!found) break;
                }
                child[j] = female[currentIndex];
                currentIndex = (currentIndex + 1) % numCities;
            }
        }

        for (int j = 0; j < numCities; j++) {
            nextGeneration[i][j] = child[j];
        }
    }
}

void Mutation(int populationSize, int population[populationSize][MAX_CITIES], int mutatedGeneration[populationSize][MAX_CITIES], int numCities) {
    for (int i = 0; i < populationSize; i++) {
        int a = rand() % numCities;
        int b = rand() % numCities;
        while (b == a) {
            b = rand() % numCities;
        }
        swap(population[i], a, b);
        for (int j = 0; j < numCities; j++) {
            mutatedGeneration[i][j] = population[i][j];
        }
    }
}

int solutionNode(int populationSize, int population[populationSize][MAX_CITIES], int numCities) {
    int bestFitness = 1e9;
    int bestChromosome = 0;
    for (int i = 0; i < populationSize; i++) {
        int fitness = findFitnessOf(population[i], numCities);
        if (fitness < bestFitness) {
            bestFitness = fitness;
            bestChromosome = i;
        }
    }
    return bestChromosome;
}

void geneticAlgorithm(int populationSize, int numCities, int terminationCondition) {
    int currentGeneration[populationSize][MAX_CITIES];
    int selectedParent[(populationSize + 1) / 2][MAX_CITIES];
    int afterCrossover[populationSize][MAX_CITIES];
    int afterMutation[populationSize][MAX_CITIES];

    initializePopulation(populationSize, numCities, currentGeneration);

    int currentBest = solutionNode(populationSize, currentGeneration, numCities);
    printf("Best distance: %d\n", findFitnessOf(currentGeneration[currentBest], numCities));

    if (terminationCondition == 1) {  
        for (int i = 0; i < 100; i++) {
            Selection(populationSize, currentGeneration, selectedParent, numCities);
            Crossover(populationSize, selectedParent, afterCrossover, numCities);
            Mutation(populationSize, afterCrossover, afterMutation, numCities);
            for (int j = 0; j < populationSize; j++) {
                for (int k = 0; k < numCities; k++) {
                    currentGeneration[j][k] = afterMutation[j][k];
                }
            }
            currentBest = solutionNode(populationSize, currentGeneration, numCities);
            printf("Generation %d, Best distance: %d\n", i + 1, findFitnessOf(currentGeneration[currentBest], numCities));
        }
    }
}

int main() {
    srand(time(NULL)); 

    int populationSize, numCities, terminationCondition;

    printf("Enter number of cities: ");
    scanf("%d", &numCities);


    for (int i = 0; i < numCities; i++) {
        for (int j = 0; j < numCities; j++) {
            if (i == j) {
                distanceMatrix[i][j] = 0; 
            } else {
                distanceMatrix[i][j] = rand() % 100 + 1; 
            }
        }
    }

    printf("Enter population size: ");
    scanf("%d", &populationSize);

    printf("Enter termination condition (1 for generational): ");
    scanf("%d", &terminationCondition);

    geneticAlgorithm(populationSize, numCities, terminationCondition);

    return 0;
}