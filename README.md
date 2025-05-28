# Performance Analysis of a Parallel Genetic Algorithm for the Travelling Salesman Problem

## Project Overview  
This project implements a **parallel genetic algorithm (GA)** to solve the Travelling Salesman Problem (TSP) and evaluates its performance on modern hardware. The GA encodes candidate tours as sequences of city IDs and evolves solutions using selection, crossover, and mutation. We parallelize the GA on both multicore CPUs and GPUs to reduce runtime on large TSP instances. Key findings include dramatic speedups on GPU hardware compared to a sequential or limited-core CPU implementation.

## Motivation and Background  
- The Travelling Salesman Problem is NP-hard, meaning exact algorithms do not scale to large city counts. Heuristic methods like genetic algorithms are widely used to find good approximate solutions for TSP.  
- Genetic algorithms evolve a population of candidate routes using biologically inspired operators. This process is naturally parallel: evaluating and generating many individuals can be done in parallel.  
- Modern parallel computing platforms (multicore CPUs and many-core GPUs) offer the opportunity to accelerate GA computations. In this study, we explore how a GA for TSP performs when parallelized using OpenMP on CPU and CUDA on GPU.

## Methodology

### Genetic Algorithm for TSP  
- **Representation:** Each individual (chromosome) is a permutation of city IDs representing a tour. The fitness is the total distance of the tour.  
- **Operators:** We use a standard GA with selection, crossover, and mutation. Multiple crossover operators were tested, including **Order-based**, **Partially Matched (PMX)**, **Edge Recombination (ERX)**, and **Single-Point Crossover (SCX)**, to study their performance.  
- **Population & Generations:** A population of candidate tours is evolved over many generations. Elitism and roulette-wheel (or tournament) selection ensure the best tours are retained and used to produce offspring.  

### Parallelization with OpenMP and CUDA  
- **OpenMP (CPU):** We use OpenMP pragmas to parallelize GA loops on a multicore CPU. In our experiments on Kaggle’s setup, only 2 CPU cores were available, so OpenMP provided modest speedup (~1.8×) over sequential execution.  
- **CUDA (GPU):** The GA is also implemented on NVIDIA GPU using CUDA. Fitness calculations and genetic operators are offloaded to thousands of GPU threads. Each CUDA thread or thread-block may process one or more individuals independently.  
- **Execution Environment:** All experiments were run on the Kaggle platform, which provides access to an NVIDIA Tesla P100 GPU (13 GB RAM) and a 2-core CPU.  

## Performance Analysis Summary  
Our experiments demonstrate that parallelization yields significant runtime improvements:  
- **GPU Acceleration:** CUDA-parallel GA achieved **≈45×–55× speedup** compared to a single-thread CPU implementation for large TSPs.  
- **CPU Multithreading:** OpenMP on 2 CPU cores provided only **moderate speedup (~1.7×–1.9×)**.  
- **Consistency with Prior Work:** Speedups are consistent with other studies, confirming that GPUs are well-suited for this workload.

## Key Results  
- **Speedup by Problem Size:**  
  - *1,000 cities:* CPU 22.0 s, GPU 0.48 s ⇒ **~46× speedup**  
  - *5,000 cities:* CPU 690 s, GPU 15.2 s ⇒ **~45× speedup**  
  - *10,000 cities:* CPU 5,670 s, GPU 103 s ⇒ **~55× speedup**  
- **Crossover Operator Performance:**  
  - **PMX:** Best performance, 93% GPU utilization  
  - **ERX:** Worst performance, 42% utilization  
  - **Order-Based & SCX:** Intermediate performance  

## Limitations and Future Work  
- **Sequential Generation Dependency:** Limits parallelism across generations.  
- **Memory Constraints:** Larger TSP instances require memory optimizations.  
- **Operator-Specific Bottlenecks:** ERX is too costly for large problems.  
- **Future Work Suggestions:**  
  - Hybrid CPU–GPU strategies  
  - Quantized/compact data representations  
  - Adaptive crossover techniques  

## Contributors
- **Parth Vijay** – CUDA implementation ([GitHub](https://github.com/CS-parth))
- **Srikar Chaturvedula** – OpenMP implementation ([GitHub](https://github.com/Srikaludev))
- **Sudip Halder** – Performance analysis ([GitHub](https://github.com/sudipme))
