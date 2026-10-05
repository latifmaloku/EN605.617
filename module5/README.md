# CUDA RGB to Grayscale Preprocessing

## Overview

This project implements RGB-to-grayscale image preprocessing using CUDA. The
program demonstrates several types of CUDA memory and compares the performance
of global memory and shared memory implementations.
The program generates a synthetic 1024 x 1024 RGB image, transfers the image
to the GPU, converts the pixels to grayscale, verifies the GPU results against
a CPU calculation, and saves the resulting grayscale image as a PNG file.

## CUDA Memory Usage

The program uses the following memory types:

- **Host Memory**
- **Global Memory**
- **Shared Memory**
- **Constant Memory**

### Build

Run:

```bash
./build.sh
```

This will generate:

* `assignment.exe` — GPU version

### Run

```bash
./run.sh <numThreads> <threadsPerBlock>
```

or:

```bash
./assignment.exe <numThreads> <threadsPerBlock>
```

For example:

```bash
./assignment.exe 512 256
```

If the requested number of threads is not evenly divisible by the block size, the GPU program rounds the total number of threads up to the next complete block.

### Sample output
sample output is provided in output.txt
in addition before and after images are saved
