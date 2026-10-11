# CUDA RGB-to-Grayscale Conversion Using Streams and Events

## Overview

This project implements RGB-to-grayscale image conversion using CUDA streams and events. The program generates a synthetic RGB image, transfers image data between host and device memory, converts RGB pixels to grayscale on the GPU, verifies the results against a CPU calculation, and reports execution times. Multiple dataset image sizes are to compare the performance of single-stream and multi-stream execution.

## CUDA Streams and Events

The program compares two execution approaches.

### Single-stream

The single-stream performs the following operations in sequence:

1. Copies the RGB image from host memory to device memory.
2. Launches the rgb-to-grayscale conversion kernel.
3. Copies the rgb-to-grayscale result from device memory to host memory.
4. Uses CUDA events to measure the elapsed time.

### Multi-stream

The multi-stream divides the image into chunks and assigns each chunk to a separate CUDA stream, and performs the following operations for each stream:

1. Copies its RGB chunk to device memory.
2. Launches the rgb-to-grayscale conversion kernel for that chunk.
3. Copies its rgb-to-grayscale result back to host memory.
4. Records a completion event.

The program waits for all streams to finish before recording the ending event and calculating the elapsed time.

## Testing
Single-stream and multi-stream execution for multiple image sizes (512, 1024, and user specified) are compared.

For each dataset, the program:

1. Generates a synthetic RGB image.
2. Allocates host and device memory.
3. Runs the single-stream.
4. Verifies the rgb-to-grayscale result.
5. Runs the multi-stream.
6. Verifies the rgb-to-grayscale result.
7. Prints execution times and the calculated speedup.
8. Saves the images when image saving is enabled.

### Build

Run:

```bash
./build.sh
```

This will generate:

* `assignment.exe` — GPU version

### Run

```bash
./run.sh <numThreads> <threadsPerBlock> <imageWidth> <imageHeight> 
```

or:

```bash
./assignment.exe <numThreads> <threadsPerBlock> <imageWidth> <imageHeight> 
```

For example:

```bash
./assignment.exe 512 256 1024 1024
```


