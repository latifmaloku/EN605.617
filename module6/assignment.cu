#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3
#define RED_WEIGHT 299
#define GREEN_WEIGHT 587
#define BLUE_WEIGHT 114
#define WEIGHT_DIVISOR 1000
#define DEFAULT_TOTAL_THREADS 262144
#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_IMAGE_WIDTH 2048
#define DEFAULT_IMAGE_HEIGHT 2048
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_IMAGE_DIMENSION 1
#define STREAM_COUNT 2
#define DATASET_COUNT 3

__constant__ int grayscaleWeights[CHANNELS];

// Warm up the GPU before timing kernels.
__global__ void warmupKernel() {}

// Convert RGB pixels to grayscale using global memory.
__global__ void rgbToGray(const unsigned char* inputRGB,
    unsigned char* outputGray, int pixelCount)
{
    // get thread inex
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Let each thread process additional pixels when needed.
    for (int pixelIndex = threadId; pixelIndex < pixelCount;
         pixelIndex += totalThreads)
    {
        int rgbIndex = pixelIndex * CHANNELS;

        // Read the RGB values directly from global memory
        int r = inputRGB[rgbIndex];
        int g = inputRGB[rgbIndex + 1];
        int b = inputRGB[rgbIndex + 2];

        // Calculate grayscale using the weights in constant memory
        int gray = (grayscaleWeights[0] * r + grayscaleWeights[1] *
             g + grayscaleWeights[2] * b) / WEIGHT_DIVISOR;

        // Write the grayscale pixel back to global memory
        outputGray[pixelIndex] = (unsigned char)gray;
    }
}

// Read command-line settings: threads, block size, maximum width, maximum height.
bool parseArguments(int argc, char* argv[], int& totalThreads,
    int& threadsPerBlock, int& maxImageWidth, int& maxImageHeight)
{
    // Set default values for the parameters
    totalThreads = DEFAULT_TOTAL_THREADS;
    threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    maxImageWidth = DEFAULT_IMAGE_WIDTH;
    maxImageHeight = DEFAULT_IMAGE_HEIGHT;

    if (argc != 1 && argc != 3 && argc != 5)
    {
        printf("Usage: %s [totalThreads threadsPerBlock [maxImageWidth maxImageHeight]]\n",
            argv[0]);
        printf("Example: %s 262144 256 2048 2048\n", argv[0]);
        return false;
    }

    if (argc >= 3)
    {
        totalThreads = atoi(argv[1]);
        threadsPerBlock = atoi(argv[2]);
    }
    if (argc == 5)
    {
        maxImageWidth = atoi(argv[3]);
        maxImageHeight = atoi(argv[4]);
    }

    if (totalThreads < 64 || threadsPerBlock <= 0 ||
        threadsPerBlock > MAX_THREADS_PER_BLOCK ||
        maxImageWidth < MIN_IMAGE_DIMENSION || maxImageHeight < MIN_IMAGE_DIMENSION)
    {
        printf("Invalid arguments. Threads must be >= 64, block size 1-%d, "
            "and image dimensions must be positive.\n", MAX_THREADS_PER_BLOCK);
        return false;
    }
    return true;
}

// Save the RGB image.
void saveRGBImage(const unsigned char* image, int width, int height)
{
    const char* outputFile = "synthetic_rgb.png";
    int result = stbi_write_png(
        outputFile, width, height, CHANNELS, image, width * CHANNELS);

    if (result == 0)
    {
        printf("Failed to write RGB PNG output.\n");
    }
    else
    {
        printf("RGB PNG output: %s\n", outputFile);
    }
}

// Creates and saves synthetic RGB image.
void generateSyntheticImage(int width, int height, unsigned char* image)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = (y * width + x) * CHANNELS;

            image[index] = (unsigned char)((x + y) % 256);
            image[index + 1] = (unsigned char)((x * 2 + y) % 256);
            image[index + 2] = (unsigned char)((x + y * 2) % 256);
        }
    }
}

// Save the grayscale image.
void saveGrayscaleImage(const unsigned char* image, int width, int height)
{
    const char* outputFile = "grayscale_output.png";
    int result = stbi_write_png(outputFile, width, height, 1, image, width);

    if (result == 0)
    {
        printf("\nFailed to write PNG output.\n");
    }
    else
    {
        printf("\nPNG output: %s\n", outputFile);
    }
}

// Free allocated memory.
void freeMemory(unsigned char* deviceRGB, unsigned char* deviceGray,
    unsigned char* hostRGB, unsigned char* hostGray)
{
    if (deviceRGB) cudaFree(deviceRGB);
    if (deviceGray) cudaFree(deviceGray);
    if (hostRGB) cudaFreeHost(hostRGB);
    if (hostGray) cudaFreeHost(hostGray);
}

// Allocate pinned host memory and device memory.
void allocateMemory(unsigned char** deviceRGB, unsigned char** deviceGray,
    unsigned char** hostRGB, unsigned char** hostGray,
    size_t rgbBytes, size_t grayBytes)
{
    cudaMallocHost((void**)hostRGB, rgbBytes);
    cudaMallocHost((void**)hostGray, grayBytes);
    cudaMalloc((void**)deviceRGB, rgbBytes);
    cudaMalloc((void**)deviceGray, grayBytes);
    
}

// Calculate block count for a dataset or chunk.
int calculateBlocks(int totalThreads, int threadsPerBlock, int pixelCount)
{
    int requestedBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    int neededBlocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    return requestedBlocks < neededBlocks ? requestedBlocks : neededBlocks;
}

// Run the grayscale in one stream and measure elapsed time.
float runSingleStream(unsigned char* hostRGB, unsigned char* hostGray,
    unsigned char* deviceRGB, unsigned char* deviceGray, int pixelCount,
    size_t rgbBytes, size_t grayBytes, int totalThreads, int threadsPerBlock)
{
    int blocks = calculateBlocks(totalThreads, threadsPerBlock, pixelCount);
    cudaStream_t stream;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    // Create a single stream and events for timing
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Start timing, and run the kernel in a single stream
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(deviceRGB, hostRGB, rgbBytes, cudaMemcpyHostToDevice, stream);
    rgbToGray<<<blocks, threadsPerBlock, 0, stream>>>(
        deviceRGB, deviceGray, pixelCount);
    
    // Copy the grayscale result back to the host
    cudaMemcpyAsync(hostGray, deviceGray, grayBytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Clean up resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return elapsedTime;
}
// Process a chunk in a specific stream.
void processStreamChunk(unsigned char* hostRGB, unsigned char* hostGray,
    unsigned char* deviceRGB, unsigned char* deviceGray, int pixelCount,
    int pixelsPerStream, int streamIndex, int totalThreads,
    int threadsPerBlock, cudaStream_t stream, cudaEvent_t streamDone)
{
    // Calculate the starting pixel index for this stream
    int startPixel = streamIndex * pixelsPerStream;
    if (startPixel >= pixelCount)
    {
        cudaEventRecord(streamDone, stream);
        return;
    }
    
    int chunk = min(pixelsPerStream, pixelCount - startPixel);

    // Calculate offsets and sizes for the RGB and grayscale data
    size_t rgbOffset = (size_t)startPixel * CHANNELS;
    size_t rgbChunkBytes = (size_t)chunk * CHANNELS;
    size_t grayOffset = (size_t)startPixel;
    size_t grayChunkBytes = (size_t)chunk;
    int blocks = calculateBlocks(totalThreads, threadsPerBlock, chunk);

    // Copy the RGB chunk to the device, process it, and copy grayscale back to the host
    cudaMemcpyAsync(deviceRGB + rgbOffset, hostRGB + rgbOffset,
        rgbChunkBytes, cudaMemcpyHostToDevice, stream);
    // Launch the kernel for this chunk
    rgbToGray<<<blocks, threadsPerBlock, 0, stream>>>(
        deviceRGB + rgbOffset, deviceGray + grayOffset, chunk);

    cudaMemcpyAsync(hostGray + grayOffset, deviceGray + grayOffset,
        grayChunkBytes, cudaMemcpyDeviceToHost, stream);

    // Record an event to signal that this stream's processing is done
    cudaEventRecord(streamDone, stream);
}

// Clean up streams and events after processing
void cleanupMultiStream(cudaStream_t streams[STREAM_COUNT],
    cudaStream_t timingStream, cudaEvent_t start, cudaEvent_t stop,
    cudaEvent_t streamDone[STREAM_COUNT])
{
    // Destroy events and streams
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaEventDestroy(streamDone[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(timingStream);
}

// Run separate image chunks in multiple streams and measure elapsed time.
float runMultiStream(unsigned char* hostRGB, unsigned char* hostGray,
    unsigned char* deviceRGB, unsigned char* deviceGray, int pixelCount,
    int totalThreads, int threadsPerBlock)
{
    // Create multiple streams and events for timing.
    cudaStream_t streams[STREAM_COUNT];
    cudaStream_t timingStream;
    cudaEvent_t start, stop, streamDone[STREAM_COUNT];
    float elapsedTime = 0.0f;
    int pixelsPerStream = (pixelCount + STREAM_COUNT - 1) / STREAM_COUNT;
    cudaStreamCreate(&timingStream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Create worker streams and events for each stream
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&streamDone[i]);
    }

    // Warm up the GPU before timing kernels.
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Start timing and make each worker stream wait for the start event.
    cudaEventRecord(start, timingStream);
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaStreamWaitEvent(streams[i], start, 0);
    }

    // Process image chunks in multiple streams.
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        processStreamChunk(hostRGB, hostGray, deviceRGB, deviceGray,
            pixelCount, pixelsPerStream, i, totalThreads, threadsPerBlock,
            streams[i], streamDone[i]);
    }

    // Wait for all worker streams before stopping the timer.
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaStreamWaitEvent(timingStream, streamDone[i], 0);
    }
    cudaEventRecord(stop, timingStream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Clean up streams and events.
    cleanupMultiStream(streams, timingStream, start, stop, streamDone);
    return elapsedTime;
}

// Verify every grayscale pixel against the CPU calculation.
bool verifyGrayscale(const unsigned char* rgb, const unsigned char* gray,
    int pixelCount)
{
    for (int i = 0; i < pixelCount; i++)
    {
        int index = i * CHANNELS;
        int expected = (RED_WEIGHT * rgb[index] + GREEN_WEIGHT * rgb[index + 1] +
            BLUE_WEIGHT * rgb[index + 2]) / WEIGHT_DIVISOR;
        if (gray[i] != (unsigned char)expected)
        {
            printf("Verification FAILED at pixel %d (expected %d, got %d)\n",
                i, expected, gray[i]);
            return false;
        }
    }
    printf("Verification: PASSED\n");
    return true;
}

// Run both single-stream and multi-stram for one dataset size and print the results.
bool runDataset(int width, int height, int totalThreads, int threadsPerBlock,
    bool saveImages)
{
    // Retrieve device properties and available memory
    int deviceId;
    cudaDeviceProp properties;
    size_t freeBytes, totalBytes;
    if ((cudaGetDevice(&deviceId) != cudaSuccess) ||
        (cudaGetDeviceProperties(&properties, deviceId) != cudaSuccess) ||
        (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess))
    {
        printf("Failed to retrieve CUDA device properties.\n");
        return false;
    }

    // Calculate pixel count, required memory, and required blocks
    size_t pixelCount = (size_t)width * (size_t)height;
    size_t rgbBytes = pixelCount * CHANNELS;
    size_t grayBytes = pixelCount;
    size_t requiredBlocks = (pixelCount + threadsPerBlock - 1) /
        threadsPerBlock;
    // Check if the required number of blocks exceeds the maximum grid size
    if (requiredBlocks > properties.maxGridSize[0])
    {
        return false;
    }
    // Check if the required memory exceeds the available device memory
    unsigned char *deviceRGB = NULL, *deviceGray = NULL;
    unsigned char *hostRGB = NULL, *hostGray = NULL;
    printf("Image-Size %d x %d (%zu pixels)\n", width, height, pixelCount);
    allocateMemory(&deviceRGB, &deviceGray, &hostRGB, &hostGray,
        rgbBytes, grayBytes);
    // Generate a synthetic RGB image for testing
    generateSyntheticImage(width, height, hostRGB);
    
    // Run single-stream and multi-stream processing, measure time, and verify results
    float singleStrmTime = runSingleStream(hostRGB, hostGray, deviceRGB, deviceGray,
        pixelCount, rgbBytes, grayBytes, totalThreads, threadsPerBlock);
    bool singlePassed = verifyGrayscale(hostRGB, hostGray, pixelCount);
    float multiStrmTime = runMultiStream(hostRGB, hostGray, deviceRGB, deviceGray,
        pixelCount, totalThreads, threadsPerBlock);
    bool multiPassed = verifyGrayscale(hostRGB, hostGray, pixelCount);

    printf("Single-stream time: %.5f ms\n", singleStrmTime);
    printf("Multi-stream time:  %.5f ms\n\n", multiStrmTime);

    if (saveImages)
    {
        saveRGBImage(hostRGB, width, height);
        saveGrayscaleImage(hostGray, width, height);
    }

    freeMemory(deviceRGB, deviceGray, hostRGB, hostGray);
    return singlePassed && multiPassed;
}

// Test 512x512, 1024x1024, and the user-configured image dimensions.
void runTests(int totalThreads, int threadsPerBlock,
    int maxImageWidth, int maxImageHeight)
{
    // Define the dataset sizes to test, including user-specified image dimensions
    const int testWidths[DATASET_COUNT] = {512, 1024, maxImageWidth};
    const int testHeights[DATASET_COUNT] = {512, 1024, maxImageHeight};
    int completed = 0;
    bool allPassed = true;

    printf("Threads: %d | Threads per block: %d\n", totalThreads, threadsPerBlock);

    // Run tests for each dataset size and print the results
    for (int i = 0; i < DATASET_COUNT; i++)
    {
        // Save images only for the user input size
        bool saveImages = (testWidths[i] == maxImageWidth && testHeights[i] == maxImageHeight);
        if (!runDataset(testWidths[i], testHeights[i], totalThreads, threadsPerBlock, saveImages))
        {
            allPassed = false;
        }
        completed++;
    }

    printf("\nRun completed %d dataset size(s).\n", completed);
    printf("Overall verification: %s\n", allPassed ? "PASSED" : "FAILED");
}

int main(int argc, char* argv[])
{
    int totalThreads, threadsPerBlock, maxImageWidth, maxImageHeight;
    // Parse CL arguments and validate them
    if (!parseArguments(argc, argv, totalThreads, threadsPerBlock,
        maxImageWidth, maxImageHeight)){
            return EXIT_FAILURE;
        }

    // Copy the grayscale weights to constant memory
    int hostWeights[CHANNELS] = {RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT};
    cudaMemcpyToSymbol(grayscaleWeights, hostWeights,
        sizeof(hostWeights));

    // Run the tests for multiple dataset sizes and print the results
    runTests(totalThreads, threadsPerBlock, maxImageWidth, maxImageHeight);
    return EXIT_SUCCESS;
}
