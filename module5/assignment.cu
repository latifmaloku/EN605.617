#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define CHANNELS 3

#define RED_WEIGHT 299
#define GREEN_WEIGHT 587
#define BLUE_WEIGHT 114
#define WEIGHT_DIVISOR 1000

#define DEFAULT_TOTAL_THREADS 262144
#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_THREADS_PER_BLOCK 1024

// Store the grayscale weights in CUDA constant memory.
__constant__ int grayscaleWeights[CHANNELS];

// Warm up the GPU before timing kernels.
__global__ void warmupKernel()
{
}



// Convert RGB pixels to grayscale using global memory.
__global__ void rgbToGrayGlobal(
    const unsigned char* inputRGB, unsigned char* outputGray, int pixelCount)
{
    // get thread inex
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Let each thread process additional pixels when needed.
    for (int pixelIndex = threadId; pixelIndex < pixelCount; pixelIndex += totalThreads)
    {
        int rgbIndex = pixelIndex * CHANNELS;

        // Read the RGB values directly from global memory.
        int r = inputRGB[rgbIndex];
        int g = inputRGB[rgbIndex + 1];
        int b = inputRGB[rgbIndex + 2];

        // Calculate grayscale using the weights in constant memory.
        int gray = (grayscaleWeights[0] * r + grayscaleWeights[1] *
             g + grayscaleWeights[2] * b) / WEIGHT_DIVISOR;

        // Write the grayscale pixel back to global memory.
        outputGray[pixelIndex] = (unsigned char)gray;
    }
}

// Convert RGB pixels to grayscale using shared memory.
__global__ void rgbToGray(
    const unsigned char* inputRGB, unsigned char* outputGray, int pixelCount)
{
    extern __shared__ unsigned char sharedRGB[];
    int localThread = threadIdx.x;
    int pixelsPerBlock = blockDim.x;
    int blockStart = blockIdx.x * pixelsPerBlock;
    int gridStride = gridDim.x * pixelsPerBlock;

    // Process additional tiles when the image has more pixels than threads.
    for (int tileStart = blockStart; tileStart < pixelCount; tileStart += gridStride)
    {
        int pixelIndex = tileStart + localThread;
        // Load each RGB pixel from global memory into shared memory.
        if (pixelIndex < pixelCount)
        {
            int rgbIndex = pixelIndex * CHANNELS;
            int sharedIndex = localThread * CHANNELS;
            sharedRGB[sharedIndex] = inputRGB[rgbIndex];
            sharedRGB[sharedIndex + 1] = inputRGB[rgbIndex + 1];
            sharedRGB[sharedIndex + 2] = inputRGB[rgbIndex + 2];
        }

        // Wait until all threads have loaded their RGB values.
        __syncthreads();
        // boundary check to so we don't read beyond the pixel count
        if (pixelIndex < pixelCount)
        {
            int sharedIndex = localThread * CHANNELS;
            // Read the RGB values from shared memory.
            int r = sharedRGB[sharedIndex];
            int g = sharedRGB[sharedIndex + 1];
            int b = sharedRGB[sharedIndex + 2];

            // Calculate grayscale using the weights in constant memory.
            int gray = (grayscaleWeights[0] * r + grayscaleWeights[1] * 
                g + grayscaleWeights[2] * b) / WEIGHT_DIVISOR;

            // Write the grayscale result to global memory.
            outputGray[pixelIndex] = (unsigned char)gray;
        }
        // Wait for all threads to finish
        __syncthreads();
    }
}

// Read command-line settings.
bool parseArguments(int argc, char* argv[], int& totalThreads, int& threadsPerBlock)
{
    totalThreads = DEFAULT_TOTAL_THREADS;
    threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;

    if (argc == 1)
    {
        return true;
    }

    if (argc != 3)
    {
        printf("Usage: %s [totalThreads] [threadsPerBlock]\n", argv[0]);
        printf("Example: %s 262144 256\n", argv[0]);
        return false;
    }

    totalThreads = atoi(argv[1]);
    threadsPerBlock = atoi(argv[2]);

    if (totalThreads <= 0 || threadsPerBlock <= 0)
    {
        printf("Invalid arguments.\n");
        return false;
    }

    if (totalThreads < 64)
    {
        printf("Total number of threads must be at least 64.\n");
        return false;
    }

    if (threadsPerBlock > MAX_THREADS_PER_BLOCK)
    {
        printf("Threads per block cannot exceed %d.\n", MAX_THREADS_PER_BLOCK);
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
    saveRGBImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
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



// Checks shared memory size is within limits for the device
bool checkSharedMemorySize(size_t sharedMemoryBytes)
{
    cudaDeviceProp deviceProperties;
    cudaError_t result = cudaGetDeviceProperties(&deviceProperties, 0);

    if (result != cudaSuccess)
    {
        printf("Failed to get CUDA device properties: %s\n", cudaGetErrorString(result));
        return false;
    }

    if (sharedMemoryBytes > deviceProperties.sharedMemPerBlock)
    {
        printf("Requested shared memory exceeds the device limit.\n");
        printf("Requested: %zu bytes\n", sharedMemoryBytes);
        printf("Available: %zu bytes\n", deviceProperties.sharedMemPerBlock);
        return false;
    }
    return true;
}

// Run the global-memory grayscale kernel and measure its time.
float runGlobalGrayscaleKernel(unsigned char* deviceRGB,
    unsigned char* deviceGray, int pixelCount, int totalThreads, int threadsPerBlock)
{
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nGlobal Memory Kernel Configuration\n");
    printf("---------------------------------\n");

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run warm-up kernel before starting the timing.
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("\nRunning global-memory grayscale kernel...\n");

    // Start timer before calling kernel.
    cudaEventRecord(start);

    // Run the grayscale kernel using global memory.
    rgbToGrayGlobal<<<blocks, threadsPerBlock>>>(deviceRGB, deviceGray, pixelCount);

    // Stop timer after everything finsihes
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

// Run the shared-memory grayscale kernel and measure its time.
float runSharedGrayscaleKernel(unsigned char* deviceRGB,
    unsigned char* deviceGray, int pixelCount, int totalThreads, int threadsPerBlock)
{
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemoryBytes = (size_t)threadsPerBlock * 
        CHANNELS * sizeof(unsigned char);

    printf("\nShared Memory Kernel Configuration\n");
    printf("----------------------------------\n");
    printf("Shared memory:        %zu bytes/block\n", sharedMemoryBytes);

    if (!checkSharedMemorySize(sharedMemoryBytes))
    {
        return -1.0f;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run the warm-up kernel before starting the timer.
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("\nRunning shared-memory grayscale kernel...\n");

    // Start timer.
    cudaEventRecord(start);


    // Run the kernel using shared memory for converting RGB to grayscale
    rgbToGray<<<blocks, threadsPerBlock, sharedMemoryBytes>>>
        (deviceRGB, deviceGray, pixelCount);

    //stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}


// Check the CUDA result against a CPU result.
void verifyGrayscale(const unsigned char* rgb, const unsigned char* gray, int pixelCount)
{
    for (int i = 0; i < pixelCount; i++)
    {
        int rgbIndex = i * CHANNELS;
        int r = rgb[rgbIndex];
        int g = rgb[rgbIndex + 1]; 
        int b = rgb[rgbIndex + 2];
        // Calculate expected grayscale value using the same formula as in the kernel
        int expected = (RED_WEIGHT * r + GREEN_WEIGHT * g 
            + BLUE_WEIGHT * b) / WEIGHT_DIVISOR;

        if (gray[i] != (unsigned char)expected)
        {
            printf("Verification FAILED at pixel %d\n", i);
            printf("Expected: %d\n", expected);
            printf("Actual:   %d\n", gray[i]);
            return;
        }
    }

    printf("\nVerification: PASSED\n");
    return;
}

// Print the timing results for both memory approaches
void printPerformance(float globalMemTimer, float sharedMemTimer)
{
    printf("\nPerformance Comparison\n");
    printf("----------------------\n");
    printf("Global memory:  %.5f ms\n", globalMemTimer);
    printf("Shared memory: %.5f ms\n", sharedMemTimer);
}

void allocateMem_AndGenerateImage(unsigned char** deviceRGB, unsigned char** deviceGray, 
    unsigned char** hostRGB, unsigned char** hostGray, 
    size_t rgbNumPixels, size_t grayNumPixels)
{
    *hostRGB = (unsigned char*)malloc(rgbNumPixels);
    *hostGray = (unsigned char*)malloc(grayNumPixels);
    cudaMalloc((void**)deviceRGB, rgbNumPixels);
    cudaMalloc((void**)deviceGray, grayNumPixels);
    // Generate synthetic RGB image and save it
    generateSyntheticImage(IMAGE_WIDTH, IMAGE_HEIGHT, *hostRGB);
}
// Free allocated memory.
void freeMemory(unsigned char* deviceRGB, unsigned char* deviceGray,
     unsigned char* hostRGB,
     unsigned char* hostGray)
{
    cudaFree(deviceRGB);
    cudaFree(deviceGray);
    free(hostRGB);
    free(hostGray);
}

void convertRGBToGrayscale(int totalThreads, int threadsPerBlock)
{
    int pixelCount = IMAGE_WIDTH * IMAGE_HEIGHT;
    size_t rgbNumPixels = (size_t)pixelCount * CHANNELS * sizeof(unsigned char);
    size_t grayNumPixels = (size_t)pixelCount * sizeof(unsigned char);
    unsigned char* deviceRGB;
    unsigned char* deviceGray;
    unsigned char* hostRGB;
    unsigned char* hostGray;
    // memory allocation and synthetic image generation
    allocateMem_AndGenerateImage(&deviceRGB, &deviceGray, 
        &hostRGB, &hostGray, rgbNumPixels, grayNumPixels);
    int hostWeights[CHANNELS] = {RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT};
    cudaMemcpyToSymbol(grayscaleWeights, hostWeights, sizeof(hostWeights));
    cudaMemcpy(deviceRGB, hostRGB, rgbNumPixels, cudaMemcpyHostToDevice);

    // Run the global-memory grayscale kernel, and verify results
    float globalMemTimer = runGlobalGrayscaleKernel(
        deviceRGB, 
        deviceGray, 
        pixelCount, 
        totalThreads, 
        threadsPerBlock);
    cudaMemcpy(hostGray, deviceGray, grayNumPixels, cudaMemcpyDeviceToHost);
    verifyGrayscale(hostRGB, hostGray, pixelCount);

    // Run the shared-memory grayscale kernel, and verify results
    float sharedMemTimer = runSharedGrayscaleKernel(deviceRGB,
         deviceGray, 
         pixelCount, 
         totalThreads, 
         threadsPerBlock);
    cudaMemcpy(hostGray, deviceGray, grayNumPixels, cudaMemcpyDeviceToHost);
    verifyGrayscale(hostRGB, hostGray, pixelCount);
    saveGrayscaleImage(hostGray, IMAGE_WIDTH, IMAGE_HEIGHT);
    freeMemory(deviceRGB, deviceGray, hostRGB, hostGray);

    // Print performance comparison
    printPerformance(globalMemTimer, sharedMemTimer);
}

// Main function to run the grayscale conversion and performance comparison
int main(int argc, char* argv[])
{
    int totalThreads;
    int threadsPerBlock;

    if (!parseArguments(argc, argv, totalThreads, threadsPerBlock))
    {
        return EXIT_FAILURE;
    }
    // Run the RGB to grayscale conversion and performance comparison
    convertRGBToGrayscale(totalThreads, threadsPerBlock);


    return EXIT_SUCCESS;
}