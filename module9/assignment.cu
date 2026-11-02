
#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>
#include <nppi.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>
#include <FreeImage.h>

#include <iostream>
#include <string>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

// Convert the input image to grayscale
FIBITMAP* convertToGrayscale(const std::string& inputPath)
{
    FREE_IMAGE_FORMAT format =
        FreeImage_GetFileType(inputPath.c_str(), 0);

    if (format == FIF_UNKNOWN)
        format = FreeImage_GetFIFFromFilename(inputPath.c_str());

    if (format == FIF_UNKNOWN) {
        printf("Unable to determine input image format.\n");
        return nullptr;
    }

    FIBITMAP* input = FreeImage_Load(format, inputPath.c_str());

    if (!input) {
        printf("Unable to load input image.\n");
        return nullptr;
    }

    FIBITMAP* gray = FreeImage_ConvertToGreyscale(input);
    FreeImage_Unload(input);

    if (!gray){
        printf("Unable to convert image to grayscale.\n");
}
    return gray;
}


// Save the final 8-bit grayscale image
bool saveGrayscaleBMP(const std::string& path,
                      const npp::ImageCPU_8u_C1& image)
{
    int width = static_cast<int>(image.width());
    int height = static_cast<int>(image.height());

    FIBITMAP* bitmap = FreeImage_Allocate(width, height, 8);

    if (!bitmap)
        return false;

    // Initialize the grayscale color palette
    RGBQUAD color;

    for (int i = 0; i < 256; ++i) {
        color.rgbRed = color.rgbGreen = color.rgbBlue = i;
        color.rgbReserved = 0;
        FreeImage_GetPalette(bitmap)[i] = color;
    }

    // Copy image data into the bitmap
    for (int y = 0; y < height; ++y) {
        BYTE* row = FreeImage_GetScanLine(bitmap, height - 1 - y);

        for (int x = 0; x < width; ++x)
            row[x] = image.data()[y * image.pitch() + x];
    }
    // Save to specified path
    bool saved = FreeImage_Save(
        FIF_BMP, bitmap, path.c_str(), BMP_DEFAULT);

    FreeImage_Unload(bitmap);
    return saved;
}


// Apply frequency-domain sharpening using Thrust
struct SharpenSpectrum
{
    // Device pointer to the frequency spectrum
    cufftComplex* spectrum;
    int width, height, spectrumWidth;
    float frequencySigma, sharpenStrength;

    // Operator to apply sharpening to each frequency component
    __device__
    void operator()(int index) const
    {
        // Calculate the (x, y) coordinates
        int x = index % spectrumWidth;
        int y = index / spectrumWidth;

        int distanceY = (y < height - y) ? y : height - y;
        // Calculate normalized frequency coordinates
        float fx = static_cast<float>(x) / width;
        float fy = static_cast<float>(distanceY) / height;
        float radiusSquared = fx * fx + fy * fy;
        // Compute the high-pass filter weight
        float highPass = 1.0f - expf(
            -radiusSquared / (2.0f * frequencySigma * frequencySigma));

        // Apply the sharpening weight to the spectrum
        float weight = 1.0f + sharpenStrength * highPass;
        spectrum[index].x *= weight;
        spectrum[index].y *= weight;
    }
};


// Manage FFT memory and plans
struct FFTResources
{
    // Device pointers for input, output, and spectrum
    Npp32f* input = nullptr;
    Npp32f* output = nullptr;
    cufftComplex* spectrum = nullptr;

    cufftHandle forwardPlan = 0;
    cufftHandle inversePlan = 0;

    bool forwardCreated = false;
    bool inverseCreated = false;

    // Destructor to clean up resources
    ~FFTResources()
    {
        if (forwardCreated) cufftDestroy(forwardPlan);
        if (inverseCreated) cufftDestroy(inversePlan);

        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (spectrum) cudaFree(spectrum);
    }
};


// Allocate device memory
bool allocateFFTMemory(FFTResources& resources,
                       int pixels, int spectrumValues)
{
    if (cudaMalloc(reinterpret_cast<void**>(&resources.input),
                   pixels * sizeof(Npp32f)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&resources.output),
                   pixels * sizeof(Npp32f)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&resources.spectrum),
                   spectrumValues * sizeof(cufftComplex)) != cudaSuccess) {
        printf("Unable to allocate FFT device memory.\n");
        return false;
    }

    return true;
}


// Create forward and inverse FFT plans
bool createFFTPlans(FFTResources& resources, int width, int height)
{
    // Create forward FFT plan real-to-complex
    if (cufftPlan2d(&resources.forwardPlan,
                    height, width, CUFFT_R2C) != CUFFT_SUCCESS) {
        printf("Unable to create forward FFT plan.\n");
        return false;
    }

    resources.forwardCreated = true;

    // Create inverse FFT plan complex-to-real
    if (cufftPlan2d(&resources.inversePlan,
                    height, width, CUFFT_C2R) != CUFFT_SUCCESS) {
        printf("Unable to create inverse FFT plan.\n");
        return false;
    }

    resources.inverseCreated = true;
    return true;
}


// Convert image data to floating point
bool convertToFloat(const npp::ImageNPP_8u_C1& source,
                    Npp32f* destination,
                    int floatPitch, NppiSize roi)
{
    // Convert 8-bit grayscale to 32-bit float
    return nppiConvert_8u32f_C1R(
        source.data(), source.pitch(),
        destination, floatPitch, roi) == NPP_SUCCESS;
}


// Perform the forward FFT
bool runForwardFFT(FFTResources& resources)
{
    // Convert to frequency domain
    return cufftExecR2C(
        resources.forwardPlan,
        resources.input,
        resources.spectrum) == CUFFT_SUCCESS;
}


// Apply frequency sharpening using Thrust
bool runSharpeningThrust(FFTResources& resources,
                         int width, int height,
                         float frequencySigma,
                         float sharpenStrength)
{
    int spectrumWidth = width / 2 + 1;
    int spectrumValues = spectrumWidth * height;

    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(spectrumValues),
        SharpenSpectrum{
            resources.spectrum, width, height, spectrumWidth,
            frequencySigma, sharpenStrength
        });

    return cudaGetLastError() == cudaSuccess;
}


// Perform the inverse FFT
bool runInverseFFT(FFTResources& resources)
{
    // Convert back to spatial domain
    return cufftExecC2R(
        resources.inversePlan,
        resources.spectrum,
        resources.output) == CUFFT_SUCCESS;
}


// Normalize and convert the result back to 8-bit
bool convertToGrayscale8u(Npp32f* input,
                          npp::ImageNPP_8u_C1& destination,
                          int floatPitch, int pixels,
                          NppiSize roi)
{
    // Normalize the floating-point result
    if (nppiMulC_32f_C1IR(
            1.0f / pixels, input, floatPitch, roi) != NPP_SUCCESS) {
        return false;
    }

    // Convert to 8-bit grayscale
    return nppiConvert_32f8u_C1R(
        input, floatPitch,
        destination.data(), destination.pitch(),
        roi, NPP_RND_NEAR) == NPP_SUCCESS;
}


// Perform FFT sharpening
bool executeFFT(const npp::ImageNPP_8u_C1& source,
                npp::ImageNPP_8u_C1& destination,
                int width, int height,
                float frequencySigma,
                float sharpenStrength)
{
    // Validate dimensions before calculating memory sizes
    if (width <= 0 || height <= 0 ||
        width > INT_MAX / height ||
        width > INT_MAX / static_cast<int>(sizeof(Npp32f))) {
        printf("Invalid image dimensions for FFT.\n");
        return false;
    }

    // Calculate memory sizes
    int pixels = width * height;
    int spectrumWidth = width / 2 + 1;
    if (spectrumWidth > INT_MAX / height) {
        printf("FFT spectrum dimensions are too large.\n");
        return false;
    }
    int spectrumValues = spectrumWidth * height;
    int floatPitch = width * sizeof(Npp32f);
    // Define the region of interest for NPP operations
    NppiSize roi = { width, height };
    FFTResources resources;

    // Allocate memory and prepare FFT input
    if (!allocateFFTMemory(resources, pixels, spectrumValues) ||
        !convertToFloat(source, resources.input, floatPitch, roi) ||
        !createFFTPlans(resources, width, height)) {
        printf("Unable to prepare FFT processing.\n");
        return false;
    }

    // Perform forward FFT
    if (!runForwardFFT(resources)) {
        printf("Forward FFT failed.\n");
        return false;
    }

    // Sharpen the frequency spectrum using Thrust
    if (!runSharpeningThrust(resources, width, height,
            frequencySigma, sharpenStrength)) {
        printf("Frequency sharpening failed.\n");
        return false;
    }

    // Perform inverse FFT
    if (!runInverseFFT(resources)) {
        printf("Inverse FFT failed.\n");
        return false;
    }

    // Normalize and convert the result
    if (!convertToGrayscale8u(resources.output, destination,
            floatPitch, pixels, roi)) {
        printf("Unable to convert FFT result.\n");
        return false;
    }

    return cudaDeviceSynchronize() == cudaSuccess;
}


// Process the image using cuFFT and Thrust
bool processImage(const npp::ImageCPU_8u_C1& hostInput,
                  const std::string& outputPath,
                  float frequencySigma,
                  float sharpenStrength)
{
    int width = static_cast<int>(hostInput.width());
    int height = static_cast<int>(hostInput.height());

    // Allocate GPU images
    npp::ImageNPP_8u_C1 input(hostInput);
    npp::ImageNPP_8u_C1 enhanced(width, height);

    // Apply frequency-domain sharpening
    if (!executeFFT(input, enhanced, width, height,
                    frequencySigma, sharpenStrength)) {
        return false;
    }

    // Copy the final image back to the CPU
    npp::ImageCPU_8u_C1 hostOutput(width, height);
    enhanced.copyTo(hostOutput.data(), hostOutput.pitch());

    // Save the sharpened image
    if (!saveGrayscaleBMP(outputPath, hostOutput)) {
        printf("Unable to save sharpened image.\n");
        return false;
    }

    printf("Image processing completed.\n"
           "Sharpened image saved to %s\n", outputPath.c_str());

    return true;
}


// Parse and validate command-line arguments
bool parseArgs(int argc, char* argv[],
               std::string& inputPath,
               float& frequencySigma,
               float& sharpenStrength)
{
    if (argc != 4) {
        printf("Usage: ./assignment.exe <input_image> "
               "<frequency_sigma> <sharpen_strength>\n");
        return false;
    }

    char* sigmaEnd;
    char* strengthEnd;

    inputPath = argv[1];
    frequencySigma = std::strtof(argv[2], &sigmaEnd);
    sharpenStrength = std::strtof(argv[3], &strengthEnd);

    // Validate parameters
    if (sigmaEnd == argv[2] || *sigmaEnd != '\0' ||
        strengthEnd == argv[3] || *strengthEnd != '\0' ||
        !std::isfinite(frequencySigma) || frequencySigma <= 0.0f ||
        !std::isfinite(sharpenStrength) || sharpenStrength < 0.0f) {
        printf("Invalid sharpening parameters.\n");
        return false;
    }

    return true;
}


// Load and process the input image
bool processInputImage(const std::string& inputPath,
                       float frequencySigma,
                       float sharpenStrength)
{
    // Create output directory and file path
    std::filesystem::create_directories("output");
    std::string name = std::filesystem::path(inputPath).stem().string();
    std::string outputPath =
    "output/" + name +
    "_sigma_" + std::to_string(frequencySigma) +
    "_strength_" + std::to_string(sharpenStrength) +
    "_sharpened.bmp";

    // Convert input image to grayscale
    std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> gray(
        convertToGrayscale(inputPath), FreeImage_Unload);

    if (!gray) {
        return false;
    }

    // Validate image dimensions
    unsigned int imageWidth = FreeImage_GetWidth(gray.get());
    unsigned int imageHeight = FreeImage_GetHeight(gray.get());
    if (imageWidth == 0 || imageHeight == 0 ||
        imageWidth > INT_MAX || imageHeight > INT_MAX) {
        printf("Invalid image dimensions.\n");
        return false;
    }

    int width = static_cast<int>(imageWidth);
    int height = static_cast<int>(imageHeight);

    // Copy grayscale pixels into the CPU image
    npp::ImageCPU_8u_C1 hostInput(width, height);
    for (int y = 0; y < height; ++y) {
        BYTE* row = FreeImage_GetScanLine(gray.get(), height - 1 - y);

        for (int x = 0; x < width; ++x)
            hostInput.data()[y * hostInput.pitch() + x] = row[x];
    }
    // Display processing parameters
    printf("Image dimensions: %d x %d\n", width, height);
    printf("Total pixels: %lld\n", static_cast<long long>(width) * height);
    printf("Frequency sigma: %f\n", frequencySigma);
    printf("Sharpen strength: %f\n", sharpenStrength);
    // Process the image and save the result
    return processImage(hostInput, outputPath, frequencySigma, sharpenStrength);
}


int main(int argc, char* argv[])
{
    std::string inputPath;
    float frequencySigma;
    float sharpenStrength;

    // Read args
    if (!parseArgs(argc, argv, inputPath, frequencySigma, sharpenStrength)){
        return EXIT_FAILURE;
    }
    FreeImage_Initialise();

    try {
        // Process input image
        if (!processInputImage(inputPath, frequencySigma, sharpenStrength)){
            throw std::runtime_error("Image processing failed.");
        }
        FreeImage_DeInitialise();
        return EXIT_SUCCESS;
    }
    catch (const npp::Exception& error) {
        printf("NPP image erro");
    }
    catch (const std::exception& error) {
        printf("Error");
    }

    FreeImage_DeInitialise();
    return EXIT_FAILURE;
}
