#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Integer weights keep the host and OpenCL grayscale calculations aligned
const cl_uint RED_WEIGHT = 299;
const cl_uint GREEN_WEIGHT = 587;
const cl_uint BLUE_WEIGHT = 114;
const cl_uint WEIGHT_DIVISOR = 1000;
const cl_uint CHANNELS = 3;
// Runtime settings for work size and generated image dimensions
struct Options
{
    cl_uint totalWorkItems = 262144;
    size_t localSize = 256;
    cl_uint width = 1024;
    cl_uint height = 1024;
};

// OpenCL handles are shared by setup, execution, and cleanup helpers
cl_platform_id platform = nullptr;
cl_device_id device = nullptr;
cl_context context = nullptr;
cl_command_queue queue = nullptr;
cl_program program = nullptr;
cl_kernel globalKernel = nullptr;
cl_kernel localKernel = nullptr;
cl_mem inputBuffer = nullptr;
cl_mem inputView = nullptr;
cl_mem globalOutput = nullptr;
cl_mem localOutput = nullptr;
cl_mem weightsBuffer = nullptr;

// Read command-line settings and validate image dimensions
bool parseArguments(int argc, char* argv[], Options& options)
{
    if (argc > 5)
    {
        return false;
    }

    int values[4] = {
        static_cast<int>(options.totalWorkItems),
        static_cast<int>(options.localSize),
        static_cast<int>(options.width),
        static_cast<int>(options.height)
    };

    // Keep defaults for omitted arguments and replace supplied values
    for (int i = 1; i < argc; ++i)
    {
        values[i - 1] = std::atoi(argv[i]);
        if (values[i - 1] <= 0)
        {
            return false;
        }
    }

    // Store work and image settings in the types used by OpenCL
    options.totalWorkItems = static_cast<cl_uint>(values[0]);
    options.localSize = static_cast<size_t>(values[1]);
    options.width = static_cast<cl_uint>(values[2]);
    options.height = static_cast<cl_uint>(values[3]);
    // Preserve a valid PNG row stride and prevent pixel-count overflow
    if (options.width > static_cast<cl_uint>(
            std::numeric_limits<int>::max() / 3))
    {
        return false;
    }

    size_t maxHeight =
        std::numeric_limits<cl_uint>::max() / options.width;
    return options.height <= maxHeight;
}

// Select the first available OpenCL platform
bool selectPlatform()
{
    // Get the number of available platforms
    cl_uint count = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &count);
    if (status != CL_SUCCESS || count == 0)
    {
        std::cerr << "No OpenCL platforms found.\n";
        return false;
    }

    // Get platform IDs and select the first one
    std::vector<cl_platform_id> platforms(count);
    status = clGetPlatformIDs(count, platforms.data(), nullptr);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to enumerate OpenCL platforms.\n";
        return false;
    }

    platform = platforms[0];
    return true;
}

// Create a GPU context, falling back to a CPU context if needed
bool createContext()
{
    // Associate this context with the selected platform
    cl_int status = CL_SUCCESS;
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform),
        0
    };
    // Try to create a GPU context first, then fall back to CPU if needed
    context = clCreateContextFromType(
        properties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status);

    if (status != CL_SUCCESS)
    {
        std::cout << "No GPU context; trying the CPU.\n";
        context = clCreateContextFromType(
            properties, CL_DEVICE_TYPE_CPU, nullptr, nullptr, &status);
    }

    if (status != CL_SUCCESS || context == nullptr)
    {
        std::cerr << "Unable to create an OpenCL context.\n";
        return false;
    }
    return true;
}

// Select a context device and create a command queue
bool createQueue()
{
    // Query the context for the size of  device list
    size_t bytes = 0;
    cl_int status = clGetContextInfo(
        context, CL_CONTEXT_DEVICES, 0, nullptr, &bytes);
    if (status != CL_SUCCESS || bytes == 0)
    {
        std::cerr << "Unable to inspect context devices.\n";
        return false;
    }
    // Retrieve context devices and select the first one
    std::vector<cl_device_id> devices(bytes / sizeof(cl_device_id));
    status = clGetContextInfo(
        context, CL_CONTEXT_DEVICES, bytes, devices.data(), nullptr);
    if (status != CL_SUCCESS || devices.empty())
    {
        std::cerr << "Unable to get context devices.\n";
        return false;
    }

    device = devices[0];
    char name[256] = "";
    status = clGetDeviceInfo(
        device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to get the device name.\n";
        return false;
    }
    std::cout << "OpenCL device: " << name << '\n';

    // Create the queue used for kernel launches and data transfers
    queue = clCreateCommandQueue(
        context, device, 0, &status);
    if (status != CL_SUCCESS || queue == nullptr)
    {
        std::cerr << "Unable to create the command queue.\n";
        return false;
    }
    return true;
}


// Load and compile the OpenCL kernel source file
bool buildProgram(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Unable to open kernel source: "
                  << filename << '\n';
        return false;
    }

    // Read the kernel source before creating the OpenCL program
    std::ostringstream sourceStream;
    sourceStream << file.rdbuf();
    std::string source = sourceStream.str();
    const char* sourceText = source.c_str();
    cl_int status = CL_SUCCESS;
    program = clCreateProgramWithSource(
        context, 1, &sourceText, nullptr, &status);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to create the OpenCL program.\n";
        return false;
    }

    // Compile the source for the selected device
    status = clBuildProgram(
        program, 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS)
    {
        return false;
    }
    return true;
}

// Create the global-memory and local-memory grayscale kernels
bool createKernels()
{
    cl_int status = CL_SUCCESS;
    // Create one kernel for each memory-access approach
    globalKernel = clCreateKernel(program, "rgbToGrayGlobal", &status);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to create the global-memory kernel.\n";
        return false;
    }

    localKernel = clCreateKernel(program, "rgbToGrayLocal", &status);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to create the local-memory kernel.\n";
        return false;
    }
    return true;
}

// Check work-group and local-memory requirements before launching
bool validateLaunch(const Options& options)
{
    // Query per-kernel work-group limits and available local memory
    size_t globalLimit = 0;
    size_t localLimit = 0;
    cl_ulong localMemoryBytes = 0;
    cl_int status = clGetKernelWorkGroupInfo(
        globalKernel, device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(globalLimit), &globalLimit, nullptr);
    if (status == CL_SUCCESS)
    {
        status = clGetKernelWorkGroupInfo(
            localKernel, device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(localLimit), &localLimit, nullptr);
    }
    if (status == CL_SUCCESS)
    {
        status = clGetDeviceInfo(
            device, CL_DEVICE_LOCAL_MEM_SIZE,
            sizeof(localMemoryBytes), &localMemoryBytes, nullptr);
    }
    // Check that the selected group and its RGB tile fit the device
    size_t requestedLocalMemory =
        options.localSize * CHANNELS * sizeof(cl_uchar);
    if (status != CL_SUCCESS ||
        options.localSize > globalLimit ||
        options.localSize > localLimit ||
        requestedLocalMemory > localMemoryBytes)
    {
        std::cerr << "Requested work-group exceeds device limits.\n";
        return false;
    }
    return true;
}

// Save the RGB image.
bool saveRGBImage(
    const cl_uchar* image,
    cl_uint width,
    cl_uint height)
{
    const char* outputFile = "synthetic_rgb.png";
    int result = stbi_write_png(
        outputFile, static_cast<int>(width), static_cast<int>(height),
        3, image, static_cast<int>(width) * CHANNELS);
    if (result == 0)
    {
        std::cerr << "Failed to write RGB PNG output.\n";
        return false;
    }
    std::cout << "RGB PNG output: " << outputFile << '\n';
    return true;
}

// Generate and save the synthetic RGB image.
bool generateSyntheticImage(
    cl_uint width,
    cl_uint height,
    cl_uchar* image)
{
    // Fill each pixel with a repeatable RGB pattern.
    for (cl_uint y = 0; y < height; ++y)
    {
        for (cl_uint x = 0; x < width; ++x)
        {
            size_t index = ((size_t)y * width + x) * CHANNELS;
            image[index] = static_cast<cl_uchar>((x + y) % 256);
            image[index + 1] =
                static_cast<cl_uchar>((x * 2 + y) % 256);
            image[index + 2] =
                static_cast<cl_uchar>((x + y * 2) % 256);
        }
    }
    return saveRGBImage(image, width, height);
}

// Save one grayscale image.
bool saveGrayscaleImage(
    const char* outputFile,
    const cl_uchar* image,
    cl_uint width,
    cl_uint height)
{
    int result = stbi_write_png(
        outputFile, static_cast<int>(width), static_cast<int>(height),
        1, image, static_cast<int>(width));
    if (result == 0)
    {
        std::cerr << "Failed to write grayscale PNG output.\n";
        return false;
    }
    std::cout << "Grayscale PNG output: " << outputFile << '\n';
    return true;
}

// Allocate an OpenCL buffer and report its purpose if allocation fails
bool createBuffer(
    cl_mem& buffer,
    cl_mem_flags flags,
    size_t bytes,
    void* hostData,
    const char* description)
{
    cl_int status = CL_SUCCESS;
    // Create the device allocation, optionally copying initial host data
    buffer = clCreateBuffer(
        context, flags, bytes, hostData, &status);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to create " << description
                  << " buffer.\n";
        return false;
    }
    return true;
}

// Allocate the RGB buffer and create a sub-buffer for its first half
bool createInputBuffers(
    cl_uchar* rgb,
    size_t rgbBytes,
    size_t subBufferPixelCount)
{
    size_t subBufferBytes =
        subBufferPixelCount * CHANNELS * sizeof(cl_uchar);
    // Copy the complete source image into a read-only device buffer
    if (!createBuffer(
            inputBuffer, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            rgbBytes, rgb, "RGB input"))
    {
        return false;
    }

    // This view covers only the prefix; later work uses the parent buffer
    cl_buffer_region region = {0, subBufferBytes};
    cl_int status = CL_SUCCESS;
    inputView = clCreateSubBuffer(
        inputBuffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to create the RGB sub-buffer.\n";
        return false;
    }
    return true;
}

// Allocate output buffers and upload the grayscale weights
bool createOutputBuffers(size_t pixelCount)
{
    size_t grayBytes = pixelCount * sizeof(cl_uchar);
    // Allocate one output buffer for each kernel
    if (!createBuffer(
            globalOutput, CL_MEM_WRITE_ONLY, grayBytes,
            nullptr, "global output") ||
        !createBuffer(
            localOutput, CL_MEM_WRITE_ONLY, grayBytes,
            nullptr, "local output"))
    {
        return false;
    }

    // Upload the grayscale coefficients for the kernels to read
    cl_uint weights[3] = {
        RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT
    };
    if (!createBuffer(
            weightsBuffer,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(weights), weights, "grayscale weights"))
    {
        return false;
    }
    return true;
}

// Bind image arguments and optional local scratch memory to a kernel
bool setKernelArguments(
    cl_kernel target,
    cl_mem input,
    cl_uint inputPixelOffset,
    cl_uint outputPixelOffset,
    cl_uint regionPixelCount,
    size_t localSize)
{
    cl_mem output = nullptr;
    if (target == globalKernel)
    {
        output = globalOutput;
    }
    else
    {
        output = localOutput;
    }
    // Bind image buffers, coefficients, offsets, and region size
    cl_int status = clSetKernelArg(target, 0, sizeof(cl_mem), &input);
    status |= clSetKernelArg(target, 1, sizeof(cl_mem), &output);
    status |= clSetKernelArg(target, 2, sizeof(cl_mem), &weightsBuffer);
    status |= clSetKernelArg(target, 3, sizeof(cl_uint), &inputPixelOffset);
    status |= clSetKernelArg(target, 4, sizeof(cl_uint), &outputPixelOffset);
    status |= clSetKernelArg(target, 5, sizeof(cl_uint), &regionPixelCount);
    // Only the local-memory kernel needs a local scratch buffer
    if (target == localKernel)
    {
        status |= clSetKernelArg(
            target, 6, localSize * CHANNELS * sizeof(cl_uchar), nullptr);
    }
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to set kernel arguments.\n";
        return false;
    }
    return true;
}

// Enqueue a kernel and wait for it to finish
bool enqueueKernel(
    cl_kernel target,
    size_t globalSize,
    size_t localSize)
{
    // Submit kernel for execution on the device
    cl_int status = clEnqueueNDRangeKernel(queue, target, 1, nullptr, &globalSize,
        &localSize, 0, nullptr, nullptr);
    if (status == CL_SUCCESS)
    {
        // finish before the same kernel receives arguments for another region
        status = clFinish(queue);
    }

    if (status != CL_SUCCESS)
    {
        std::cerr << "Kernel launch or execution failed: "
                  << status << '\n';
        return false;
    }
    return true;
}

// Read one grayscale output buffer back to host memory
bool readGrayBuffer(
    cl_mem buffer,
    cl_uchar* gray,
    size_t grayBytes)
{
    // A blocking read transfers the completed device output to host memory
    cl_int status = clEnqueueReadBuffer(
        queue, buffer, CL_TRUE, 0, grayBytes,
        gray, 0, nullptr, nullptr);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Unable to read a grayscale output buffer.\n";
        return false;
    }
    return true;
}

// Calculate the CPU reference value for one RGB pixel
cl_uchar calculateGrayscale(const cl_uchar* pixel)
{
    // Apply integer luminance weights to one RGB pixel
    cl_uint gray =
        RED_WEIGHT * pixel[0] +
        GREEN_WEIGHT * pixel[1] +
        BLUE_WEIGHT * pixel[2];
    return static_cast<cl_uchar>(gray / WEIGHT_DIVISOR);
}

// Compare all pixels against the CPU grayscale reference
bool verifyGrayscale(
    const cl_uchar* rgb,
    const cl_uchar* gray,
    size_t pixelCount)
{
    // Compare every device result with the CPU reference
    for (size_t pixel = 0; pixel < pixelCount; ++pixel)
    {
        cl_uchar expected = calculateGrayscale(&rgb[pixel * CHANNELS]);
        if (gray[pixel] != expected)
        {
            return false;
        }
    }
    return true;
}

// Round the global size up to a complete number of work-groups
size_t calculateGlobalSize(const Options& options)
{
    // Round up so the global size contains complete work-groups
    size_t groups =
        (options.totalWorkItems + options.localSize - 1) /
        options.localSize;
    return groups * options.localSize;
}

// Read, verify, and save both grayscale results
bool readVerifyAndSave(
    const Options& options,
    const cl_uchar* rgb,
    size_t pixelCount)
{
    // Read both output buffers before verifying or saving either result
    std::vector<cl_uchar> globalGray(pixelCount);
    std::vector<cl_uchar> localGray(pixelCount);
    size_t grayBytes = pixelCount * sizeof(cl_uchar);
    if (!readGrayBuffer(globalOutput, globalGray.data(), grayBytes) ||
        !readGrayBuffer(localOutput, localGray.data(), grayBytes))
    {
        return false;
    }

    // Save files only after both GPU outputs match the CPU reference
    if (!verifyGrayscale(rgb, globalGray.data(), pixelCount) ||
        !verifyGrayscale(rgb, localGray.data(), pixelCount))
    {
        return false;
    }
    return saveGrayscaleImage(
               "global_gray.png", globalGray.data(),
               options.width, options.height) &&
           saveGrayscaleImage(
               "local_gray.png", localGray.data(),
               options.width, options.height);
}

// Release every OpenCL object created by this program
void cleanup()
{
    // Release memory objects before the program, queue, and context
    if (weightsBuffer)
    {
        clReleaseMemObject(weightsBuffer);
    }
    if (localOutput)
    {
        clReleaseMemObject(localOutput);
    }
    if (globalOutput)
    {
        clReleaseMemObject(globalOutput);
    }
    if (inputView)
    {
        clReleaseMemObject(inputView);
    }
    if (inputBuffer)
    {
        clReleaseMemObject(inputBuffer);
    }
    if (localKernel)
    {
        clReleaseKernel(localKernel);
    }
    if (globalKernel)
    {
        clReleaseKernel(globalKernel);
    }
    if (program)
    {
        clReleaseProgram(program);
    }
    if (queue)
    {
        clReleaseCommandQueue(queue);
    }
    if (context)
    {
        clReleaseContext(context);
    }
}

// Run the global-memory or local-memory kernel on both image regions
bool runKernelRegions(
    cl_kernel target,
    cl_mem firstInput,
    cl_uint firstPixelCount,
    cl_uint secondPixelCount,
    const Options& options)
{
    size_t globalSize = calculateGlobalSize(options);
    cl_uint zero = 0;
    // Run the first image region through the zero-offset sub-buffer
    if (!setKernelArguments(
            target, firstInput, zero, zero, firstPixelCount,
            options.localSize))
    {
        return false;
    }

    if (!enqueueKernel(target, globalSize, options.localSize))
    {
        return false;
    }

    if (secondPixelCount == 0)
    {
        return true;
    }

    // Run remaining pixels through the parent buffer at their image offset
    cl_uint secondOffset = firstPixelCount;
    if (!setKernelArguments(
            target, inputBuffer, secondOffset, secondOffset,
            secondPixelCount, options.localSize))
    {
        return false;
    }

    return enqueueKernel(target, globalSize, options.localSize);
}

// Run both grayscale kernels
bool runKernels(
    cl_uint firstPixelCount,
    cl_uint secondPixelCount,
    const Options& options)
{
    // Process both memory approaches using the same two image regions
    std::cout << "Running global-memory kernel...\n";
    std::cout.flush();
    if (!runKernelRegions(
            globalKernel, inputView, firstPixelCount,
            secondPixelCount, options))
    {
        return false;
    }

    std::cout << "Running local-memory kernel...\n";
    std::cout.flush();
    if (!runKernelRegions(
        localKernel, inputView, firstPixelCount,
        secondPixelCount, options))
    {
        return false;
    }
    return true;
}

// Generate the image, run both conversions, and report the results
bool convertRGBToGrayscale(const Options& options)
{
    // Generate the source image and save it before transferring it
    size_t pixelCount = (size_t)options.width * options.height;
    std::cout << "Generating synthetic RGB image...\n";
    std::cout.flush();
    std::vector<cl_uchar> rgb(pixelCount * CHANNELS);
    if (!generateSyntheticImage(
            options.width, options.height, rgb.data()))
    {
        return false;
    }

    // Divide pixels so the prefix uses a sub-buffer and the rest the parent
    cl_uint firstPixelCount =
        static_cast<cl_uint>((pixelCount + 1) / 2);
    cl_uint secondPixelCount =
        static_cast<cl_uint>(pixelCount - firstPixelCount);
    std::cout << "Sub-buffer region: first " << firstPixelCount
              << " pixels; parent-buffer region: "
              << secondPixelCount << " pixels.\n";
    std::cout << "Creating OpenCL buffers...\n";
    std::cout.flush();
    // Allocate OpenCL storage, run both kernels, then verify the results
    if (!createInputBuffers(
            rgb.data(), rgb.size() * sizeof(cl_uchar),
            firstPixelCount) ||
        !createOutputBuffers(pixelCount))
    {
        return false;
    }

    if (!runKernels(firstPixelCount, secondPixelCount, options))
    {
        return false;
    }

    if (!readVerifyAndSave(options, rgb.data(), pixelCount))
    {
        return false;
    }

    return true;
}

// Validate settings, initialize OpenCL, and run the conversion
int main(int argc, char* argv[])
{
    Options options;
    // Read optional run settings before initializing the OpenCL objects
    if (!parseArguments(argc, argv, options))
    {
        std::cerr
            << "Usage: assignment [total_work_items local_size "
            << "width height]\n";
        return 1;
    }

    // Select a device, build its kernels, and validate the launch settings
    if (!selectPlatform() || !createContext() || !createQueue() ||
        !buildProgram("grayscale.cl") || !createKernels() ||
        !validateLaunch(options))
    {
        cleanup();
        return 1;
    }

    // Complete the image work before releasing OpenCL resources
    bool success = convertRGBToGrayscale(options);
    cleanup();
    if (success)
    {
        return 0;
    }
    return 1;
}
