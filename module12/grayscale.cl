// Calculate integer luminance using the weights stored in constant memory.
inline uchar calculateGray(
    uchar3 rgb,
    __constant const uint* weights)
{
    uint total = rgb.x * weights[0] + rgb.y * weights[1] +
    rgb.z * weights[2];
    return (uchar)(total / 1000);
}

// Convert RGB pixels using direct global-memory reads.
__kernel void rgbToGrayGlobal(
    __global const uchar* inputRGB,
    __global uchar* outputGray,
    __constant const uint* weights,
    uint inputPixelOffset,
    uint outputPixelOffset,
    uint regionPixelCount)
{
    size_t firstPixel = get_global_id(0);
    size_t stride = get_global_size(0);
    // The grid-stride loop covers images larger than the NDRange.
    for (size_t pixel = firstPixel; pixel < regionPixelCount; pixel += stride)
    {
        uchar3 rgb = vload3(inputPixelOffset + pixel, inputRGB);
        outputGray[outputPixelOffset + pixel] =
            calculateGray(rgb, weights);
    }
}

// Convert RGB pixels using a work-group tile in local memory.
__kernel void rgbToGrayLocal(
    __global const uchar* inputRGB,
    __global uchar* outputGray,
    __constant const uint* weights,
    uint inputPixelOffset,
    uint outputPixelOffset,
    uint regionPixelCount,
    __local uchar* localRGB)
{
    size_t localId = get_local_id(0);
    size_t localSize = get_local_size(0);
    size_t blockStart = get_group_id(0) * localSize;
    size_t gridStride = get_global_size(0);

    for (size_t tileStart = blockStart; tileStart < regionPixelCount; tileStart += gridStride)
    {
        size_t pixel = tileStart + localId;
        // Each work-item copies one RGB into the local tile.
        if (pixel < regionPixelCount)
        {
            size_t rgbIndex = (inputPixelOffset + pixel) * 3;
            size_t localIndex = localId * 3;
            localRGB[localIndex] = inputRGB[rgbIndex];
            localRGB[localIndex + 1] = inputRGB[rgbIndex + 1];
            localRGB[localIndex + 2] = inputRGB[rgbIndex + 2];
        }

        // Tail work-items must also reach both barriers.
        barrier(CLK_LOCAL_MEM_FENCE);
        if (pixel < regionPixelCount)
        {
            uchar3 rgb = vload3(localId, localRGB);
            outputGray[outputPixelOffset + pixel] =
                calculateGray(rgb, weights);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
