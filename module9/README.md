# CUDA Image Sharpening Using cuFFT, Thrust, and NPP

## Overview

This assignment implements image sharpening using CUDA and GPU-accelerated libraries. The program loads an input image, converts it to grayscale, and transfers the image data to the GPU. It then uses the Fast Fourier Transform (FFT) to convert the image into the frequency domain, applies a high-pass sharpening filter to enhance high-frequency details, and performs an inverse FFT to reconstruct the sharpened image.

## Libraries
cuFFT is used for frequency-domain transformations, Thrust to parallelize the sharpening filter across the frequency spectrum, and NPP for image conversion and processing operations.


## Build

Run:

```bash
./build.sh
```

This will generate:

- `assignment.exe` — GPU version

## Run

```bash
./run.sh <input_image> <frequency_sigma> <sharpen_strength>
```

Or:

```bash
./assignment.exe <input_image> <frequency_sigma> <sharpen_strength>
```

For example:

```bash
./assignment.exe ships.bmp 0.50 30.50
```

## Parameters

- `input_image` — Path to the input image.
- `frequency_sigma` — Controls the frequency range affected by the high-pass filter.
- `sharpen_strength` — Controls the intensity of the sharpening effect.

## Output

The sharpened grayscale image is saved in the `output` directory. The output filename includes the input image name and the sharpening parameters used.