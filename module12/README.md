# OpenCL RGB-to-Grayscale Processing

## Overview

This project demonstrates RGB-to-grayscale image processing using OpenCL. It generates a synthetic RGB image and processes it on the GPU using two different kernels. The first kernel reads pixel data directly from global memory, while the second uses local memory to temporarily store RGB data within each work-group before processing it.

Both kernels use the same grayscale formula and their results are compared against a CPU reference implementation to verify correctness. The program also demonstrates device setup, memory management, kernel execution, work-group synchronization, and sub-buffer usage.

The program generates a 1024 x 1024 RGB image by default and saves the input and grayscale results as PNG files.

## OpenCL Memory Usage

The program demonstrates:

- Host memory for image generation and result verification
- Global memory for RGB input and grayscale output buffers
- A sub-buffer for the first half of the RGB image
- Local memory for work-group RGB tiles
- Constant memory for grayscale weights
- `uchar3` vector loads for packed RGB pixels

The first half of the input is processed through the sub-buffer. The rest is processed through the parent buffer. The local-memory kernel uses work-group barriers to synchronize work-items after loading RGB data into local memory and before processing each tile.

The grayscale formula is:

```text
gray = (299 * red + 587 * green + 114 * blue) / 1000
```

### Build

From the `module12` directory, run:

```bash
./build.sh
```

This will generate:

- `assignment.exe` — OpenCL application

The build requires a C++ compiler, OpenCL headers, and an OpenCL loader. The PNG-writing header is included in this directory.

### Run

From the `module12` directory, run:

```bash
./run.sh [totalWorkItems] [localSize] [imageWidth] [imageHeight]
```

Or run the executable directly:

```bash
./assignment.exe [totalWorkItems] [localSize] [imageWidth] [imageHeight]
```

The parameters are:

- `totalWorkItems` — Total number of OpenCL work-items launched for each kernel.
- `localSize` — Number of work-items in each work-group.
- `imageWidth` — Width of the generated RGB image in pixels.
- `imageHeight` — Height of the generated RGB image in pixels.

Default values are `262144 256 1024 1024`.

For example:

```bash
./run.sh 262144 256 1024 1024
```

This runs each kernel with 262,144 work-items, organized into work-groups of 256 work-items, processing a 1024 x 1024 image.

### Output

Successful runs save:

- `synthetic_rgb.png` — generated RGB input
- `global_gray.png` — global-memory kernel output
- `local_gray.png` — local-memory kernel output