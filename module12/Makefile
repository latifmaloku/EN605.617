assignment.exe: assignment.c grayscale.cl stb_image_write.h
	g++ -std=c++11 -Wall -Wextra -isystem . \
		-DCL_TARGET_OPENCL_VERSION=120 \
		assignment.c -o assignment.exe -lOpenCL

clean:
	rm -f assignment.exe synthetic_rgb.png global_gray.png local_gray.png
