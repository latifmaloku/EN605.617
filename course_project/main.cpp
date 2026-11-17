#include "contour_selector.hpp"
#include "snake_CPU.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string imagePath;

    // Check if the image path is provided as arg
    if (argc > 1) {
        imagePath = argv[1];;
    } else {
        std::cout << "Enter the path to the image file: ";
        std::getline(std::cin, imagePath);
        if (!imagePath.empty() && imagePath.front() == '"' && imagePath.back() == '"') {
            imagePath = imagePath.substr(1, imagePath.size() - 2);
        }
    }
    // Load the image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image at '" << imagePath << "'\n";
        return -1;
    }

    // Select initial points
    ContourSelector selector;
    std::vector<cv::Point> initialPoints;
    if (!selector.selectPoints(image, initialPoints)) {
        std::cout << "Selection canceled.\n";
        return 0;
    }

    Snake_CPU  snake(1.2, 0.5, 3.5);
    snake.initialize(image, initialPoints);
    snake.run(300);

    return 0;
}