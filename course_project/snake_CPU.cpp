#include "snake_CPU.hpp"
#include <iostream>

Snake_CPU::Snake_CPU(double alpha, double beta, double gamma)
    : m_alpha(alpha), m_beta(beta), m_gamma(gamma) {}

void Snake_CPU::initialize(const cv::Mat& image, const std::vector<cv::Point>& initialPoints) {
    m_image = image.clone();
    m_snakePoints = initialPoints;
    std::cout << "Snake_CPU initialized with " << m_snakePoints.size() << " points.\n";
}

void Snake_CPU::run(int maxIterations) {
    // TODO: Implement Snake algorithm here
    std::cout << "Running Snake_CPU for " << maxIterations << " iterations (Not implemented yet)...\n";
}

std::vector<cv::Point> Snake_CPU::getPoints() const {
    return m_snakePoints;
}
