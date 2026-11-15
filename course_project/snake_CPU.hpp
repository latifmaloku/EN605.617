#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class Snake_CPU
 * @brief CPU-based implementation of an Active Contour (Snake) segmentation model.
 *
 * Evolves control points over a target image by minimizing an energy functional
 * composed of elasticity and rigidity and image gradients/edges.
 */
class Snake_CPU {
public:
    /**
     * @brief Constructs a new Snake_CPU with specified energy weights
     * 
     * @param alpha Elasticity weight 
     * @param beta  Rigidity weight 
     * @param gamma External energy weight
     */
    Snake_CPU(double alpha = 0.1, double beta = 0.1, double gamma = 1.0);

    /**
     * @brief Initializes the active contour with a target image and starting vertices
     * 
     * Stores the input image and prepares the initial control points for energy minimization
     * 
     * @param[in] image         The source image
     * @param[in] initialPoints Ordered list of (x, y) pixel coordinates defining the starting closed loop
     */
    void initialize(const cv::Mat& image, const std::vector<cv::Point>& initialPoints);

    /**
     * @brief Executes the iterative energy minimization loop to deform the contour toward object boundaries
     * 
     * @param maxIterations Max number of optimization steps before stopping
     */
    void run(int maxIterations = 100);

    /**
     * @brief Retrieves the current contour coordinates
     * 
     * @return std::vector<cv::Point> Ordered list of coordinates representing the current state
     */
    std::vector<cv::Point> getPoints() const;

private:
    double m_alpha; // Weight controlling contour resistance to stretching
    double m_beta;  // Weight controlling contour resistance to bending
    double m_gamma; // Weight controlling attraction to edges

    cv::Mat m_image;                        // Working copy of the target image
    std::vector<cv::Point> m_snakePoints;   // Current ordered vertices of the active contour
};
