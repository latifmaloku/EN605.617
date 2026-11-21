#pragma once

#include <opencv2/core.hpp>
#include "snake_base.hpp"
#include <vector>

/**
 * @class Snake_CPU
 * @brief CPU-based implementation of an Active Contour (Snake) segmentation model.
 *
 * Evolves control points over a target image by minimizing an energy functional
 * composed of elasticity and rigidity and image gradients/edges.
 */
class Snake_CPU : public Snake_Base
{
public:
    /**
     * @brief Constructs a new Snake_CPU with specified energy weights
     * 
     * @param alpha Elasticity weight 
     * @param beta  Rigidity weight 
     * @param gamma External energy weight
     * @param balloonWeight Balloon force weight
     * @param windowRadius Radius of the local search window for candidate points
     * @param resampleInterval Number of iterations between contour resampling
     * @param resampleSpacing Spacing between points after resampling
     * @param initialResampleSpacing Spacing for initial resampling of user-defined contour
     * @param gaussianSigma Standard deviation for Gaussian smoothing of the image gradient
     */
    Snake_CPU(double alpha = SnakeConstants::DEFAULT_ALPHA,
              double beta = SnakeConstants::DEFAULT_BETA,
              double gamma = SnakeConstants::DEFAULT_GAMMA,
              double balloonWeight = SnakeConstants::BALLOON_WEIGHT,
              int windowRadius = SnakeConstants::WINDOW_RADIUS,
              int resampleInterval = SnakeConstants::RESAMPLE_INTERVAL,
              float resampleSpacing = SnakeConstants::RESAMPLE_SPACING,
              float initialResampleSpacing = SnakeConstants::INITIAL_RESAMPLE_SPACING,
              float gaussianSigma = SnakeConstants::GAUSSIAN_SIGMA);

    /**
     * @brief Executes the iterative energy minimization loop to deform the contour toward object boundaries
     * 
     * @param maxIterations Max number of optimization steps before stopping
     */
    void run(int maxIterations = SnakeConstants::DEFAULT_MAX_ITERATIONS) override;
};
