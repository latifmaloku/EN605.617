#include "snake_CPU.hpp"
#include "snake_constants.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <chrono>
namespace {
    using namespace SnakeConstants;

    // Converts a vector of integer OpenCV points to floating-point points
    std::vector<cv::Point2f> toFloatPoints(
        const std::vector<cv::Point> &points)
    {
        std::vector<cv::Point2f> result;
        result.reserve(points.size());
        for (const cv::Point &point : points)
            result.emplace_back(static_cast<float>(point.x),
                                static_cast<float>(point.y));
        return result;
    }

    // Computes the gradient magnitude of the input image
    cv::Mat computeGradientMagnitude(const cv::Mat &image, float gaussianSigma)
    {
        // Convert to grayscale if necessary
        cv::Mat gray;
        if (image.channels() == 3)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        }
        else if (image.channels() == 1)
        {
            gray = image.clone();
        }
        else
        {
            throw std::runtime_error("Input image must have 1, 3, or 4 channels.");
        }
        // Ensure the image is in 8-bit unsigned format
        if (gray.depth() != CV_8U)
        {
            gray.convertTo(gray, CV_8U);
        }
        // Create a separable Gaussian kernel for smoothing
        float weights[GAUSSIAN_SIZE];
        float weightSum = 0.0f;
        for (int i = -GAUSSIAN_RADIUS; i <= GAUSSIAN_RADIUS; ++i)
        {
            const float value = std::exp(
                -(static_cast<float>(i * i)) /
                (2.0f * gaussianSigma * gaussianSigma));
            weights[i + GAUSSIAN_RADIUS] = value;
            weightSum += value;
        }
        // Normalize the weights to ensure they sum to 1
        for (float &weight : weights)
        {
            weight /= weightSum;
        }
        // Apply Gaussian smoothing using separable filters
        cv::Mat grayFloat;
        gray.convertTo(grayFloat, CV_32F);
        cv::Mat gaussianKernel(1, GAUSSIAN_SIZE, CV_32F, weights);
        cv::Mat blurred;
        cv::sepFilter2D(grayFloat, blurred, CV_32F,
                        gaussianKernel, gaussianKernel.t(),
                        cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
        // Compute the gradients in x and y directions using Sobel operator
        cv::Mat gradX, gradY, magnitude;
        cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3,
                  1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3,
                  1.0, 0.0, cv::BORDER_REPLICATE);
        cv::magnitude(gradX, gradY, magnitude);
        // Normalize the gradient magnitude to the range [0, 1]
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(magnitude, &minValue, &maxValue);
        if (maxValue > ENERGY_EPSILON)
        {
            magnitude /= static_cast<float>(maxValue);
        }
        else
        {
            magnitude.setTo(0.0f);
        }

        return magnitude;
    }

}

Snake_CPU::Snake_CPU(double alpha, double beta, double gamma,
                     double balloonWeight, int windowRadius,
                     int resampleInterval, float resampleSpacing,
                     float initialResampleSpacing, float gaussianSigma)
    : Snake_Base(alpha, beta, gamma, balloonWeight, windowRadius,
                 resampleInterval, resampleSpacing,
                 initialResampleSpacing, gaussianSigma) {}

void Snake_CPU::run(int maxIterations) {
    if (!canRun(maxIterations, "Snake_CPU"))
    {
        return;
    }
    // Start timing the setup phase
    const auto setupStart = std::chrono::high_resolution_clock::now();
    // Compute the gradient magnitude of the image for external energy
    const int cols = m_image.cols;
    const int rows = m_image.rows;
    const cv::Mat gradMag = computeGradientMagnitude(
        m_image, m_gaussianSigma);

    // Convert the initial snake points to floating-point for evolution
    std::vector<cv::Point2f> pointsIn = toFloatPoints(m_snakePoints);
    std::vector<cv::Point2f> pointsOut(pointsIn.size());
    const auto contourUpdateStart = std::chrono::high_resolution_clock::now();
    double visualizationTime = 0.0;
    // Main iterative evolution loop
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        if (iter > 0 && iter % m_resampleInterval == 0 &&
            pointsIn.size() > RESAMPLE_POINT_THRESHOLD)
        {
            pointsIn = resampleClosedContour(pointsIn, m_resampleSpacing);
            pointsOut.resize(pointsIn.size());
        }

        const int n = static_cast<int>(pointsIn.size());
        if (n < SnakeConstants::MIN_CONTOUR_POINTS)
        {
            break;
        }

        // Compute the centroid of the current contour for balloon force direction
        cv::Point2f centroid(0.0f, 0.0f);
        for (const cv::Point2f &point : pointsIn)
        {
            centroid.x += point.x;
            centroid.y += point.y;
        }
        centroid.x /= static_cast<float>(n);
        centroid.y /= static_cast<float>(n);

        int pointsMoved = 0;

        // Iterate over each point in the contour to compute energy and update positions
        for (int i = 0; i < n; ++i)
        {
            // Determine the previous and next points in the closed contour
            const int prev = (i - 1 + n) % n;
            const int next = (i + 1) % n;
            const cv::Point2f pPrev = pointsIn[prev];
            const cv::Point2f pCurr = pointsIn[i];
            const cv::Point2f pNext = pointsIn[next];
            // Compute the tangent vector and its length for normal direction
            const float tx = pNext.x - pPrev.x;
            const float ty = pNext.y - pPrev.y;
            const float tangentLength = std::hypot(tx, ty);
            float nxNorm = 0.0f;
            float nyNorm = 0.0f;
            // Compute the outward normal vector based on the tangent
            if (tangentLength > ENERGY_EPSILON)
            {
                nxNorm = -ty / tangentLength;
                nyNorm = tx / tangentLength;
                const float toCenterX = centroid.x - pCurr.x;
                const float toCenterY = centroid.y - pCurr.y;
                if (nxNorm * toCenterX + nyNorm * toCenterY < 0.0f)
                {
                    nxNorm = -nxNorm;
                    nyNorm = -nyNorm;
                }
            }
            // Arrays to hold energy values for the local search window
            const int windowSize =
                (2 * m_windowRadius + 1) * (2 * m_windowRadius + 1);
            std::vector<float> eCont(windowSize);
            std::vector<float> eCurv(windowSize);
            std::vector<float> eBalloon(windowSize);
            std::vector<float> eImg(windowSize);
            std::vector<cv::Point> candidates(windowSize);
            float maxCont = ENERGY_EPSILON;
            float maxCurv = ENERGY_EPSILON;

            int idx = 0;
            // Evaluate energy for each candidate point in the local search window
            for (int dy = -m_windowRadius; dy <= m_windowRadius; ++dy)
            {
                for (int dx = -m_windowRadius; dx <= m_windowRadius; ++dx)
                {
                    // Clamp candidate coordinates to stay within image bounds
                    const int baseX = cvRound(pCurr.x);
                    const int baseY = cvRound(pCurr.y);
                    const int nx = std::clamp(baseX + dx, 1, cols - 2);
                    const int ny = std::clamp(baseY + dy, 1, rows - 2);
                    candidates[idx] = cv::Point(nx, ny);
                    // Compute continuity energy based on distances to previous and next points
                    const float distPrevX = nx - pPrev.x;
                    const float distPrevY = ny - pPrev.y;
                    const float distNextX = pNext.x - nx;
                    const float distNextY = pNext.y - ny;
                    const float cont = distPrevX * distPrevX +
                                       distPrevY * distPrevY +
                                       distNextX * distNextX +
                                       distNextY * distNextY;
                    eCont[idx] = cont;
                    maxCont = std::max(maxCont, cont);
                    // Compute curvature energy based on second derivative approximation
                    const float d2x = pPrev.x - 2.0f * nx + pNext.x;
                    const float d2y = pPrev.y - 2.0f * ny + pNext.y;
                    const float curv = d2x * d2x + d2y * d2y;
                    eCurv[idx] = curv;
                    maxCurv = std::max(maxCurv, curv);
                    // Compute balloon and image energies
                    const float moveX = nx - pCurr.x;
                    const float moveY = ny - pCurr.y;
                    eBalloon[idx] =
                        -(moveX * nxNorm + moveY * nyNorm) / m_windowRadius;
                    eImg[idx] = -gradMag.at<float>(ny, nx);
                    ++idx;
                }
            }
            // Find the candidate point with the minimum total energy
            float bestEnergy = std::numeric_limits<float>::max();
            cv::Point2f bestPoint = pCurr;
            for (int k = 0; k < windowSize; ++k)
            {
                // Normalize continuity and curvature energies to [0, 1] range
                const float normCont = eCont[k] / maxCont;
                const float normCurv = eCurv[k] / maxCurv;
                const float totalEnergy =
                    static_cast<float>(m_alpha) * normCont +
                    static_cast<float>(m_beta) * normCurv +
                    static_cast<float>(m_balloonWeight) * eBalloon[k] +
                    static_cast<float>(m_gamma) * eImg[k];
                // Update the best point if a lower energy is found
                if (totalEnergy < bestEnergy)
                {
                    bestEnergy = totalEnergy;
                    bestPoint = cv::Point2f(
                        static_cast<float>(candidates[k].x),
                        static_cast<float>(candidates[k].y));
                }
            }
            // Update the output point and count if it moved significantly
            pointsOut[i] = bestPoint;
            if (std::fabs(bestPoint.x - pCurr.x) > GEOMETRY_EPSILON ||
                std::fabs(bestPoint.y - pCurr.y) > GEOMETRY_EPSILON)
            {
                ++pointsMoved;
            }
        }

        pointsIn.swap(pointsOut);

        // Report progress and check for convergence
        const bool reportIteration =
            ((iter + 1) % m_resampleInterval == 0) ||
            (iter == maxIterations - 1);
        if (reportIteration)
        {
            // Visualize the current state of the snake and measure visualization time
            const auto visualizationStart =
                std::chrono::high_resolution_clock::now();
            visualizeProgress(pointsIn, iter + 1);
            visualizationTime += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() -
                visualizationStart).count();
            if (pointsMoved < std::max(1, n / 50))
            {
                std::cout << "Snake converged at iteration " << (iter + 1) << ".\n";
                break;
            }
        }
    }

    // Convert the final floating-point contour back to integer pixel coordinates
    m_snakePoints.clear();
    m_snakePoints.reserve(pointsIn.size());
    for (const cv::Point2f &point : pointsIn)
    {
        m_snakePoints.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    // End timing the evolution phase
    const auto contourUpdateEnd = std::chrono::high_resolution_clock::now();
    const double setupTime = std::chrono::duration<double, std::milli>(
        contourUpdateStart - setupStart).count();
    const double contourUpdateWallTime = std::chrono::duration<double, std::milli>(
        contourUpdateEnd - contourUpdateStart).count();

    std::cout << "CPU setup time: " << setupTime << " ms\n";
    std::cout << "CPU contour update time: "
              << std::max(0.0, contourUpdateWallTime - visualizationTime)
              << " ms\n";
    std::cout << "CPU visualization time: "
              << visualizationTime << " ms\n";
    std::cout << "CPU total active time: "
              << setupTime + contourUpdateWallTime << " ms\n";

    std::cout << "Finished Active Contour. Press any key on the image window to exit.\n";
    cv::waitKey(0);
    cv::destroyWindow("Active Contour Evolution");
}
