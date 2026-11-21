#pragma once

#include "snake_constants.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

/**
 * @class Snake_Base
 * @brief Base class shared by Snake implementations.
 *
 * Stores the source image, active contour points, energy weights, and
 * configurable contour evolution parameters. It also provides shared
 * initialization, validation, resampling, visualization, and point access.
 */
class Snake_Base
{
public:
    /**
     * @brief Constructs a Snake with configurable contour parameters.
     *
     * @param alpha Elasticity weight
     * @param beta Rigidity weight
     * @param gamma External image energy weight
     * @param balloonWeight Balloon force weight
     * @param windowRadius Local candidate search radius
     * @param resampleInterval Iterations between contour resampling
     * @param resampleSpacing Distance between points during resampling
     * @param initialResampleSpacing Distance between initial contour points
     * @param gaussianSigma Standard deviation of the Gaussian smoothing kernel
     */
    Snake_Base(double alpha = SnakeConstants::DEFAULT_ALPHA,
               double beta = SnakeConstants::DEFAULT_BETA,
               double gamma = SnakeConstants::DEFAULT_GAMMA,
               double balloonWeight = SnakeConstants::BALLOON_WEIGHT,
               int windowRadius = SnakeConstants::WINDOW_RADIUS,
               int resampleInterval = SnakeConstants::RESAMPLE_INTERVAL,
               float resampleSpacing = SnakeConstants::RESAMPLE_SPACING,
               float initialResampleSpacing = SnakeConstants::INITIAL_RESAMPLE_SPACING,
               float gaussianSigma = SnakeConstants::GAUSSIAN_SIGMA)
        : m_alpha(alpha),
          m_beta(beta),
          m_gamma(gamma),
          m_balloonWeight(balloonWeight),
          m_windowRadius(std::clamp(windowRadius,
                                    SnakeConstants::WINDOW_RADIUS_MIN,
                                    SnakeConstants::WINDOW_RADIUS_MAX)),
          m_resampleInterval(resampleInterval > 0 ? resampleInterval : 1),
          m_resampleSpacing(resampleSpacing > 0.0f ? resampleSpacing : 0.1f),
          m_initialResampleSpacing(initialResampleSpacing > 0.0f
                                        ? initialResampleSpacing
                                        : 0.1f),
          m_gaussianSigma(gaussianSigma > 0.0f ? gaussianSigma : 0.1f)
    {
    }

    virtual ~Snake_Base() = default;

    /**
     * @brief Initializes the active contour with an image and starting points.
     *
     * @param[in] image Source image
     * @param[in] initialPoints Ordered initial contour points
     */
    virtual void initialize(const cv::Mat &image,
                            const std::vector<cv::Point> &initialPoints)
    {
        // Store a copy of the image and resample the initial contour points
        m_image = image.clone();
        m_snakePoints = resampleClosedContour(
            initialPoints, m_initialResampleSpacing);

        std::cout << "Snake initialized with " << m_snakePoints.size()
                  << " evenly spaced points.\n";
    }

    /**
     * @brief Executes the contour evolution.
     *
     * @param maxIterations Maximum number of optimization iterations
     */
    virtual void run(int maxIterations = SnakeConstants::DEFAULT_MAX_ITERATIONS) = 0;

    /**
     * @brief Retrieves the current contour points.
     *
     * @return Ordered list of contour coordinates
     */
    std::vector<cv::Point> getPoints() const
    {
        return m_snakePoints;
    }

protected:
    /**
     * @brief Checks whether the Snake is ready to run.
     *
     * @param maxIterations Maximum number of optimization iterations
     * @param implementationName Name used in validation error messages
     * @return true if initialization and iteration settings are valid
     */
    bool canRun(int maxIterations, const char *implementationName) const
    {
        if (m_image.empty() ||
            m_snakePoints.size() < SnakeConstants::MIN_CONTOUR_POINTS)
        {
            std::cerr << "Error: " << implementationName
                      << " must be initialized with a valid image and >= 3 points.\n";
            return false;
        }
        if (m_image.cols < 3 || m_image.rows < 3)
        {
            std::cerr << "Error: Image must be at least 3x3 pixels.\n";
            return false;
        }
        return maxIterations > 0;
    }

    /**
     * @brief Draws the current contour and iteration number on the source image.
     *
     * @param[in] points Current contour points
     * @param iter Current iteration number
     */
    void visualizeProgress(const std::vector<cv::Point> &points, int iter) const
    {
        // Create a copy of the original image to draw on
        cv::Mat display = m_image.clone();
        if (points.size() > 1)
        {
            // Draw lines between consecutive points and connect the last to the first
            for (size_t i = 0; i < points.size(); ++i)
            {
                cv::line(display, points[i], points[(i + 1) % points.size()],
                         cv::Scalar(0, 255, 0), 2);
                cv::circle(display, points[i], 2,
                           cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::putText(display, "Iteration: " + std::to_string(iter),
                    cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 0), 2);
        cv::imshow("Active Contour Evolution", display);
        cv::waitKey(1);
    }

    /**
     * @brief Draws floating-point contour points after converting them to pixels.
     *
     * @param[in] points Current floating-point contour points
     * @param iter Current iteration number
     */
    void visualizeProgress(const std::vector<cv::Point2f> &points, int iter) const
    {
        std::vector<cv::Point> integerPoints;
        integerPoints.reserve(points.size());
        // Convert floating-point points to integer pixel coordinates for visualization
        for (const cv::Point2f &point : points)
        {
            integerPoints.emplace_back(cvRound(point.x), cvRound(point.y));
        }
        visualizeProgress(integerPoints, iter);
    }

    /**
     * @brief Resamples a closed contour of integer points at equal spacing.
     *
     * @param[in] rawPoints Original ordered contour points
     * @param spacing Requested distance between output points
     * @return Resampled integer contour points
     */
    static std::vector<cv::Point> resampleClosedContour(
        const std::vector<cv::Point> &rawPoints, float spacing)
    {
        // Convert integer points to floating-point for resampling
        std::vector<cv::Point2f> clean;
        clean.reserve(rawPoints.size());
        // Remove duplicate points that are too close together
        for (const cv::Point &point : rawPoints)
        {
            cv::Point2f candidate(static_cast<float>(point.x),
                                  static_cast<float>(point.y));
            if (clean.empty() ||
                cv::norm(candidate - clean.back()) > SnakeConstants::GEOMETRY_EPSILON)
            {
                clean.push_back(candidate);
            }
        }
        // If the contour is closed, remove the last point if it's too close to the first
        if (clean.size() > 1 &&
            cv::norm(clean.front() - clean.back()) <= SnakeConstants::GEOMETRY_EPSILON)
        {
            clean.pop_back();
        }
        if (clean.size() < SnakeConstants::MIN_CONTOUR_POINTS)
        {
            std::vector<cv::Point> result;
            for (const cv::Point2f &point : clean)
            {
                result.emplace_back(cvRound(point.x), cvRound(point.y));
            }
            return result;
        }
        // Compute the lengths of each segment and the total perimeter
        std::vector<float> lengths(clean.size());
        float perimeter = 0.0f;
        for (size_t i = 0; i < clean.size(); ++i)
        {
            lengths[i] = cv::norm(clean[(i + 1) % clean.size()] - clean[i]);
            perimeter += lengths[i];
        }
        if (perimeter <= SnakeConstants::GEOMETRY_EPSILON)
        {
            return {};
        }

        const int outputCount = std::max(
            3, static_cast<int>(std::round(perimeter / spacing)));
        std::vector<cv::Point> output;
        output.reserve(outputCount);

        size_t segment = 0;
        float segmentStart = 0.0f;
        // Generate evenly spaced points along the contour
        for (int i = 0; i < outputCount; ++i)
        {
            const float target = perimeter * static_cast<float>(i) / outputCount;
            while (segment + 1 < clean.size() &&
                   segmentStart + lengths[segment] < target)
            {
                segmentStart += lengths[segment];
                ++segment;
            }
            // Interpolate between the two points of the current segment
            const float segmentLength = lengths[segment];
            const float t = (segmentLength > SnakeConstants::ENERGY_EPSILON)
                                ? (target - segmentStart) / segmentLength
                                : 0.0f;
            const cv::Point2f &a = clean[segment];
            const cv::Point2f &b = clean[(segment + 1) % clean.size()];
            output.emplace_back(
                cvRound(a.x + t * (b.x - a.x)),
                cvRound(a.y + t * (b.y - a.y)));
        }
        return output;
    }

    /**
     * @brief Resamples a closed contour of floating-point points at equal spacing.
     *
     * @param[in] rawPoints Original ordered contour points
     * @param spacing Requested distance between output points
     * @return Resampled floating-point contour points
     */
    static std::vector<cv::Point2f> resampleClosedContour(
        const std::vector<cv::Point2f> &rawPoints, float spacing)
    {
        std::vector<cv::Point2f> clean;
        clean.reserve(rawPoints.size());
        // Remove duplicate points that are too close together
        for (const cv::Point2f &point : rawPoints)
        {
            if (clean.empty() ||
                cv::norm(point - clean.back()) > SnakeConstants::GEOMETRY_EPSILON)
            {
                clean.push_back(point);
            }
        }

        if (clean.size() > 1 &&
            cv::norm(clean.front() - clean.back()) <= SnakeConstants::GEOMETRY_EPSILON)
        {
            clean.pop_back();
        }
        if (clean.size() < SnakeConstants::MIN_CONTOUR_POINTS)
        {
            return clean;
        }
        // Compute the lengths of each segment and the total perimeter
        std::vector<float> lengths(clean.size());
        float perimeter = 0.0f;
        for (size_t i = 0; i < clean.size(); ++i)
        {
            lengths[i] = cv::norm(clean[(i + 1) % clean.size()] - clean[i]);
            perimeter += lengths[i];
        }
        if (perimeter <= SnakeConstants::GEOMETRY_EPSILON)
        {
            return clean;
        }

        const int outputCount = std::max(
            3, static_cast<int>(std::round(perimeter / spacing)));
        std::vector<cv::Point2f> output;
        output.reserve(outputCount);

        size_t segment = 0;
        float segmentStart = 0.0f;
        // Generate evenly spaced points along the contour
        for (int i = 0; i < outputCount; ++i)
        {
            const float target = perimeter * static_cast<float>(i) / outputCount;
            while (segment + 1 < clean.size() &&
                   segmentStart + lengths[segment] < target)
            {
                segmentStart += lengths[segment];
                ++segment;
            }
            // Interpolate between the two points of the current segment
            const float segmentLength = lengths[segment];
            const float t = (segmentLength > SnakeConstants::ENERGY_EPSILON)
                                ? (target - segmentStart) / segmentLength
                                : 0.0f;
            const cv::Point2f &a = clean[segment];
            const cv::Point2f &b = clean[(segment + 1) % clean.size()];
            output.emplace_back(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y));
        }
        return output;
    }

    double m_alpha;
    double m_beta;
    double m_gamma;
    double m_balloonWeight;
    int m_windowRadius;
    int m_resampleInterval;
    float m_resampleSpacing;
    float m_initialResampleSpacing;
    float m_gaussianSigma;
    cv::Mat m_image;
    std::vector<cv::Point> m_snakePoints;
};
