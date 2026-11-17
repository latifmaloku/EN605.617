#include "snake_CPU.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace {
    // Helper: Resamples a sparse closed polygon into evenly spaced points
    std::vector<cv::Point> resampleClosedContour(const std::vector<cv::Point>& rawPoints, double stepSize = 4.0) {
        std::vector<cv::Point> densePoints;
        size_t n = rawPoints.size();
        if (n < 3) return rawPoints;

        for (size_t i = 0; i < n; ++i) {
            cv::Point2d p1 = rawPoints[i];
            cv::Point2d p2 = rawPoints[(i + 1) % n];

            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double segmentLength = std::hypot(dx, dy);

            int numSteps = std::max(1, static_cast<int>(std::round(segmentLength / stepSize)));
            for (int step = 0; step < numSteps; ++step) {
                double t = static_cast<double>(step) / numSteps;
                densePoints.emplace_back(
                    static_cast<int>(std::round(p1.x + t * dx)),
                    static_cast<int>(std::round(p1.y + t * dy))
                );
            }
        }
        return densePoints;
    }

    // Helper: Computes normalized gradient magnitude map (edges = high values)
    cv::Mat computeGradientMagnitude(const cv::Mat& image) {
        cv::Mat gray, blurred, gradX, gradY, mag;

        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Larger kernel (11x11) smooths out wood grain, hair texture, and noise
        cv::GaussianBlur(gray, blurred, cv::Size(11, 11), 1.0);

        cv::Sobel(blurred, gradX, CV_64F, 1, 0, 3);
        cv::Sobel(blurred, gradY, CV_64F, 0, 1, 3);
        cv::magnitude(gradX, gradY, mag);

        // Normalize gradient magnitude to [0.0, 1.0]
        double minVal = 0.0, maxVal = 0.0;
        cv::minMaxLoc(mag, &minVal, &maxVal);
        if (maxVal > 1e-6) {
            mag /= maxVal;
        }
        return mag;
    }


    // Helper: Draws the evolving contour onto a window
    void visualizeProgress(const cv::Mat& baseImage, const std::vector<cv::Point>& points, int iter) {
        cv::Mat display = baseImage.clone();
        if (points.size() > 1) {
            for (size_t i = 0; i < points.size(); ++i) {
                cv::line(display, points[i], points[(i + 1) % points.size()], cv::Scalar(0, 255, 0), 2);
                cv::circle(display, points[i], 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::putText(display, "Iteration: " + std::to_string(iter), cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        cv::imshow("Active Contour Evolution", display);
        cv::waitKey(15);
    }
}

Snake_CPU::Snake_CPU(double alpha, double beta, double gamma)
    : m_alpha(alpha), m_beta(beta), m_gamma(gamma) {}

void Snake_CPU::initialize(const cv::Mat& image, const std::vector<cv::Point>& initialPoints) {
    m_image = image.clone();
    // Resample the initial contour to ensure uniform spacing of control points
    m_snakePoints = resampleClosedContour(initialPoints, 10.0);
    std::cout << "Snake_CPU initialized with " << m_snakePoints.size() << " resampled points.\n";
}

void Snake_CPU::run(int maxIterations) {
    if (m_image.empty() || m_snakePoints.size() < 3) {
        std::cerr << "Error: ActiveContour must be initialized with a valid image and >= 3 points.\n";
        return;
    }

    cv::Mat gradMag = computeGradientMagnitude(m_image);
    const int windowRadius = 3; // 7x7 search window
    const int windowSize = (2 * windowRadius + 1) * (2 * windowRadius + 1);
    const double balloonWeight = 0.35; // Positive = shrink inward, Negative = expand outward

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Resample every 10 iterations so vertices don't bunch up as the loop shrinks
        if (iter > 0 && iter % 10 == 0 && m_snakePoints.size() > 15) {
            m_snakePoints = resampleClosedContour(m_snakePoints, 4.0);
        }

        const int n = static_cast<int>(m_snakePoints.size());
        int pointsMoved = 0;

        // Compute contour centroid to determine inward vs. outward direction
        cv::Point2d centroid(0.0, 0.0);
        for (const auto& pt : m_snakePoints) {
            centroid.x += pt.x;
            centroid.y += pt.y;
        }
        centroid.x /= n;
        centroid.y /= n;

        for (int i = 0; i < n; ++i) {
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;

            cv::Point pPrev = m_snakePoints[prev];
            cv::Point pCurr = m_snakePoints[i];
            cv::Point pNext = m_snakePoints[next];

            // Compute unit tangent and unit inward normal at pCurr
            double tx = pNext.x - pPrev.x;
            double ty = pNext.y - pPrev.y;
            double tLen = std::hypot(tx, ty);
            double nxNorm = 0.0, nyNorm = 0.0;
            if (tLen > 1e-6) {
                // Perpendicular vector
                nxNorm = -ty / tLen;
                nyNorm = tx / tLen;
                // Ensure normal points inward toward the centroid
                double toCenterX = centroid.x - pCurr.x;
                double toCenterY = centroid.y - pCurr.y;
                if (nxNorm * toCenterX + nyNorm * toCenterY < 0) {
                    nxNorm = -nxNorm;
                    nyNorm = -nyNorm;
                }
            }

            std::vector<cv::Point> candidates;
            candidates.reserve(windowSize);

            std::vector<double> eCont(windowSize, 0.0);
            std::vector<double> eCurv(windowSize, 0.0);
            std::vector<double> eBalloon(windowSize, 0.0);
            std::vector<double> eImg(windowSize, 0.0);

            double maxCont = 1e-6;
            double maxCurv = 1e-6;

            int idx = 0;
            for (int dy = -windowRadius; dy <= windowRadius; ++dy) {
                for (int dx = -windowRadius; dx <= windowRadius; ++dx) {
                    int nx = std::clamp(pCurr.x + dx, 1, m_image.cols - 2);
                    int ny = std::clamp(pCurr.y + dy, 1, m_image.rows - 2);
                    cv::Point cand(nx, ny);
                    candidates.push_back(cand);

                    // Continuity Energy
                    double distPrev = std::hypot(cand.x - pPrev.x, cand.y - pPrev.y);
                    double distNext = std::hypot(pNext.x - cand.x, pNext.y - cand.y);
                    double cont = distPrev * distPrev + distNext * distNext;
                    eCont[idx] = cont;
                    maxCont = std::max(maxCont, cont);

                    // Curvature Energy
                    double d2x = pPrev.x - 2.0 * cand.x + pNext.x;
                    double d2y = pPrev.y - 2.0 * cand.y + pNext.y;
                    double curv = d2x * d2x + d2y * d2y;
                    eCurv[idx] = curv;
                    maxCurv = std::max(maxCurv, curv);

                    // Balloon Energy: rewards moving in the inward normal direction
                    double moveX = cand.x - pCurr.x;
                    double moveY = cand.y - pCurr.y;
                    // Normalize projection by windowRadius so it stays in [-1.0, 1.0]
                    eBalloon[idx] = -(moveX * nxNorm + moveY * nyNorm) / windowRadius;

                    // External Image Energy (edges = -1.0, flat = 0.0)
                    eImg[idx] = -gradMag.at<double>(ny, nx);

                    idx++;
                }
            }

            double bestEnergy = std::numeric_limits<double>::max();
            cv::Point bestPoint = pCurr;

            for (int k = 0; k < idx; ++k) {
                double normCont = eCont[k] / maxCont;
                double normCurv = eCurv[k] / maxCurv;

                double totalEnergy = m_alpha * normCont 
                                   + m_beta * normCurv 
                                   + balloonWeight * eBalloon[k] 
                                   + m_gamma * eImg[k];

                if (totalEnergy < bestEnergy) {
                    bestEnergy = totalEnergy;
                    bestPoint = candidates[k];
                }
            }

            if (bestPoint != pCurr) {
                m_snakePoints[i] = bestPoint;
                pointsMoved++;
            }
        }

        visualizeProgress(m_image, m_snakePoints, iter + 1);

        if (pointsMoved < std::max(1, n / 50)) {
            std::cout << "Snake converged at iteration " << (iter + 1) << ".\n";
            break;
        }
    }

    std::cout << "Finished Active Contour. Press any key on the image window to exit.\n";
    cv::waitKey(0);
    cv::destroyWindow("Active Contour Evolution");
}


std::vector<cv::Point> Snake_CPU::getPoints() const {
    return m_snakePoints;
}
