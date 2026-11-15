#include "contour_selector.hpp"
#include <iostream>

ContourSelector::ContourSelector(const std::string& windowName) 
    : m_windowName(windowName) {}

void ContourSelector::redrawCanvas() {
    m_displayImage = m_originalImage.clone();
    // Draw lines between points
    if (m_controlPoints.size() > 1) {
        // Draw lines between consecutive points
        for (size_t i = 0; i < m_controlPoints.size() - 1; ++i) {
            cv::line(m_displayImage, m_controlPoints[i], m_controlPoints[i + 1], cv::Scalar(0, 255, 0), 2);
        }
        // If the contour is closed, draw a line connecting the last point to the first
        if (m_isClosed) {
            cv::line(m_displayImage, m_controlPoints.back(), m_controlPoints.front(), cv::Scalar(0, 255, 0), 2);
        }
    }
    // Draw points
    for (const auto& pt : m_controlPoints) {
        cv::circle(m_displayImage, pt, 4, cv::Scalar(0, 0, 255), -1);
    }
    // Show updated image
    cv::imshow(m_windowName, m_displayImage);
}

void ContourSelector::onMouse(int event, int x, int y, int flags, void* userData) {
    ContourSelector* self = static_cast<ContourSelector*>(userData);
    // Handle left-click to add a new point
    if (event == cv::EVENT_LBUTTONDOWN && !self->m_isClosed) {
        self->m_controlPoints.emplace_back(x, y);
        self->redrawCanvas();
    } // Handle right-click to undo the the last click
    else if (event == cv::EVENT_RBUTTONDOWN && !self->m_controlPoints.empty()) {
        self->m_controlPoints.pop_back();
        self->m_isClosed = false;
        self->redrawCanvas();
    }
}

bool ContourSelector::selectPoints(const cv::Mat& image, std::vector<cv::Point>& outPoints) {
    m_originalImage = image.clone();
    m_controlPoints.clear();
    m_isClosed = false;
    // Set up OpenCV window and mouse callback
    cv::namedWindow(m_windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(m_windowName, onMouse, this);
    redrawCanvas();
    // Display controls
    std::cout << "\n--- Controls ---\n"
              << "Left Click  : Add contour control point\n"
              << "Right Click : Undo last point\n"
              << "'c' key     : Close the contour loop\n"
              << "'Enter'     : Confirm points and proceed\n"
              << "'Esc'       : Exit\n\n";

    while (true) {
        int key = cv::waitKey(20);

        if (key == 27) { // ESC
            cv::destroyWindow(m_windowName);
            return false;
        } 
        else if (key == 'c' || key == 'C') { // Close contour
            if (m_controlPoints.size() >= MIN_POINTS) {
                m_isClosed = true;
                redrawCanvas();
            } else {
                std::cout << "Need at least " << MIN_POINTS << " points to close a contour.\n";
            }
        } 
        else if (key == 13 || key == 10) { // ENTER
            if (m_controlPoints.size() < MIN_POINTS) {
                std::cout << "Please select at least " << MIN_POINTS << " points before proceeding.\n";
            } else {
                m_isClosed = true;
                redrawCanvas();
                outPoints = m_controlPoints;
                cv::destroyWindow(m_windowName);
                return true;
            }
        }
    }
}
