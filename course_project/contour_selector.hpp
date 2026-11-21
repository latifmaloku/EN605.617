#pragma once

#include <opencv2/opencv.hpp>
#include "snake_constants.hpp"
#include <vector>
#include <string>
/**
 * @class ContourSelector
 * @brief GUI tool for selecting initial control points.
 *
 * Displays window where users can click points to place vertices, undo placements,
 * close polygons
 */
class ContourSelector {
public:
    /**
     * @brief Constructor
     * 
     * @param windowName Title displayed GUI window.
     */
    explicit ContourSelector(const std::string& windowName = "Active Contour Initialization");
    
    /**
     * @brief Launches selection window and captures user points.
     * 
     * Controls:
     * - **Left Click**: Add a point
     * - **'c' / 'C'**: Connect the first and last points
     * - **Enter**: Confirm points and close the window.
     * - **Esc**: Cancel selection and close the window.
     * 
     * @param[in]  image     Source image to display and annotate.
     * @param[out] outPoints Vector populated with the confirmed (x, y) pixel coordinates.
     * 
     * @return true  If user confirmed a valid contour.
     * @return false If user exited via the Escape key.
     */
    bool selectPoints(const cv::Mat& image, std::vector<cv::Point>& outPoints);

private:
    std::string m_windowName;              // title of window
    cv::Mat m_originalImage;               // input image
    cv::Mat m_displayImage;                // working image buffer
    std::vector<cv::Point> m_controlPoints;// vector of control points
    bool m_isClosed = false;               // flag indicating whether the contour loop is closed

    /**
     * @brief Render the display with current control points and connecting edges.
     */
    void redrawCanvas();

    /**
     * @brief OpenCV mouse event callback bridge.
     * 
     * @param event    mouse event
     * @param x        x-coordinate of mouuse cursor
     * @param y        y-coordinate of mouse cursor
     * @param flags    Event flags passed by OpenCV
     * @param userdata pointer to contour selector instance
     */
    static void onMouse(int event, int x, int y, int flags, void* userData);
};
