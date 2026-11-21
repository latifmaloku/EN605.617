#pragma once

namespace SnakeConstants
{
    inline constexpr double DEFAULT_ALPHA = 2.0;
    inline constexpr double DEFAULT_BETA = 1.5;
    inline constexpr double DEFAULT_GAMMA = 5.0;
    inline constexpr int DEFAULT_MAX_ITERATIONS = 500;
    inline constexpr int RUN_MAX_ITERATIONS = 500;

    inline constexpr int WINDOW_RADIUS = 1;
    inline constexpr int WINDOW_DIM = 2 * WINDOW_RADIUS + 1;
    inline constexpr int WINDOW_SIZE = WINDOW_DIM * WINDOW_DIM;

    inline constexpr int GAUSSIAN_RADIUS = 5;
    inline constexpr int GAUSSIAN_SIZE = 2 * GAUSSIAN_RADIUS + 1;

    inline constexpr int RESAMPLE_INTERVAL = 20;
    inline constexpr int RESAMPLE_POINT_THRESHOLD = 15;
    inline constexpr int MIN_CONTOUR_POINTS = 3;
    inline constexpr float INITIAL_RESAMPLE_SPACING = 10.0f;
    inline constexpr float RESAMPLE_SPACING = 4.0f;
    inline constexpr float BALLOON_WEIGHT = 0.15f;
    inline constexpr int WINDOW_RADIUS_MIN = 1;
    inline constexpr int WINDOW_RADIUS_MAX = 3;
    inline constexpr float GAUSSIAN_SIGMA = 1.0f;

    inline constexpr float GEOMETRY_EPSILON = 1e-4f;
    inline constexpr float ENERGY_EPSILON = 1e-6f;
}
