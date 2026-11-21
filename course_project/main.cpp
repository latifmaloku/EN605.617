#include "contour_selector.hpp"
#include "snake_CPU.hpp"
#include "snake_constants.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

namespace
{
    struct SnakeOptions
    {
        double alpha = SnakeConstants::DEFAULT_ALPHA;
        double beta = SnakeConstants::DEFAULT_BETA;
        double gamma = SnakeConstants::DEFAULT_GAMMA;
        double balloonWeight = SnakeConstants::BALLOON_WEIGHT;
        int windowRadius = SnakeConstants::WINDOW_RADIUS;
        int resampleInterval = SnakeConstants::RESAMPLE_INTERVAL;
        float resampleSpacing = SnakeConstants::RESAMPLE_SPACING;
        float initialResampleSpacing = SnakeConstants::INITIAL_RESAMPLE_SPACING;
        float gaussianSigma = SnakeConstants::GAUSSIAN_SIGMA;
        int maxIterations = SnakeConstants::RUN_MAX_ITERATIONS;
    };

    bool parseDoubleOption(int argc, char **argv, int &i,
                           const char *name, double &value)
    {
        if (i + 1 >= argc)
        {
            std::cerr << "Error: " << name << " requires a value.\n";
            return false;
        }
        char *end = nullptr;
        value = std::strtod(argv[++i], &end);
        if (end == argv[i] || *end != '\0')
        {
            std::cerr << "Error: Invalid value for " << name << ".\n";
            return false;
        }
        return true;
    }

    bool parseIntOption(int argc, char **argv, int &i,
                        const char *name, int &value)
    {
        if (i + 1 >= argc)
        {
            std::cerr << "Error: " << name << " requires a value.\n";
            return false;
        }
        char *end = nullptr;
        long parsed = std::strtol(argv[++i], &end, 10);
        if (end == argv[i] || *end != '\0')
        {
            std::cerr << "Error: Invalid value for " << name << ".\n";
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    }
}

int main(int argc, char **argv)
{
    std::string imagePath;
    SnakeOptions options;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        double doubleValue = 0.0;
        int intValue = 0;

        if (arg == "--help")
        {
            std::cout
                << "Usage: snake_app [image] [options]\n"
                << "  --alpha <value>\n"
                << "  --beta <value>\n"
                << "  --gamma <value>\n"
                << "  --balloon <value>\n"
                << "  --window-radius <1-3>\n"
                << "  --resample-interval <iterations>\n"
                << "  --resample-spacing <pixels>\n"
                << "  --initial-spacing <pixels>\n"
                << "  --gaussian-sigma <value>\n"
                << "  --iterations <count>\n";
            return 0;
        }
        else if (arg == "--alpha")
        {
            if (!parseDoubleOption(argc, argv, i, "--alpha", doubleValue))
            {
                return -1;
            }
            options.alpha = doubleValue;
        }
        else if (arg == "--beta")
        {
            if (!parseDoubleOption(argc, argv, i, "--beta", doubleValue))
            {
                return -1;
            }
            options.beta = doubleValue;
        }
        else if (arg == "--gamma")
        {
            if (!parseDoubleOption(argc, argv, i, "--gamma", doubleValue))
            {
                return -1;
            }
            options.gamma = doubleValue;
        }
        else if (arg == "--balloon")
        {
            if (!parseDoubleOption(argc, argv, i, "--balloon", doubleValue))
            {
                return -1;
            }
            options.balloonWeight = doubleValue;
        }
        else if (arg == "--gaussian-sigma")
        {
            if (!parseDoubleOption(argc, argv, i, "--gaussian-sigma", doubleValue))
            {
                return -1;
            }
            options.gaussianSigma = static_cast<float>(doubleValue);
        }
        else if (arg == "--window-radius")
        {
            if (!parseIntOption(argc, argv, i, "--window-radius", intValue))
            {
                return -1;
            }
            options.windowRadius = intValue;
        }
        else if (arg == "--resample-interval")
        {
            if (!parseIntOption(argc, argv, i, "--resample-interval", intValue))
            {
                return -1;
            }
            options.resampleInterval = intValue;
        }
        else if (arg == "--resample-spacing")
        {
            if (!parseDoubleOption(argc, argv, i, "--resample-spacing", doubleValue))
            {
                return -1;
            }
            options.resampleSpacing = static_cast<float>(doubleValue);
        }
        else if (arg == "--initial-spacing")
        {
            if (!parseDoubleOption(argc, argv, i, "--initial-spacing", doubleValue))
            {
                return -1;
            }
            options.initialResampleSpacing = static_cast<float>(doubleValue);
        }
        else if (arg == "--iterations")
        {
            if (!parseIntOption(argc, argv, i, "--iterations", intValue))
            {
                return -1;
            }
            options.maxIterations = intValue;
        }
        else if (arg == "--cpu")
        {
            // CPU is the only available implementation.
        }
        else if (imagePath.empty())
        {
            imagePath = arg;
        }
        else
        {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            return -1;
        }
    }

    if (imagePath.empty())
    {
        std::cout << "Enter the path to the image file: ";
        std::getline(std::cin, imagePath);
        if (!imagePath.empty() && imagePath.front() == '"' && imagePath.back() == '"')
        {
            imagePath = imagePath.substr(1, imagePath.size() - 2);
        }
    }
    // Validate options
    if (options.windowRadius < SnakeConstants::WINDOW_RADIUS_MIN ||
        options.windowRadius > SnakeConstants::WINDOW_RADIUS_MAX)
    {
        std::cerr << "Error: --window-radius must be between 1 and 3.\n";
        return -1;
    }
    if (options.resampleInterval <= 0 ||
        options.resampleSpacing <= 0.0f ||
        options.initialResampleSpacing <= 0.0f ||
        options.gaussianSigma <= 0.0f ||
        options.maxIterations <= 0)
    {
        std::cerr << "Error: Iteration, spacing, and Gaussian values must be positive.\n";
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image at '" << imagePath << "'\n";
        return -1;
    }

    // Select initial points
    ContourSelector selector;
    std::vector<cv::Point> initialPoints;
    if (!selector.selectPoints(image, initialPoints))
    {
        std::cout << "Selection canceled.\n";
        return 0;
    }

    Snake_CPU snake(
        options.alpha, options.beta, options.gamma,
        options.balloonWeight, options.windowRadius,
        options.resampleInterval, options.resampleSpacing,
        options.initialResampleSpacing, options.gaussianSigma);
    snake.initialize(image, initialPoints);
    snake.run(options.maxIterations);

    return 0;
}
