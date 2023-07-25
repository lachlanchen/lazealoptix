#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// Function to match descriptors and return the matches
std::vector<cv::DMatch> matchDescriptors(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches) {
        if (m[0].distance < 0.5 * m[1].distance)
            goodMatches.push_back(m[0]);
    }

    return goodMatches;
}

int main() {
    // Create SIFT detector and descriptor
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    
    // Read images
    fs::path dir = "all/";
    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(dir)) {
        images.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
    }

    // Detect keypoints and compute descriptors for all images
    std::vector<std::vector<cv::KeyPoint>> allKeypoints;
    std::vector<cv::Mat> allDescriptors;
    for (const auto& img : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        allKeypoints.push_back(keypoints);
        allDescriptors.push_back(descriptors);
    }

    // Initialize chains with matches between the first and second image
    std::vector<std::vector<int>> chains;
    std::vector<cv::DMatch> matches = matchDescriptors(allDescriptors[0], allDescriptors[1]);
    for (const auto& m : matches) {
        chains.push_back({m.queryIdx, m.trainIdx});
    }

    // Extend the chains with matches in subsequent images
    for (size_t i = 2; i < images.size(); ++i) {
        std::vector<std::vector<int>> newChains;
        matches = matchDescriptors(allDescriptors[i-1], allDescriptors[i]);
        for (const auto& chain : chains) {
            for (const auto& m : matches) {
                if (chain.back() == m.queryIdx)
                    newChains.push_back(chain + std::vector<int>{m.trainIdx});
            }
        }
        chains = newChains;
    }

    // Draw the chains
    int h = 0, w = 0;
    for (const auto& img : images) {
        h = std::max(h, img.rows);
        w += img.cols;
    }
    cv::Mat outputImg = cv::Mat::zeros(h, w, CV_8UC3);

    std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255)};
    for (const auto& chain : chains) {
        for (size_t i = 0; i < chain.size() - 1; ++i) {
            cv::Point pt1 = allKeypoints[i][chain[i]].pt;
            cv::Point pt2 = allKeypoints[i+1][chain[i+1]].pt;
            pt1.x += i * images[i].cols;
            pt2.x += (i+1) * images[i+1].cols;
            cv::line(outputImg, pt1, pt2, colors[i % colors.size()], 2);
        }
    }

    // Save result image with a unique timestamp suffix
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "result_" << time << ".png";
    cv::imwrite(ss.str(), outputImg);

    return 0;
}

