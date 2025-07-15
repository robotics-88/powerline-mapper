/*
© 2025 Robotics 88
Author: Erin Linebarger <erin@robotics88.com>
*/

#ifndef POWERLINE_MAPPER_H_
#define POWERLINE_MAPPER_H_

#include <rclcpp/rclcpp.hpp>

#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/opencv.hpp>

/**
 * @class PowerlineMapper
 * @brief A class for detecting powerlines in point clouds
 */
class PowerlineMapper : public rclcpp::Node {

  public:
    PowerlineMapper();
    ~PowerlineMapper();

  private:
    std::string point_cloud_topic_;
    cv::Mat distance_matrix_;
    std::map<std::pair<int, int>, std::vector<float>>
        distance_map; // Store multiple distances per pixel
    cv::Size image_size_;
    double meters_per_pixel_;
    double origin_x_, origin_y_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    bool detection_enabled_;
    double ground_filter_height_;
    double ground_elevation_;
    geometry_msgs::msg::PoseStamped::SharedPtr
        current_pose_; // Store the latest pose for elevation requests

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr powerline_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr distance_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void detectPowerLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void updateImage(const pcl::PointXYZ &point, float distance);
    void saveGeoTIFF();

}; // class PowerlineMapper

#endif