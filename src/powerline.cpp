/*
© 2025 Robotics 88
Author: Erin Linebarger <erin@robotics88.com>
*/

#include "powerline_mapper/powerline.h"
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>

#include <gdal_priv.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/passthrough.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <filesystem>
#include <thread>

#include "messages_88/srv/get_map_data.hpp"

using std::placeholders::_1;

using namespace std::chrono_literals;

PowerlineMapper::PowerlineMapper()
    : Node("powerline_mapper"),
      point_cloud_topic_(""),
      image_size_(1000, 1000),
      meters_per_pixel_(0.5),
      origin_x_(-250),
      origin_y_(-250),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_),
      detection_enabled_(true),
      ground_filter_height_(1.0),
      ground_elevation_(0.0) {
    // Get params
    std::string pointcloud_out_topic;
    this->declare_parameter("point_cloud_topic", point_cloud_topic_);

    this->get_parameter("point_cloud_topic", point_cloud_topic_);

    // Set up pubs and subs
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/mavros/vision_pose/pose", 10,
        std::bind(&PowerlineMapper::poseCallback, this, std::placeholders::_1));
    point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        point_cloud_topic_, 10, std::bind(&PowerlineMapper::pointCloudCallback, this, _1));
    powerline_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("power_line_cloud", 10);
    distance_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("power_line_distances", 10);
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("powerline_image", 10);

    // Initialize OpenCV image
    distance_matrix_ = cv::Mat(image_size_, CV_32F, std::numeric_limits<float>::max());
}

PowerlineMapper::~PowerlineMapper() {
    saveGeoTIFF();
}

void PowerlineMapper::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    current_pose_ = msg; // Store the latest pose for elevation requests
}

void PowerlineMapper::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!detection_enabled_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "⚠ Waiting for MAV altitude > 5m...");
        return;
    }

    // Convert ROS msg to PCL and store
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);
    std_msgs::msg::Header header = msg->header;

    // Get latest elevation
    std::shared_ptr<rclcpp::Node> get_elevation_node =
        rclcpp::Node::make_shared("get_elevation_node");
    auto get_elevation_client = get_elevation_node->create_client<messages_88::srv::GetMapData>(
        "/task_manager/get_map_data");
    auto elevation_req = std::make_shared<messages_88::srv::GetMapData::Request>();
    elevation_req->map_position =
        current_pose_->pose.position; // Use the current position for elevation
    elevation_req->adjust_params = false;

    auto result = get_elevation_client->async_send_request(elevation_req);
    if (rclcpp::spin_until_future_complete(get_elevation_node, result, 1s) ==
        rclcpp::FutureReturnCode::SUCCESS) {

        try {
            ground_elevation_ = result.get()->ret_altitude;
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to get elevation result, using default alt of %fm",
                         ground_elevation_);
        }

    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to get elevation");
    }

    // Filter on elevation
    pcl::PassThrough<pcl::PointXYZ> pass;
    // Z
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    double lo = ground_elevation_ + ground_filter_height_; // Adjusted to filter out ground points
    pass.setFilterLimits(lo, std::numeric_limits<double>::max());
    pass.filter(*cloud);
    if (cloud->points.empty()) {
        return;
    }

    detectPowerLines(cloud);
}

void PowerlineMapper::detectPowerLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    // Step 1: Cluster extraction to isolate power lines
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(5000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Step 2: Identify linear structures using RANSAC and store them
    pcl::PointCloud<pcl::PointXYZ>::Ptr power_line_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_power_line_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr power_line_indices(new pcl::PointIndices);
    for (const auto &indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : indices.indices)
            cluster->push_back(cloud->points[index]);

        if (cluster->size() > 10) {
            // Perform PCA (Principal Component Analysis)
            pcl::PCA<pcl::PointXYZ> pca;
            pca.setInputCloud(cluster);
            Eigen::Vector3f eigenvalues = pca.getEigenValues();
            Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

            // Principal direction
            Eigen::Vector3f direction = eigenvectors.col(0);
            float a = direction(0), b = direction(1), c = direction(2);

            // Check if the cluster is a horizontal power line
            if (std::abs(c) < 0.1) {
                *power_line_cloud += *cluster;
                for (int index : indices.indices)
                    power_line_indices->indices.push_back(index);
            }
        }
    }

    // Step 5: Extract Non-Power Line Cloud (Everything Except Power Line Points)
    pcl::ExtractIndices<pcl::PointXYZ> non_power_extract;
    non_power_extract.setInputCloud(cloud);
    non_power_extract.setIndices(power_line_indices);
    non_power_extract.setNegative(true);
    non_power_extract.filter(*non_power_line_cloud);

    // Step 6: Compute Nearest Neighbor Distance
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(non_power_line_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr power_line_with_distances(
        new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto &point : power_line_cloud->points) {
        std::vector<int> nearest_index(1);
        std::vector<float> nearest_distance(1);

        pcl::PointXYZ search_point = point;
        pcl::PointXYZI point_with_distance;
        point_with_distance.x = search_point.x;
        point_with_distance.y = search_point.y;
        point_with_distance.z = search_point.z;

        if (kdtree.nearestKSearch(search_point, 1, nearest_index, nearest_distance) > 0) {
            float distance = std::sqrt(nearest_distance[0]);
            point_with_distance.intensity = distance; // Distance to nearest non-power-line point
            updateImage(search_point, distance);
        } else {
            point_with_distance.intensity = -1; // No nearby point found
        }

        power_line_with_distances->push_back(point_with_distance);
    }
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", distance_matrix_).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "map";
    image_pub_->publish(*msg);

    // Publish the detected power line cloud
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*power_line_cloud, output);
    output.header.frame_id = "map"; // Adjust to match your frame
    output.header.stamp = this->now();
    powerline_pub_->publish(output);

    // Step 6: Publish Power Line Distances
    sensor_msgs::msg::PointCloud2 distance_output;
    pcl::toROSMsg(*power_line_with_distances, distance_output);
    distance_output.header.frame_id = "map";
    distance_output.header.stamp = this->now();
    distance_pub_->publish(distance_output);
}

void PowerlineMapper::updateImage(const pcl::PointXYZ &point, float distance) {
    // Convert map coordinates to image pixel coordinates
    int img_x =
        static_cast<int>((point.x - origin_x_) / meters_per_pixel_); // + image_size_.width / 2;
    int img_y =
        static_cast<int>((point.y - origin_y_) / meters_per_pixel_); // + image_size_.height / 2;

    // ✅ Fix: Flip Y-axis since OpenCV uses (0,0) at the top
    img_y = image_size_.height - img_y;

    // Ensure pixel is within bounds before updating
    if (img_x >= 0 && img_x < distance_matrix_.cols && img_y >= 0 &&
        img_y < distance_matrix_.rows) {
        float &current_val = distance_matrix_.at<float>(img_y, img_x);
        current_val = std::min(current_val, distance);
        distance_map[{img_x, img_y}].push_back(distance);
    }
}

void PowerlineMapper::saveGeoTIFF() {
    // Convert `origin_x_` and `origin_y_` to UTM
    double utm_x, utm_y;
    try {
        geometry_msgs::msg::TransformStamped transform_stamped =
            tf_buffer_.lookupTransform("utm", "map", tf2::TimePointZero, tf2::durationFromSec(1.0));

        utm_x = transform_stamped.transform.translation.x + origin_x_;
        utm_y = transform_stamped.transform.translation.y - origin_y_;

        RCLCPP_INFO(this->get_logger(), "✅ Transformed origin (%.2f, %.2f) → UTM (%.2f, %.2f)",
                    origin_x_, origin_y_, utm_x, utm_y);
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF2 Transform Failed: %s", ex.what());
        utm_x = origin_x_;
        utm_y = origin_y_;
    }

    // Get the package path
    std::string package_path = ament_index_cpp::get_package_share_directory("pcl_analysis");
    std::string output_dir = package_path + "/outputs";
    std::string filename = output_dir + "/powerline_distances.tif";

    // Ensure the directory exists
    std::filesystem::create_directories(output_dir);

    // Convert distance_map to a median distance matrix
    cv::Mat median_distance_matrix(image_size_, CV_32F, std::numeric_limits<float>::max());

    int populated_pixels = 0;

    for (const auto &[pixel, distances] : distance_map) {
        int x = pixel.first;
        int y = pixel.second;

        if (!distances.empty()) {
            std::vector<float> sorted_distances = distances;
            std::sort(sorted_distances.begin(), sorted_distances.end());

            // Compute median
            float median_distance;
            size_t size = sorted_distances.size();
            if (size % 2 == 0)
                median_distance =
                    (sorted_distances[size / 2 - 1] + sorted_distances[size / 2]) / 2.0;
            else
                median_distance = sorted_distances[size / 2];

            median_distance_matrix.at<float>(y, x) = median_distance;
            populated_pixels++;
        }
    }

    // Debug: Ensure at least some pixels were populated
    RCLCPP_INFO(this->get_logger(), "Populated %d pixels in the TIFF.", populated_pixels);
    if (populated_pixels == 0) {
        RCLCPP_WARN(this->get_logger(), "Warning: No valid pixels were populated.");
        return;
    }

    // Convert the median matrix to a color image
    cv::Mat color_image(image_size_, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < median_distance_matrix.rows; y++) {
        for (int x = 0; x < median_distance_matrix.cols; x++) {
            float distance = median_distance_matrix.at<float>(y, x);

            if (distance == std::numeric_limits<float>::max())
                continue; // Ignore uninitialized pixels

            cv::Vec3b &pixel = color_image.at<cv::Vec3b>(y, x);

            // Convert m to ft
            if (distance * 3.281 < 3.0) {
                pixel = cv::Vec3b(0, 0, 255); // Red
            } else if (distance * 3.281 < 10.0) {
                pixel = cv::Vec3b(0, 255, 255); // Yellow
            } else {
                pixel = cv::Vec3b(0, 255, 0); // Green
            }
        }
    }

    // Convert BGR → RGB for GDAL compatibility
    cv::Mat color_image_rgb;
    cv::cvtColor(color_image, color_image_rgb, cv::COLOR_BGR2RGB);

    // Write to GeoTIFF
    GDALAllRegister();
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *dataset = driver->Create(filename.c_str(), color_image_rgb.cols,
                                          color_image_rgb.rows, 3, GDT_Byte, nullptr);

    OGRSpatialReference srs;
    srs.importFromEPSG(32610);
    char *wkt = nullptr;
    srs.exportToWkt(&wkt);
    dataset->SetProjection(wkt);
    CPLFree(wkt);

    double geotransform[6] = {utm_x, meters_per_pixel_, 0, utm_y, 0, -meters_per_pixel_};
    dataset->SetGeoTransform(geotransform);

    // Split channels for GeoTIFF writing
    cv::Mat channels[3];
    cv::split(color_image_rgb, channels); // Now it's in RGB format

    for (int i = 0; i < 3; i++) {
        CPLErr err = dataset->GetRasterBand(i + 1)->RasterIO(
            GF_Write, 0, 0, color_image.cols, color_image.rows, channels[i].data, color_image.cols,
            color_image.rows, GDT_Byte, 0, 0);
        if (err != CE_None) {
            RCLCPP_ERROR(this->get_logger(), "Error writing GeoTIFF band %d", i + 1);
        }
    }

    GDALClose(dataset);
    RCLCPP_INFO(this->get_logger(), "✅ Successfully saved powerline distances to: %s",
                filename.c_str());
}