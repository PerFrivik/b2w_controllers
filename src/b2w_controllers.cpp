#include "b2w_controllers/b2w_controllers.hpp"
#include <iostream>
#include <vector>

B2WControllers::B2WControllers()
    : Node("b2w_controllers"),
      env_(ORT_LOGGING_LEVEL_WARNING, "onnx_policy"),
      session_(env_, "policy/policy_blind_b2w.onnx", Ort::SessionOptions()),
      allocator_(),
      input_name_(nullptr),
      output_name_(nullptr),
      joint_positions_(16),
      joint_velocities_(16),
      default_joint_positions_(16),
      last_actions_(16),
      odometry_received_(false) {

    default_joint_positions_ << 0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0;

    // Initialize ONNX input and output names
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_ptr.get(); // Assign the name to input_name_

    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name_ptr.get(); // Assign the name to output_name_

    // Subscribers
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "model/b2w/odometry", 10, std::bind(&B2WControllers::odometryCallback, this, std::placeholders::_1));
    
    velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", 10, std::bind(&B2WControllers::velocityCallback, this, std::placeholders::_1));

    action_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_commands", 10);

    // Timer to ensure 50 Hz frequency
    timer_ = this->create_wall_timer(
    std::chrono::milliseconds(20), std::bind(&B2WControllers::inference, this));
}

void B2WControllers::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    base_lin_vel_ = msg->twist.twist.linear;
    base_ang_vel_ = msg->twist.twist.angular;

    // Extract quaternion and convert to rotation matrix
    Eigen::Quaterniond quat(msg->pose.pose.orientation.w,
                            msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y,
                            msg->pose.pose.orientation.z);
    rotation_matrix_ = quat.toRotationMatrix();

    odometry_received_ = true;

    // process odometry
    processOdometry();
}

void B2WControllers::velocityCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    cmd_vel_ = *msg;
}

void B2WControllers::processOdometry() {

    if (!odometry_received_) {
        RCLCPP_WARN(this->get_logger(), "No odometry data received yet.");
        return;
    }

    std::lock_guard<std::mutex> lock(odometry_mutex_);
    Eigen::Vector3d gravity(0, 0, -9.81);
    projected_gravity_ = rotation_matrix_ * gravity;
    
    RCLCPP_INFO(this->get_logger(), "Processed odometry at 50 Hz");
}

void B2WControllers::inference() {

    std::vector<float> input_data(60, 0.0); 
    input_data[0] = base_lin_vel_.x;
    input_data[1] = base_lin_vel_.y;
    input_data[2] = base_lin_vel_.z;
    input_data[3] = base_ang_vel_.x;
    input_data[4] = base_ang_vel_.y;
    input_data[5] = base_ang_vel_.z;
    input_data[6] = projected_gravity_.x();
    input_data[7] = projected_gravity_.y();
    input_data[8] = projected_gravity_.z();
    input_data[9] = cmd_vel_.linear.x;
    input_data[10] = cmd_vel_.linear.y;
    input_data[11] = cmd_vel_.linear.z;

    // Fill joint positions and velocities
    for (Eigen::Index i = 0; i < joint_positions_.size(); ++i) {
        input_data[12 + i] = joint_positions_[i] - default_joint_positions_[i];
        input_data[28 + i] = joint_velocities_[i];
        input_data[44 + i] = last_actions_[i];
    }

    std::vector<int64_t> input_shape = {1, 60};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    try {
        // Run the ONNX model
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, // Default options
            &input_name_,             // Input names
            &input_tensor,            // Input tensors
            1,                        // Number of inputs
            &output_name_,            // Output names
            1                         // Number of outputs
        );

        // Process the output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        last_actions_ = Eigen::Map<Eigen::VectorXf>(output_data, 16).cast<double>();

        RCLCPP_INFO(this->get_logger(), "Inference output:");
        for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "%.2f", last_actions_(i));
        }
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
    }

    // Publish the actions
    auto msg = std_msgs::msg::Float64MultiArray();
    // msg.data = last_actions_;
    for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
        msg.data.push_back(last_actions_(i));
    }
    action_pub_->publish(msg);

    RCLCPP_INFO(this->get_logger(), "Published actions");
    // print the actions 
    for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "%.2f", last_actions_(i));
    }
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<B2WControllers>());
    rclcpp::shutdown();
    return 0;
}