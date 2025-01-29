#include "b2w_controllers/b2w_controllers.hpp"
#include <iostream>
#include <vector>
#include <filesystem> // C++17 feature for paths

B2WControllers::B2WControllers()
    : Node("b2w_controllers"),
      env_(ORT_LOGGING_LEVEL_WARNING, "onnx_policy"),
      allocator_(),
      input_name_("input"),  // Hardcoded input name
      output_name_("output"), // Hardcoded output name
      joint_positions_(16),
      joint_velocities_(16),
      default_joint_positions_(16),
      last_actions_(16),
      odometry_received_(false),
      session_(env_,
               (std::filesystem::path(__FILE__).parent_path() / "policy" / "policy_blind_b2w.onnx").string().c_str(),
               Ort::SessionOptions()) {

    // std::filesystem::path current_file_path(__FILE__); // Path to this source file
    // std::string model_path = (current_file_path.parent_path() / "policy" / "policy_blind_b2w.onnx").string();

    // // Initialize ONNX session with the model
    // session_ = Ort::Session(env_, model_path.c_str(), Ort::SessionOptions());
    

    default_joint_positions_ << 0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0, 0.0;

    // initialize cmd_vel_ with zeros
    cmd_vel_.linear.x = 0.0;
    cmd_vel_.linear.y = 0.0;
    cmd_vel_.linear.z = 0.0;

    cmd_vel_.angular.x = 0.0;
    cmd_vel_.angular.y = 0.0;
    cmd_vel_.angular.z = 0.0;

    // initialize joint_positions_ and velocities with zeros
    joint_positions_ = Eigen::VectorXd::Zero(16);
    joint_velocities_ = Eigen::VectorXd::Zero(16);

    // initialize last acitons with zeros
    last_actions_ = Eigen::VectorXd::Zero(16);

    // // Initialize ONNX input and output names
    // auto input_name_ptr = session_.GetInputNameAllocated(0, allocator_);
    // input_name_ = input_name_ptr.get(); // Assign the name to input_name_

    // auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator_);
    // output_name_ = output_name_ptr.get(); // Assign the name to output_name_

    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_ptr.get(); // Assign the name to input_name_
    RCLCPP_INFO(this->get_logger(), "Input name: %s", input_name_ ? input_name_ : "null");

    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name_ptr.get(); // Assign the name to output_name_
    RCLCPP_INFO(this->get_logger(), "Output name: %s", output_name_ ? output_name_ : "null");


    // Subscribers
    odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "model/b2w/odometry", 10, std::bind(&B2WControllers::odometryCallback, this, std::placeholders::_1));
    
    velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", 10, std::bind(&B2WControllers::velocityCallback, this, std::placeholders::_1));

    action_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_commands", 10);

    // Timer to ensure 50 Hz frequency
    timer_ = this->create_wall_timer(
    std::chrono::milliseconds(20), std::bind(&B2WControllers::inference, this));

    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
}

void B2WControllers::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    base_lin_vel_ = msg->twist.twist.linear;
    base_ang_vel_ = msg->twist.twist.angular;

    printf("Odometry data received\n");

    // Extract quaternion and convert to rotation matrix
    Eigen::Quaterniond quat(msg->pose.pose.orientation.w,
                            msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y,
                            msg->pose.pose.orientation.z);
    rotation_matrix_ = quat.toRotationMatrix();

    odometry_received_ = true;

    RCLCPP_INFO(this->get_logger(), "Odometry data received.");

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

    printf("processing odometry\n");

    std::lock_guard<std::mutex> lock(odometry_mutex_);
    Eigen::Vector3d gravity(0, 0, -9.81);
    projected_gravity_ = rotation_matrix_ * gravity;
    
    RCLCPP_INFO(this->get_logger(), "Processed odometry at 50 Hz");
}

void B2WControllers::inference() {

    // if (!odometry_received_) {
    //     RCLCPP_WARN(this->get_logger(), "No odometry data received yet. aaaaaa");
    //     return;
    // }

    RCLCPP_INFO(this->get_logger(), "Timer triggered.");


    // RCLCPP_INFO(this->get_logger(), "Inference started");

    printf("Inference started\n");

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

    printf("Input data:\n");
    // for (Eigen::Index i = 0; i < input_data.size(); ++i) {
    //     printf("%.2f\n", input_data[i]);
    // }

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
        output_name_ = "output";
        input_name_ = "input";
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
        printf("Inference output:\n");
        for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "%.2f", last_actions_(i));
        }
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
        printf("ONNX Runtime error: %s\n", e.what());
    }

    // std::vector<int> joint_inidices = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    // std::vector<int> wheel_indices = {12, 13, 14, 15}
    // std::vector<int> reorder_indices = {
    //     0, 12, 1, 2,   // FL: calf, foot, hip, thigh
    //     3, 13, 4, 5,   // FR: calf, foot, hip, thigh
    //     6, 14, 7, 8, // RL: calf, foot, hip, thigh
    //     9, 15, 10, 11 // RR: calf, foot, hip, thigh
    // };

    std::vector<int> reorder_indices = {
        0, 1, 2, 3,   // FL: calf, foot, hip, thigh
        4, 5, 6, 7,   // FR: calf, foot, hip, thigh
        8, 9, 10, 11, // RL: calf, foot, hip, thigh
        12, 13, 14, 15 // RR: calf, foot, hip, thigh
    };

    // Create a reordered vector
    Eigen::VectorXd reordered_actions(16);
    for (size_t i = 0; i < reorder_indices.size(); ++i) {
        reordered_actions[i] = last_actions_[reorder_indices[i]];
    }

    // Publish the reordered actions
    auto msg = std_msgs::msg::Float64MultiArray();
    for (Eigen::Index i = 0; i < reordered_actions.size(); ++i) {
        msg.data.push_back(reordered_actions(i));
    }
    action_pub_->publish(msg);


    // RCLCPP_INFO(this->get_logger(), "Published actions");
    printf("Published actions\n");
    // print the actions 
    // for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
    //     RCLCPP_INFO(this->get_logger(), "%.2f", last_actions_(i));
    // }
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<B2WControllers>());
    rclcpp::shutdown();
    return 0;
}