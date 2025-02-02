#include "b2w_controllers/b2w_controllers.hpp"
#include <iostream>
#include <vector>
#include <filesystem> // C++17 feature for paths
#include <chrono>

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
               (std::filesystem::path(__FILE__).parent_path() / "policy" / "policy.onnx").string().c_str(),
               Ort::SessionOptions()) {

    // std::filesystem::path current_file_path(__FILE__); // Path to this source file
    // std::string model_path = (current_file_path.parent_path() / "policy" / "policy_blind_b2w.onnx").string();

    // // Initialize ONNX session with the model
    // session_ = Ort::Session(env_, model_path.c_str(), Ort::SessionOptions());
    

    default_joint_positions_ << 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,
                                -1.0, -1.0, -1.0, -1.0,
                                0.0, 0.0, 0.0, 0.0;

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

    // init h_in_data_ and c_in_data_ with zeros
    h_in_data_ = std::vector<float>(128, 0.0);
    c_in_data_ = std::vector<float>(128, 0.0);

    // initialize last acitons with zeros
    last_actions_ = Eigen::VectorXd::Zero(16);
    reordered_actions_ = Eigen::VectorXd::Zero(16);

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

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>("/world/obstacles/model/b2w/joint_state", 10, std::bind(&B2WControllers::jointStateCallback, this, std::placeholders::_1));

    action_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_commands", 10);

    // Timer to ensure 50 Hz frequency
    // timer_ = this->create_wall_timer(
    // std::chrono::milliseconds(20), std::bind(&B2WControllers::inference, this));

    // Set the 'use_sim_time' parameter to true
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));

    // Create a timer that uses the node's clock (simulation time)
    timer_ = this->create_timer(
        std::chrono::milliseconds(20),  // 50Hz = 20ms
        std::bind(&B2WControllers::inference, this));


    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
    RCLCPP_INFO(this->get_logger(), "B2W Controllers node has been initialized.");
}

void B2WControllers::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // std::lock_guard<std::mutex> lock(odometry_mutex_);
    base_lin_vel_ = msg->twist.twist.linear;
    base_ang_vel_ = msg->twist.twist.angular;

    printf("Odometry data received\n");

    // Extract quaternion and convert to rotation matrix
    Eigen::Quaterniond quat(msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y,
                            msg->pose.pose.orientation.z,
                            msg->pose.pose.orientation.w);
    rotation_matrix_ = quat.toRotationMatrix();

    rotation_matrix_ = rotation_matrix_.inverse();

    // tf2::Quaternion quat;
    // tf2::fromMsg(msg->pose.pose.orientation, quat);

    // tf2::Vector3 gravity_world(0.0, 0.0, -1.0);
    // tf2::Matrix3x3 rotation_matrix(quat);

    // tf2::Vector3 gravity_robot = rotation_matrix * gravity_world;

    // projected_gravity_ = Eigen::Vector3d(gravity_robot.x(), gravity_robot.y(), gravity_robot.z());


    odometry_received_ = true;

    RCLCPP_INFO(this->get_logger(), "Odometry data received.");

    // process odometry
    // processOdometry();
}

void B2WControllers::velocityCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    cmd_vel_ = *msg;
}

void B2WControllers::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {

    stored_positions_ = msg->position;
    stored_velocities_ = msg->velocity;

    std::vector<int> reorder_indices = {
        0, 4, 8, 12,  // FL_hip, FR_hip, RL_hip, RR_hip
        1, 5, 9, 13,  // FL_thigh, FR_thigh, RL_thigh, RR_thigh
        2, 6, 10, 14, // FL_calf, FR_calf, RL_calf, RR_calf
        3, 7, 11, 15  // FL_foot, FR_foot, RL_foot, RR_foot
    };  

    std::vector<double> reordered_positions(16);
    std::vector<double> reordered_velocities(16);

    // Reorganize using simple loops
    for (int i = 0; i < 16; ++i) {
        reordered_positions[i] = stored_positions_[reorder_indices[i]];
        reordered_velocities[i] = stored_velocities_[reorder_indices[i]];
    }

    stored_positions_ = reordered_positions;
    stored_velocities_ = reordered_velocities;

    // print position and velocity information 
    // for (Eigen::Index i = 0; i < stored_positions_.size(); ++i) {
    //     printf("Joint %d: position: %.2f, velocity: %.2f\n", i, stored_positions_[i], stored_velocities_[i]);
    // }

    joint_positions_ = Eigen::Map<Eigen::VectorXd>(stored_positions_.data(), stored_positions_.size());
    joint_velocities_ = Eigen::Map<Eigen::VectorXd>(stored_velocities_.data(), stored_velocities_.size());
}

void B2WControllers::processOdometry() {

    if (!odometry_received_) {
        RCLCPP_WARN(this->get_logger(), "No odometry data received yet.");
        return;
    }

    printf("processing odometry\n");

    // std::lock_guard<std::mutex> lock(odometry_mutex_);
    Eigen::Vector3d gravity(0, 0, -1.0);
    projected_gravity_ = rotation_matrix_ * gravity;

    // print projected gravity
    printf("Projected gravity: %.2f, %.2f, %.2f\n", projected_gravity_.x(), projected_gravity_.y(), projected_gravity_.z());
    
    // RCLCPP_INFO(this->get_logger(), "Processed odometry at 50 Hz");
    printf("Processed odometry at 50 Hz\n");
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
    input_data[11] = cmd_vel_.angular.z;

    // Fill joint positions and velocities
    for (Eigen::Index i = 0; i < joint_positions_.size(); ++i) {
        input_data[12 + i] = joint_positions_[i] - default_joint_positions_[i];
        input_data[28 + i] = joint_velocities_[i];
        input_data[44 + i] = reordered_actions_[i];
    }

    printf("base_line_vel input\n");
    printf("base lin vel x: %.2f\n", input_data[0]);
    printf("base lin vel y: %.2f\n", input_data[1]);
    printf("base lin vel z: %.2f\n", input_data[2]);

    printf("base_ang_vel input\n");
    printf("base ang vel x: %.2f\n", input_data[3]);
    printf("base ang vel y: %.2f\n", input_data[4]);
    printf("base ang vel z: %.2f\n", input_data[5]);

    printf("projected_gravity input\n");
    printf("projected gravity x: %.2f\n", input_data[6]);
    printf("projected gravity y: %.2f\n", input_data[7]);
    printf("projected gravity z: %.2f\n", input_data[8]);

    printf("cmd_vel input\n");
    printf("cmd vel x: %.2f\n", input_data[9]);
    printf("cmd vel y: %.2f\n", input_data[10]);
    printf("cmd vel z: %.2f\n", input_data[11]);

    printf("joint_positions input\n");
    for (Eigen::Index i = 0; i < joint_positions_.size(); ++i) {
        printf("position %.2f\n", input_data[12 + i]);
    }

    printf("joint_velocities input\n");
    for (Eigen::Index i = 0; i < joint_velocities_.size(); ++i) {
        printf("velocity %.2f\n", input_data[28 + i]);
    }

    printf("last_actions input\n");
    for (Eigen::Index i = 0; i < reordered_actions_.size(); ++i) {
        printf("last action %.2f\n", input_data[44 + i]);
    }

    std::array<int64_t, 2> obs_shape = {1, 60};
    std::array<int64_t, 3> hidden_shape  = {1, 1, 128};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value obs_tensor  = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        obs_shape.data(), obs_shape.size()
    );
    Ort::Value h_in_tensor = Ort::Value::CreateTensor<float>(
        memory_info, h_in_data_.data(), h_in_data_.size(),
        hidden_shape.data(), hidden_shape.size()
    );
    Ort::Value c_in_tensor = Ort::Value::CreateTensor<float>(
        memory_info, c_in_data_.data(), c_in_data_.size(),
        hidden_shape.data(), hidden_shape.size()
    );

    std::vector<const char*> input_node_names = {"obs", "h_in", "c_in"};
    std::vector<const char*> output_node_names = {"actions", "h_out", "c_out"};

    // Package these into a vector for session.Run()
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(obs_tensor));
    input_tensors.push_back(std::move(h_in_tensor));
    input_tensors.push_back(std::move(c_in_tensor));

    try {
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), 
            input_tensors.data(), 
            input_tensors.size(),
            output_node_names.data(), 
            output_node_names.size()
        );

        //     [actions, h_out, c_out]
        float* actions_data = output_tensors[0].GetTensorMutableData<float>();
        float* h_out_data   = output_tensors[1].GetTensorMutableData<float>();
        float* c_out_data   = output_tensors[2].GetTensorMutableData<float>();

        // Update h_tensor_ and c_tensor_ with new hidden and cell states
        std::copy(h_out_data, h_out_data + h_in_data_.size(), h_in_data_.begin());
        std::copy(c_out_data, c_out_data + c_in_data_.size(), c_in_data_.begin());

        last_actions_ = Eigen::Map<Eigen::VectorXf>(actions_data, 16).cast<double>();

        RCLCPP_INFO(this->get_logger(), "Inference output:");
        printf("Inference output:\n");
        // for (Eigen::Index i = 0; i < last_actions_.size(); ++i) {
        //     RCLCPP_INFO(this->get_logger(), "%.2f", last_actions_(i));
        // }
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "ONNX Runtime error: %s", e.what());
        printf("ONNX Runtime error: %s\n", e.what());
        printf("im in hereeee");
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
    // for (size_t i = 0; i < reorder_indices.size(); ++i) {
    //     reordered_actions[i] = last_actions_[reorder_indices[i]];
    // }

    reordered_actions = last_actions_;


    // printf("Reordered actions\n");
    // for (Eigen::Index i = 0; i < reordered_actions.size(); ++i) {
    //     printf("aaaaaaaaaaaaaaaaaa %.2f\n", reordered_actions(i));
    // }
    reordered_actions_ = reordered_actions;

    // Scale all non-foot joints (indices 0 to 11)

    reordered_actions.segment(0, 12) *= 0.5;

    // Scale all foot joints (indices 12 to 15)
    reordered_actions.segment(12, 4) *= 5.0;

    reordered_actions.segment(8, 4) = reordered_actions.segment(8, 4) - Eigen::VectorXd::Ones(4);
    // subtract -1 from all calf joints (indices 8 to 11)


    // Publish the reordered actions
    auto msg = std_msgs::msg::Float64MultiArray();
    for (Eigen::Index i = 0; i < reordered_actions.size(); ++i) {
        msg.data.push_back(reordered_actions(i));
    }

    // temporay override to test, set hip to 0.1, thigh to 0.2, calf to 0.3, foot to 0.4

    // for (Eigen::Index i = 0; i < reordered_actions.size(); ++i) {
    //     if (i < 4) {
    //         msg.data.push_back(0.1);
    //     } else if (i < 8) {
    //         msg.data.push_back(0.2);
    //     } else if (i < 12) {
    //         msg.data.push_back(0.3);
    //     } else {
    //         msg.data.push_back(0.4);
    //     }
    // }

    action_pub_->publish(msg);


    // RCLCPP_INFO(this->get_logger(), "Published actions");
    printf("Published actions\n");
    // print actions with printf
    for (Eigen::Index i = 0; i < reordered_actions.size(); ++i) {
        printf("%.2f\n", reordered_actions(i));
    }

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