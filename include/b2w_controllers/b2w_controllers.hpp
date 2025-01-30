#ifndef B2W_CONTROLLERS_HPP
#define B2W_CONTROLLERS_HPP

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <onnxruntime_cxx_api.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
// #include <ament_index_cpp/get_package_share_directory.hpp>

class B2WControllers : public rclcpp::Node {
public:
    B2WControllers();

private:
    // Callback functions
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void velocityCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void processOdometry();

    void inference(); 

    // ROS components
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    // ONNX components
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::Session session_;
    const char *input_name_;
    const char *output_name_;

    // Variables to store data
    geometry_msgs::msg::Vector3 base_lin_vel_;
    geometry_msgs::msg::Vector3 base_ang_vel_;
    Eigen::Vector3d projected_gravity_;
    geometry_msgs::msg::Twist cmd_vel_;

    std::vector<double> stored_positions_;
    std::vector<double> stored_velocities_;


    // Order of joints: [FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL_foot_joint, 
    //                   FR_hip_joint, FR_thigh_joint, FR_calf_joint, FR_foot_joint, 
    //                   RL_hip_joint, RL_thigh_joint, RL_calf_joint, RL_foot_joint, 
    //                   RR_hip_joint, RR_thigh_joint, RR_calf_joint, RR_foot_joint]
    Eigen::VectorXd joint_positions_;
    Eigen::VectorXd joint_velocities_;
    Eigen::VectorXd default_joint_positions_;

    Eigen::VectorXd last_actions_;

    Eigen::Matrix3d rotation_matrix_;
    bool odometry_received_;
    std::mutex odometry_mutex_;
};


#endif // B2W_CONTROLLERS_HPP
