# The "brain" image from Nav2 wizard step 2: FROM ros:jazzy -> ROS 2 -> Nav2 -> tools (RViz, map) -> environment.
# Build: docker build -t ros2-nav2:jazzy .     (about 1 GB of packages per the infographic; ~6 GB per the talk)
# Not executed here. For Humble, change both "jazzy" strings and use nav2_params.yaml with Humble plugin names.
FROM ros:jazzy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-nav2-bringup \
        ros-jazzy-rviz2 \
        ros-jazzy-tf2-tools \
        ros-jazzy-teleop-twist-keyboard \
        python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
WORKDIR /ws
