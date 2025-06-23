# Refrect_LiDAR

#lanch \n
ros2 launch try_navigation reflection_check.launch.py  \n

#run  \n
#reflection intensity map \n
Node(package='try_navigation', \n
     executable='reflection_intensity_check', \n
          name='reflection_intensity_map_node', \n
          output='screen', \n
          arguments=[], \n
    ), \n
