# Refrect_LiDAR

#lanch
'''
ros2 launch try_navigation reflection_check.launch.py 
'''
#run 
#reflection intensity map
Node(package='try_navigation',
     executable='reflection_intensity_check',
          name='reflection_intensity_map_node',
          output='screen',
          arguments=[],
    ),
