## Info

dict_keys(['pose', 'wall_collision', 'opponent_collisions', 'acceleration', 'velocity', 'progress', 'obstacle', 'dist_goal', 'time', 'checkpoint', 'lap', 'wrong_way', 'rank', 'collision_penalties', 'n_collision', 'reward'])

<class 'numpy.ndarray'>
<class 'bool'>
<class 'list'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'float'>
<class 'int'>
<class 'int'>
<class 'bool'>
<class 'int'>
<class 'numpy.ndarray'>
<class 'int'>
<class 'float'>

{'pose': array([ 1.37210399e+01, -1.96794580e+01,  5.31897174e-02, -1.73809630e-02,
       -2.10672602e-02,  3.11607207e+00]), 'wall_collision': False, 'opponent_collisions': [], 'acceleration': array([0., 0., 0., 0., 0., 0.]), 'velocity': array([ 2.15738462e+00,  2.14236438e-03,  1.46142136e-01, -1.50859292e-01,
       -1.67305062e-01, -2.05158957e-02]), 'progress': 0.3618628067967275, 'obstacle': 0.6549763986894771, 'dist_goal': 0.6388073059991044, 'time': 0.9600000000000005, 'checkpoint': 1, 'lap': 1, 'wrong_way': False, 'rank': 1, 'collision_penalties': array([], dtype=float64), 'n_collision': 0, 'reward': -0.009370673379484043}


## Reward 
foward : reward * 2 if no retuen reward
wall_collision: -=1
(obstacle - 0.5) / 10
frame -= -5e-6






