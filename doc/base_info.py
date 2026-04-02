'''

基本信息：范围

actuators:
0 act_OA_x_right trnid [ 1 -1]
1 act_OA_y_right trnid [ 3 -1]
2 act_OB_x_right trnid [ 5 -1]
3 act_OB_y_right trnid [ 7 -1]
4 act_OA_x_left trnid [ 9 -1]
5 act_OA_y_left trnid [11 -1]
6 act_OB_x_left trnid [13 -1]
7 act_OB_y_left trnid [15 -1]



ctrlrange:
0 act_OA_x_right [-1.0053  1.089 ]
1 act_OA_y_right [-0.28  0.28]
2 act_OB_x_right [-1.3543754   0.56548667]
3 act_OB_y_right [-0.28  0.28]
4 act_OA_x_left [-1.0053  1.089 ]
5 act_OA_y_left [-0.28  0.28]
6 act_OB_x_left [-1.3543754   0.56548667]
7 act_OB_y_left [-0.28  0.28]








观测输入：

Part 1: qpos_act (8维) - 执行器驱动的关节角度

    obs[0:8] = [
    0:  OA_active_x_right   (弧度)
    1:  OA_active_y_right   (弧度)
    2:  OB_active_x_right   (弧度)
    3:  OB_active_y_right   (弧度)
    4:  OA_active_x_left    (弧度)
    5:  OA_active_y_left    (弧度)
    6:  OB_active_x_left    (弧度)
    7:  OB_active_y_left    (弧度)
    ]


Part 2: filtered_force (6维) - 力传感器读数


    obs[8:14] = [
      8:   Fx_right          (单位: N, 或仿真中的归一化力)
      9:   Fy_right          
      10:  Fz_right
      11:  Fx_left           
      12:  Fy_left
      13:  Fz_left
    ]

    来源：
    - sensor_F_right 和 sensor_F_left (XML中定义)
    - 经过EMA低通滤波后的值


Part 3: obj_pose (7维) - 物体位姿


    obs[14:21] = [
      14:  obj_x             (物体中心X坐标, 米)
      15:  obj_y             (物体中心Y坐标, 米)
      16:  obj_z             (物体中心Z坐标, 米)
      17:  obj_qx            (物体方向四元数X)
      18:  obj_qy            (物体方向四元数Y)
      19:  obj_qz            (物体方向四元数Z)
      20:  obj_qw            (物体方向四元数W)
    ]

    来源：data.qpos[:7]


Part 4: rel_pos (6维) - 指尖到物体的相对位置


    obs[21:27] = [
      21:  rel_x_right = obj_x - tip_x_right
      22:  rel_y_right = obj_y - tip_y_right
      23:  rel_z_right = obj_z - tip_z_right
      24:  rel_x_left  = obj_x - tip_x_left
      25:  rel_y_left  = obj_y - tip_y_left
      26:  rel_z_left  = obj_z - tip_z_left
    ]

    物理含义：从指尖看向物体的方向向量（未归一化）









输出动作:

obs[0:8] = [
  data.qpos[7],    # OA_active_x_right   (actuator[0] → qpos[7])
  data.qpos[9],    # OA_active_y_right   (actuator[1] → qpos[9])
  data.qpos[11],   # OB_active_x_right   (actuator[2] → qpos[11])
  data.qpos[13],   # OB_active_y_right   (actuator[3] → qpos[13])
  data.qpos[15],   # OA_active_x_left    (actuator[4] → qpos[15])
  data.qpos[17],   # OA_active_y_left    (actuator[5] → qpos[17])
  data.qpos[19],   # OB_active_x_left    (actuator[6] → qpos[19])
  data.qpos[21],   # OB_active_y_left    (actuator[7] → qpos[21])
]



















'''