double_type_names = [
    'heart_rate',
    'wrist_accelerometer_x',
    'wrist_accelerometer_y',
    'wrist_accelerometer_z',
    'wrist_gyroscope_x',
    'wrist_gyroscope_y',
    'wrist_gyroscope_z',
    'wrist_magnetometer_x',
    'wrist_magnetometer_y',
    'wrist_magnetometer_z',
    'chest_accelerometer_x',
    'chest_accelerometer_y',
    'chest_accelerometer_z',
    'chest_gyroscope_x',
    'chest_gyroscope_y',
    'chest_gyroscope_z',
    'chest_magnetometer_x',
    'chest_magnetometer_y',
    'chest_magnetometer_z',
    'ankle_accelerometer_x',
    'ankle_accelerometer_y',
    'ankle_accelerometer_z',
    'ankle_gyroscope_x',
    'ankle_gyroscope_y',
    'ankle_gyroscope_z',
    'ankle_magnetometer_x',
    'ankle_magnetometer_y',
    'ankle_magnetometer_z',
]

target_column = 'activity_id'
default_drop_col = 'user_id'

target_values = [
    'standing',
    'running',
    'sitting',
    'lying',
    'rope_jumping',
    'walking',
    'cycling',
    'descending_stairs',
    'ascending_stairs'
]

split_dataframe_weight = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1
]

knnr_train_test_weight = [
    0.7,
    0.3
]
