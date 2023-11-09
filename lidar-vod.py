if __cfg__.DETECT_OBJECT == 'Car':
    #cfg - Config
    
    #Random sampling threshold value (T)
    __cfg__.MAX_POINT_NUMBER = 35
    
    #[Z_MIN, Z_MAX], [Y_MIN, Y_MAX], [X_MIN, X_MAX] indicates 3D space of range Z,H,W respectively
    __cfg__.Z_MIN = -3
    __cfg__.Z_MAX = 1
    __cfg__.Y_MIN = -40
    __cfg__.Y_MAX = 40
    __cfg__.X_MIN = 0
    __cfg__.X_MAX = 70.4
    
    # [VOXEL_X_SIZE, VOXEL_Y_SIZE, VOXEL_Z_SIZE] represents voxel dimesions 𝑣_𝐷,𝑣_(𝐻,) 𝑣_𝑊
    __cfg__.VOXEL_X_SIZE = 0.2
    __cfg__.VOXEL_Y_SIZE = 0.2
    __cfg__.VOXEL_Z_SIZE = 0.4
    __cfg__.VOXEL_POINT_COUNT = 35
    
    #Dividing the 3D spaces into voxel grids
    __cfg__.INPUT_WIDTH = int((__cfg__.X_MAX - __cfg__.X_MIN) / __cfg__.VOXEL_X_SIZE)
    __cfg__.INPUT_HEIGHT = int((__cfg__.Y_MAX - __cfg__.Y_MIN) / __cfg__.VOXEL_Y_SIZE)
    __cfg__.INPUT_DEPTH = int((__cfg__.Z_MAX - __cfg__.Z_MIN) / __cfg__.VOXEL_Z_SIZE)
    __cfg__.LIDAR_COORD = [0, 40, 3]
    __cfg__.FEATURE_RATIO = 2
    __cfg__.FEATURE_WIDTH = int(__cfg__.INPUT_WIDTH / __cfg__.FEATURE_RATIO)
    __cfg__.FEATURE_HEIGHT = int(__cfg__.INPUT_HEIGHT / __cfg__.FEATURE_RATIO)


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated

class ConvMiddleLayer(tf.keras.layers.Layer):
    """
    Convolutional Middle Layer class
    Args:
        out_shape : 4-list, int32, dimensions of the output (batch_size, new_chnnles, height, widht)
    """
    def __init__(self, out_shape):
    super(ConvMiddleLayer, self).__init__()
    self.out_shape = out_shape

    self.conv1 = tf.keras.layers.Conv3D(64, (3,3,3), (2,1,1), data_format="channels_first", padding="VALID")
    self.conv2 = tf.keras.layers.Conv3D(64, (3,3,3), (1,1,1), data_format="channels_first", padding="VALID")
    self.conv3 = tf.keras.layers.Conv3D(64, (3,3,3), (2,1,1), data_format="channels_first", padding="VALID")

    self.bn1 = tf.keras.layers.BatchNormalization(trainable=True)
    self.bn2 = tf.keras.layers.BatchNormalization(trainable=True)
    self.bn3 = tf.keras.layers.BatchNormalization(trainable=True)

    def call(self, input):
        """
        Call Method
        Args:
        input : 5D Tensor, float32, shape=[batch_size, channels(128), Depth(10), Height(400), Width(352)]
        returns:

        """
        # Refer to the paper, section 3 for details 
        out = tf.pad(input, [(0,0)]*2 + [(1,1)]*3)
        out = tf.nn.relu(self.bn1(self.conv1(out)))

        out = tf.pad(out, [(0,0)]*3 + [(1,1)]*2)
        out = tf.nn.relu(self.bn2(self.conv2(out)))

        out = tf.pad(out, [(0,0)]*2 + [(1,1)]*3)
        out = tf.nn.relu(self.bn3(self.conv3(out)))
        return tf.reshape(out, self.out_shape)


    # block 1
    self.conv1_block1, self.bn1_block1 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block1, self.bn2_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block1, self.bn3_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block1, self.bn4_block1 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)

    # block 2
    self.conv1_block2, self.bn1_block2 = self.conv_layer(128, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block2, self.bn2_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block2, self.bn3_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block2, self.bn4_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv5_block2, self.bn5_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)
    self.conv6_block2, self.bn6_block2 = self.conv_layer(128, (3,3),(1,1)), BN(trainable=True)

    # block 3
    self.conv1_block3, self.bn1_block3 = self.conv_layer(256, (3,3),(2,2)), BN(trainable=True)
    self.conv2_block3, self.bn2_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv3_block3, self.bn3_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv4_block3, self.bn4_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv5_block3, self.bn5_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)
    self.conv6_block3, self.bn6_block3 = self.conv_layer(256, (3,3),(1,1)), BN(trainable=True)

    # deconvolutions
    self.deconv_1, self.deconv_bn1 = self.deconv_layer(256, (3,3), (1,1)), BN(trainable=True)
    self.deconv_2, self.deconv_bn2 = self.deconv_layer(256, (2,2), (2,2)), BN(trainable=True)
    self.deconv_3, self.deconv_bn3 = self.deconv_layer(256, (4,4), (4,4)), BN(trainable=True)

    # probability and regression maps
    self.prob_map_conv = self.conv_layer(self.num_anchors_per_cell,(1,1),(1,1))
    self.reg_map_conv = self.conv_layer(7*self.num_anchors_per_cell, (1,1),(1,1))
