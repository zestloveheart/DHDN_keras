from keras.layers import Input,PReLU,Conv2D,Add,Concatenate
from keras.layers import MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
import keras 
import tensorflow as tf

def init_name_counter():
    name_counter = {}
    name_counter['DCR'] = 0
    name_counter['level'] = 0
    name_counter['down'] = 0
    name_counter['up'] = 0
    return name_counter
name_counter = init_name_counter()

def DCR(filter):
    def wrapper(inputs):
        with tf.name_scope('DCR'+str(name_counter['DCR'])):
            name_counter['DCR']+=1
            origin_input = inputs
            for _ in range(2):
                x = Conv2D(filter//2,3,padding='same')(inputs)
                x = PReLU(shared_axes=[1, 2, 3])(x)
                inputs = Concatenate()([inputs,x])
            x = Conv2D(filter,3,padding='same')(inputs)
            x = PReLU(shared_axes=[1, 2, 3])(x)
            x = Add()([origin_input,x])
            x = PReLU(shared_axes=[1, 2, 3])(x)
        return x
    return wrapper

def level_block(filter):
    def wrapper(inputs):
        with tf.name_scope('level'+str(name_counter['level'])):
            for _ in range(2):
                x = DCR(filter)(inputs)
                inputs = Add()([inputs,x])
            name_counter['level']+=1
        return inputs
    return wrapper

def downsampling_block(filter,factor=2):
    def wrapper(inputs):
        with tf.name_scope('down'+str(name_counter['down'])):
            x = MaxPooling2D(factor,factor)(inputs)
            x = Conv2D(filter,3,padding='same')(x)
            x = PReLU(shared_axes=[1, 2, 3])(x)
            name_counter['down']+=1
        return x
    return wrapper

def upsampling_block(filter,factor=2):
    # from subpixel import SubpixelConv2D
    from model.subpixel import SubpixelConv2D
    def wrapper(inputs):
        with tf.name_scope('up'+str(name_counter['up'])):
            x = Conv2D(filter*4,3,padding='same')(inputs)
            x = PReLU(shared_axes=[1, 2, 3])(x)
            # the paper is sub-pix interpolation, the implement is different, I haven't study the sub-pix's detail
            # maybe it will decrease the efficient
            x = SubpixelConv2D(upsampling_factor=2)(x)
            # x = UpSampling2D(factor)(x)
            name_counter['up']+=1
        return x
    return wrapper

def DHDN():
    input_channel = 3
    input_shape = (None,None,input_channel)
    init_filter = 128
    level_number = 3
    
    inputs = Input(shape=input_shape)
    x = Conv2D(init_filter,1,padding='same')(inputs)

    # contracting path
    level_outputs = []
    for i in range(level_number):
        # c is every level's output to expanding path before downsampling
        c = level_block(init_filter*2**i)(x)
        level_outputs.append(c)
        x = downsampling_block(init_filter*2**(i+1),2)(c)

    # bottom level
    c = level_block(init_filter*2**(level_number))(x)
    x = Concatenate()([x,c])

    # expanding path
    for i in range(level_number-1,-1,-1):
        x = upsampling_block(init_filter*2**i,2)(x)
        x = Concatenate()([level_outputs[i],x])
        x = level_block(init_filter*2**(i+1))(x)

    # last level
    x = Conv2D(input_channel,1,padding='same')(x)
    outputs = x

    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    return model

def stats_graph(graph):
    import tensorflow as tf
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def test_flops():
    DHDN()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)

if __name__ == "__main__":
    DHDN()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("E:/TensorBoard",sess.graph)
        writer.close()
