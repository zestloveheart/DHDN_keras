from keras.layers import Input,PReLU,Conv2D,Add,Concatenate
from keras.layers import MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
import keras 
import tensorflow as tf

def init_name_counter():
    name_counter = {}
    name_counter['DUB'] = 0
    name_counter['down'] = 0
    name_counter['up'] = 0
    name_counter['Conv2d_PReLU'] = 0
    name_counter['reconstruction'] = 0
    name_counter['enhancement'] = 0

    return name_counter
name_counter = init_name_counter()

def Conv2d_PReLU(filter):
    def wrapper(inputs):
        with tf.name_scope('Conv2d_PReLU'+str(name_counter['Conv2d_PReLU'])):
            name_counter['Conv2d_PReLU']+=1
            x = Conv2D(filter,3,padding='same')(inputs)
            x = PReLU(shared_axes=[1, 2])(x)
        return x
    return wrapper

def downsampling_block(filter,factor=2):
    def wrapper(inputs):
        with tf.name_scope('down'+str(name_counter['down'])):
            name_counter['down']+=1
            x = Conv2D(filter,3,strides=factor,padding='same')(inputs)
        return x
    return wrapper

def upsampling_block(filter,factor=2):
    # from subpixel import SubpixelConv2D
    from model.subpixel import SubpixelConv2D
    # from subpixel import SubpixelConv2D
    def wrapper(inputs):
        with tf.name_scope('up'+str(name_counter['up'])):
            name_counter['up']+=1
            # the paper is sub-pix interpolation, the implement is different, I haven't study the sub-pix's detail
            # maybe it will decrease the efficient
            x = SubpixelConv2D(upsampling_factor=2)(inputs)
            # x = UpSampling2D(factor)(x)
        return x
    return wrapper

def DUB(filter):
    def wrapper(inputs):
        with tf.name_scope('DUB'+str(name_counter['DUB'])):
            name_counter['DUB']+=1
            level_outputs = []
            level_outputs.append(inputs)
            
            x = Conv2d_PReLU(filter)(inputs)
            x = Conv2d_PReLU(filter)(x)
            x = Add()([inputs,x])
            level_outputs.append(x)
            
            inputs = downsampling_block(filter*2)(x)
            x = Add()([inputs,Conv2d_PReLU(filter*2)(inputs)])
            level_outputs.append(x)

            inputs = downsampling_block(filter*4)(x)
            x = Add()([inputs,Conv2d_PReLU(filter*4)(inputs)])
            x = Conv2D(filter*8,1,padding='same')(x)

            inputs = Concatenate()([level_outputs[-1],upsampling_block(filter*2)(x)])
            x = Conv2D(filter*2,1,padding='same')(inputs)
            x = Add()([x,Conv2d_PReLU(filter*2)(x)])
            x = Conv2D(filter*4,1,padding='same')(x)

            inputs = Concatenate()([level_outputs[-2],upsampling_block(filter)(x)])
            inputs = Conv2D(filter,1,padding='same')(inputs)
            x = Conv2d_PReLU(filter)(inputs)
            x = Conv2d_PReLU(filter)(x)
            x = Add()([inputs,x])
            x = Conv2d_PReLU(filter)(x)
            x = Add()([level_outputs[-3],x])
        return x
    return wrapper

def reconstruction(filter):
    def wrapper(inputs):
        with tf.name_scope('reconstruction'+str(name_counter['reconstruction'])):
            name_counter['reconstruction']+=1
            for _ in range(4):
                x = Conv2d_PReLU(filter)(inputs)
                x = Conv2d_PReLU(filter)(x)
                inputs = Add()([inputs,x])
            x = Conv2d_PReLU(filter)(inputs)
        return x
    return wrapper

def enhancement(filter):
    def wrapper(inputs):
        with tf.name_scope('enhancement'+str(name_counter['enhancement'])):
            name_counter['enhancement']+=1
            x = Conv2D(filter,1,padding='same')(inputs)
            x = Add()([x,Conv2d_PReLU(filter)(x)])
            x = upsampling_block(filter)(x)
        return x
    return wrapper

def DIDN():
    input_channel = 3
    input_shape = (64,64,input_channel)
    init_filter = 128
    DUB_number = 6
    
    inputs = Input(shape=input_shape)
    x = Conv2d_PReLU(init_filter)(inputs)
    x = downsampling_block(init_filter*2)(x)

    # DUBs
    DUB_outputs = []
    for _ in range(DUB_number):
        # x is every DUB's output to reconstruction
        x = DUB(init_filter*2)(x)
        DUB_outputs.append(x)
    
    # I concatenate all DUB_outputs before reconstruction, 
    # which is different from the paper's description 
    # as concatenate all reconstruction's outputs 
    # I don't know how to implement the same architecture as the paper.
    x = Concatenate()(DUB_outputs)
    x = reconstruction(init_filter*2*DUB_number)(x)
    x = enhancement(init_filter*2)(x)
    x = Conv2d_PReLU(input_channel)(x)
    outputs = Add()([inputs,x])

    model = Model(inputs=inputs,outputs=outputs)
    return model

def stats_graph(graph):
    import tensorflow as tf
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def test_flops():
    DIDN()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)

if __name__ == "__main__":
    DIDN()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("E:/TensorBoard",sess.graph)
        writer.close()
