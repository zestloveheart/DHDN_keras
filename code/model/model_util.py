import model.DHDN as DHDN
import model.DIDN as DIDN
import model.DHDN_raw as DHDN_raw
from pathlib import Path
from keras import backend as K

def model_getter(model_name,model_path="",summary = False):
    model_catalog = {"DHDN":DHDN.DHDN,"DIDN":DIDN.DIDN,"DHDN_raw":DHDN_raw.DHDN}
    assert model_name in model_catalog, "the model_name is not exist !"
    model = model_catalog[model_name]()
    if summary:
        model.summary()
    if model_path != "":
        assert Path(model_path).exists(),'can not load the model from the path, maybe is not exist'
        model.load_weights(model_path)
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
    model_getter("DHDN_raw",summary=True)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("E:/TensorBoard",sess.graph)
        writer.close()
