import tensorflow as tf
import tensorflow.keras.applications as models
import datetime

# Load a pre-trained model (e.g., MobileNetV2)
model = models.MobileNetV2()

# Function to compute FLOPS
def get_flops(model):
    # Convert model to ConcreteFunction
    concrete_function = tf.function(lambda inputs: model(inputs))
    concrete_function = concrete_function.get_concrete_function(
        tf.TensorSpec([1] + list(model.input_shape[1:]), model.input.dtype)
    )

    # Convert ConcreteFunction to frozen function
    frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess=tf.compat.v1.Session(),
        input_graph_def=concrete_function.graph.as_graph_def(),
        output_node_names=[out.op.name for out in concrete_function.outputs]
    )

    # Profile FLOPS
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        frozen_func,
        options=options
    )

    return flops.total_float_ops if flops is not None else 0

# Enable TensorFlow V1 compatibility for graph-based profiling
tf.compat.v1.disable_eager_execution()

# Measure FLOPS
flops = get_flops(model)

# Set up TensorBoard logging
log_dir = "/home/sergi_alcala/sergi_data/CLEAN_AZTEC_Extension/Framework/logs/flops"
writer = tf.summary.create_file_writer(log_dir)

with writer.as_default():
    tf.summary.scalar("FLOPS", flops, step=1)

print(f"Model FLOPS: {flops}")
print(f"Log files written to {log_dir}")
