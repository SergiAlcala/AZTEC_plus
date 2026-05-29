import tensorflow as tf
import datetime

# Define a simple computation graph
x = tf.constant(2.0)
y = tf.constant(3.0)
z = x * y

# Create a summary writer
log_dir = "logs/hello_world"
writer = tf.summary.create_file_writer(log_dir)

with writer.as_default():
    tf.summary.scalar("result", z.numpy(), step=1)

print(f"Log files written to {log_dir}")
