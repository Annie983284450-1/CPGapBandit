import tensorflow as tf
import time
# import jax
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Simple matrix multiplication to test performance
# print(f'Checkingh jax .... :\n')
# print(jax.devices())
with tf.device('/GPU:0'):  # Or '/CPU:0' to compare performance
    start_time = time.time()
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
    print(f"Matrix multiplication result shape: {c.shape}")
    print(f"Computation time: {time.time() - start_time} seconds")
    print(tf.__version__)
