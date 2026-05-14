import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check for GPU
gpu = tf.config.list_physical_devices('GPU')
if gpu:
    print("GPU devices found:", gpu)
    try:
        # Print memory details if possible
        for device in gpu:
            details = tf.config.experimental.get_device_details(device)
            print("  - Device name:", details.get('device_name', 'Unknown'))
            print("  - Device memory:", details.get('memory_limit', 'Unknown'))
    except Exception as e:
        print("Error getting device details:", e)
else:
    print("No GPU devices found. TensorFlow is running on CPU.")

# Run a simple operation to verify
print("Running a simple tensor operation:")
with tf.device('/cpu:0'):  # Force CPU for this test if needed
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b
    print("Result:", c.numpy())