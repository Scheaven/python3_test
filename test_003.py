import tensorflow as tf

sess = tf.Session()
input_data = tf.constant([[1, 2, 3], [4, 5, 6]])
print("input",sess.run(input_data))
print('T',sess.run(tf.transpose(input_data)))
# [[1 4]
#  [2 5]
#  [3 6]]
# [[1 2 3]
#  [4 5 6]]
print('perm=[1, 0]',sess.run(tf.transpose(input_data, perm=[1, 0])))
# [[1 4]
#  [2 5]
#  [3 6]]
input_data = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
print('input_data shape: ', sess.run(tf.shape(input_data)))
print('input_data shape: ', sess.run(input_data))
# [1, 4, 3]
output_data = tf.transpose(input_data, perm=[1, 2, 0])
print('output_data shape: ', sess.run(output_data))
print('output_data shape: ', sess.run(tf.shape(output_data)))
# [4, 3, 1]
print(sess.run(output_data))