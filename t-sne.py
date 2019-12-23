import tensorflow as tf
from tensorboard.plugins import projector


embedding_var = tf.Variable(tf.truncated_normal([100, 10]), name='embedding')

with tf.Session() as sess:
    # Create summary writer.
    writer = tf.summary.FileWriter('./graphs/embedding_test', sess.graph)
    # Initialize embedding_var
    sess.run(embedding_var.initializer)
    # Create Projector config
    config = projector.ProjectorConfig()
    # Add embedding visualizer
    embedding = config.embeddings.add()
    # Attache the name 'embedding'
    embedding.tensor_name = embedding_var.name
    # Metafile which is described later
    embedding.metadata_path = './100_vocab.csv'
    # Add writer and config to Projector
    projector.visualize_embeddings(writer, config)
    # Save the model
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, './graphs/embedding_test/embedding_test.ckpt', 1)

writer.close()
