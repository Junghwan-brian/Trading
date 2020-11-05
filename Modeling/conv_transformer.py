import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    Dense,
    Dropout,
    LayerNormalization,
    Conv2D,
    MaxPool2D,
    Flatten,
    BatchNormalization,
)

tf.random.set_seed(10)


class ConvBlock(Layer):
    def __init__(self, input_shape, m_dim=256):
        super(ConvBlock, self).__init__(name="Conv_block")
        self.conv1 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            input_shape=input_shape,
            kernel_initializer=tf.initializers.he_normal(),
        )
        self.layerNorm1 = LayerNormalization(axis=-1)
        self.conv2 = Conv2D(
            filters=m_dim,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=tf.initializers.he_normal(),
        )
        self.layerNorm2 = LayerNormalization(axis=-1)
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=1)

    def call(self, x, batch_size):
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = tf.nn.swish(x)
        x = self.conv2(x)
        x = self.layerNorm2(x)
        x = tf.nn.swish(x)
        x = self.maxpool(x)
        x = tf.reshape(x, (batch_size, 15, -1))
        return x


class SelfAttention(Layer):
    # m_dim 은 Conv Block에서의 마지막 Conv Channel 수와 같아야 한다.
    def __init__(self, m_dim=256):
        super(SelfAttention, self).__init__(name="self_attention_layer")
        self.m_dim = tf.cast(m_dim, tf.float32)

    def call(self, q, k, v, training=None, mask=None):  # (q,k,v)
        k_T = tf.transpose(k, perm=[0, 1, 3, 2])  # batch,num_heads, depth, seq_len_k
        comp = tf.divide(
            tf.matmul(q, k_T), tf.math.sqrt(self.m_dim)
        )  # batch,num_heads,seq_len_q,seq_len_k
        attention_weights = tf.nn.softmax(
            comp, axis=-1
        )  # batch,num_heads,seq_len_q,seq_len_k
        outputs = tf.matmul(attention_weights, v)  # batch,num_heads, seq_len_q, depth
        return outputs


class MultiHeadAttention(Layer):
    def __init__(self, input_shape, num_heads=8, m_dim=256):
        super(MultiHeadAttention, self).__init__(name="MultiHeadAttention")
        self.num_heads = num_heads
        self.m_dim = m_dim
        assert m_dim % self.num_heads == 0

        self.depth = m_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(m_dim, input_shape=input_shape)
        self.wk = tf.keras.layers.Dense(m_dim, input_shape=input_shape)
        self.wv = tf.keras.layers.Dense(m_dim, input_shape=input_shape)

        self.selfAttn = SelfAttention(m_dim=m_dim)
        self.dense = tf.keras.layers.Dense(m_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, batch_size):
        q = self.wq(q)  # (batch_size, seq_len, m_dim)
        k = self.wk(k)  # (batch_size, seq_len, m_dim)
        v = self.wv(v)  # (batch_size, seq_len, m_dim)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = self.selfAttn(q, k, v)
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.m_dim)
        )  # (batch_size, seq_len_q, m_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, m_dim)
        return output


class TransformerBlock(Layer):
    def __init__(self, input_shape, num_heads=8, m_dim=256):
        super(TransformerBlock, self).__init__(name="Transformer_Block")
        self.multiAttn = MultiHeadAttention(
            input_shape=input_shape, num_heads=num_heads, m_dim=m_dim
        )
        self.dropout1 = Dropout(0.3)
        self.layerNorm1 = LayerNormalization(axis=-1)
        self.dense1 = Dense(m_dim, kernel_initializer=tf.initializers.he_normal())
        self.dense2 = Dense(m_dim, kernel_initializer=tf.initializers.he_normal())
        self.dropout2 = Dropout(0.3)
        self.layerNorm2 = LayerNormalization(axis=-1)

    def call(self, query, key, value, batch_size, training=False):
        context_v = self.multiAttn(query, key, value, batch_size, training=training)
        dropout1 = self.dropout1(context_v, training=training)
        residual_con1 = dropout1 + value
        ln1 = self.layerNorm1(residual_con1)
        dense1 = self.dense1(ln1)
        dense1 = tf.nn.swish(dense1)
        dense2 = self.dense2(dense1)
        dense2 = tf.nn.swish(dense2)
        dropout2 = self.dropout2(dense2, training=training)
        residual_con2 = ln1 + dropout2
        output = self.layerNorm2(residual_con2)
        return output


class ConvTransformer(tf.keras.Model):
    def __init__(self, num_heads=8, m_dim=256):
        super(ConvTransformer, self).__init__(name="Conv_Transformer")
        self.encConvBlock = ConvBlock(input_shape=(-1, 4, 6, 7), m_dim=m_dim)
        self.encTransBlock = TransformerBlock(
            input_shape=(-1, 15, m_dim), num_heads=num_heads, m_dim=m_dim,
        )
        self.decConvBlock = ConvBlock(input_shape=(-1, 4, 6, 5), m_dim=m_dim)
        self.decTransBlock = TransformerBlock(
            input_shape=(-1, 15, m_dim), num_heads=num_heads, m_dim=m_dim,
        )
        self.finalBlock = TransformerBlock(
            input_shape=(-1, 15, m_dim), num_heads=num_heads, m_dim=m_dim,
        )
        self.flat = Flatten()
        self.var_batchNorm1 = BatchNormalization(axis=-1)
        self.var_dense1 = Dense(64, kernel_initializer=tf.initializers.he_normal())
        self.var_dense2 = Dense(64, kernel_initializer=tf.initializers.he_normal())
        self.dense1 = Dense(512, kernel_initializer=tf.initializers.he_normal())
        self.dense2 = Dense(128, kernel_initializer=tf.initializers.he_normal())
        self.dense3 = Dense(3, activation="softmax")

    def call(self, inputs, training=False, mask=None):
        # inputs = (batch,25,12)
        # arr = (batch,24,12) (min_seqs(4),vol_seqs(1),이동평균비율(1),추가정보(6))
        # var = (batch,1, 12)
        batch_size = tf.shape(inputs)[0]
        data, var = tf.split(inputs, [24, 1], axis=1)
        dec_data, enc_data = tf.split(data, [5, 7], axis=-1)
        enc_data = tf.reshape(enc_data, (batch_size, 4, 6, 7))  # batch,4,6,7
        dec_data = tf.reshape(dec_data, (batch_size, 4, 6, 5))  # batch,4,6,5
        encConvOutput = self.encConvBlock(enc_data, batch_size)
        encOutput = self.encTransBlock(
            encConvOutput, encConvOutput, encConvOutput, batch_size, training=training
        )
        decConvOutput = self.decConvBlock(dec_data, batch_size)

        decOutput = self.decTransBlock(
            decConvOutput, decConvOutput, decConvOutput, batch_size, training=training
        )
        decTransOutput = self.finalBlock(
            encOutput, decOutput, decOutput, batch_size, training=training
        )
        flat = self.flat(decTransOutput)

        var1 = self.flat(var)
        var1 = self.var_batchNorm1(var1, training=training)
        var1 = self.var_dense1(var1)  # (batch,64)
        var1 = tf.nn.swish(var1)
        var2 = self.var_dense2(var1)
        var2 = tf.nn.swish(var2)
        var2 = var1 + var2

        concat = tf.concat([var2, flat], axis=-1)

        dense1 = self.dense1(concat)
        dense1 = tf.nn.swish(dense1)

        dense2 = self.dense2(dense1)
        dense2 = tf.nn.swish(dense2)

        output = self.dense3(dense2)
        return output
