import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.backend import int_shape

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Reshape, Conv2D, Concatenate, Dot
from keras.layers import LSTM, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, RepeatVector

from keras.models import Model
from keras.optimizers import Adam

from keras.utils import plot_model

X = pickle.load( open( "../images.pickle", "rb" ) )
Yoh = pickle.load( open( "../word_vec.pickle", "rb" ) )
Y = pickle.load( open( "../captions.pickle", "rb" ) )
X = np.array(X)
Yoh = np.array(Yoh)

X=np.reshape(X, (X.shape[0],40,40,1))
print("X.shape:", X.shape)
print("Yoh.shape:", Yoh.shape)

Tx = np.shape(X[1])
Ty = np.shape(Yoh[1])
machine_vocab = np.zeros((Ty[1],))

def CNN_block(X):
    """
    Implementation of the CNN_block.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- output of the CNN block
    """

    # CONV -> BN -> RELU Block applied to X
    a = Conv2D(8, (7, 7), strides=(1, 1), name='conv0', padding='same')(X)
    a = Activation('selu')(a)

    # MAXPOOL
    a = MaxPooling2D((2, 2), name='max_pool1')(a)

    # CONV -> BN -> RELU Block applied to X
    a = Conv2D(32, (5, 5), strides=(1, 1), name='conv1', padding='valid')(a)
    a = Activation('selu')(a)

    # MAXPOOL
    a = MaxPooling2D((2, 2), name='max_pool2')(a)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    a = Reshape((-1, 32))(a)

    return a

# Defined shared layers as global variables
repeator = RepeatVector(64)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1)
activator = Activation("softmax", name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the CNN, numpy-array of shape (m, n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])

    return context

n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True, dropout = 0.)
output_layer = Dense(len(machine_vocab), activation='softmax')

def captioning_model(X_shape, Ty, n_s):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=X_shape)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###
    # Step 1: Define your CNN
    a = CNN_block(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):

        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model

#model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model = captioning_model(Tx, Ty[0], n_s)

model.summary()

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

m = X.shape[0]
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

model.fit([X, s0, c0], outputs, epochs=1, batch_size=64)

model.evaluate([X, s0, c0], outputs, batch_size=128)

Y_hat = np.array(model.predict([X, s0, c0])).swapaxes(0,1)

plt.imshow(Yoh[8])

plt.imshow(Y_hat[1])

plt.imshow(np.reshape(X[0], (40,40)))

Y[0]
