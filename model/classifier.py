from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
import seaborn as sns
import matplotlib.pyplot as plt


### First Model using LSTM
def build_model(optimizer='adam',loss='categorical_crossentropy'):
    """
    - Instanciation of keras tensor with Input()
    - Embedding layer 
    - Bidirectional LSTM 
    - Softmax classifier in Dense Layer

    Output : compiled model
    """
    model=Sequential()
    model.add(Embedding(21, 100, input_length=100))
    model.add(Bidirectional(LSTM(100,kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


### Second model unsing a residual block and convolution
def residual_block(filters, dilatation_rate):
    """
    -data: input
    - filters: convolution filters
    -dilatation_rate: dilation rate
    """
    model=Sequential()
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters, 1, dilation_rate=dilatation_rate, padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))) 
    return model

def protccn_model(input_shape):
    """ 
    - Model using previous function residual_block()
    - Returns the built model
    """
    model=Sequential()
    model.add(Conv1D(128, 1, padding='same'))
    # per-residue representation
    model.add(residual_block(128, 2))
    model.add(residual_block(128, 3))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Flatten())
    model.add(Dense(1000, activation='softmax', kernel_regularizer=l2(0.0001)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=input_shape)
    return model


def train_model(model,x_train,y_train,x_val,y_val,epochs=50,batch_size=256, model_name="lstm"):
    """ 
    - train function for both models 
    - default epochs (50) parameters and batch size (256)
    - using earlystopping to prevent overtraining 
    """
    history=model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(x_val,y_val),callbacks=EarlyStopping(monitor="val_loss",verbose=1,patience=5))
    model.save_weights("model_"+model_name)
    return history

def plot_accuracy_train_val(history):
    """ 
    - functions to visualize accuracy loss for both traing and validation
    """
    accuracy_train=history.history["accuracy"]
    loss_train=history.history["loss"]
    accuracy_val=history.history["val_accuracy"]
    loss_val=history.history["val_loss"]
    x = range(1, len(accuracy_train) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x,accuracy_train, label='Training accuracy')
    sns.lineplot(x,loss_train, label='Training loss')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.lineplot(x, accuracy_val, label='Training loss')
    sns.lineplot(x, loss_val, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

