from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical


def get_classes(dataset):
    #return class list
    return dataset['family_accession'].value_counts().index.tolist()


def get_classes_top1000(dataset):
    #return class list
    return dataset['family_accession'].value_counts()[:1000].index.tolist()


""" 
Creation of a dictionnary mapping each amino acid to its numerical code
"""
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
char_dict = {}
for index, val in enumerate(codes):
    char_dict[val] = index+1

def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
      and rest 4 are categorized as 0.
    """
    encode_list = []
    for row in data['sequence']:
      row_encode = []
      for code in row:
        row_encode.append(char_dict.get(code, 0))
      encode_list.append(np.array(row_encode))
    return encode_list

def process_dataset(df,classes):
    """ 
    - Choose the data which is inside the most frequent selected classes 
    - Apply integer encoding function to each sequence of amino acids 
    - Pad the sequences to make sure they all have the same length
    - One-hot encode the sequences 
    """
    df=df.loc[df['family_accession'].isin(classes)].reset_index()
    df_encoded=integer_encoding(df)
    df_padded_encoded=pad_sequences(df_encoded,maxlen=100,padding='post', truncating='post')
    return df_padded_encoded


def additional_porcess(data):
    one_hot_encode=to_categorical(data)
    return one_hot_encode

def process_labels(data,classes):
    """
    
    """
    data_in_classes=data.loc[data['family_accession'].isin(classes)].reset_index()
    label_encoder=LabelEncoder()
    data_encoded=label_encoder.fit_transform(data_in_classes['family_accession'])
    #data_encoded=data_encoded.toarray()
    data_encoded=to_categorical(data_encoded)
    return data_encoded
