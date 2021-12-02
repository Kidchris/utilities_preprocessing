""" [summary] : 
                    This file is for main utils we sometimes need to build computer vision project\n
                        It contains many functions that are added as gradualy.\n
                        In this File main functions names are in french but they are wroten in english.\n

        [Functions]:\n 

                    ==> interpolatioon(old, new) :\n
                        this function return the right interpolation based on the image size.\n
                            params[old("shapelike", the image size before resizing)\n
                            new("shapelike", the image size after  resizing)]\n
    """

#######################################################################################
#                           Image preprocessing functions                             #
#######################################################################################
def interpolation(old, new):
    import cv2 as cv
    if old[0]< new[0] or old[1]< new[1]:
        return cv.INTER_CUBIC
    else:
        return cv.INTER_AREA
    



def extract_features(directory, sample_count, conv_base=None):
    """This is for Features Extraction using ConvNet
        It is a function that takes a generator and returns the feature and labels
        of batch_generator. 
        This function aims mainly to to construct a modellike full of conv2d and maxpooling.
        It need to be feeded to a Dense in order to acheive his main goal
    Returns:
        It needs a batch datagenrator provided by the ImageDataGenerator of Keras.
        [type]: [It returns two values (features and labels) ]
    """
    import os
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    
    
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20
    
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, 
    target_size=(150, 150),
    class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

###############################################################################################
#                       Sqlite3 function for saving data to csv                               #
###############################################################################################
    
def save_csv(table, file_name="data.csv", data_base="data.db"):
    """This function take a sql database and save it to a csv file
    Args:
        table (The table where you want to select rows file): 
        file_name (str, optional): [description]. Defaults to "data.csv".
        data_base (str,(*.db file) optional): [description]. Defaults to "data.db".
    """
    import sqlite3
    
    con = sqlite3.connect(data_base)
    cursor = con.cursor()
    csv_file = file_name
    file = open(csv_file, "w+")
    
    try:
        c = cursor.execute(f"SELECT * from {table}")
        des = c.description
        names = [_[0] for _ in des]
        print(names)
        file.write(f"{','.join(names)},\n")
    except Exception as e:
        print(e)

    for raw in cursor.execute(f"SELECT * FROM {table} ORDER BY ID"):
        print(raw)
        for v in range(0, len(raw)):
            file.write(f"{str(raw[v])},")
        file.write("\n")
    file.close()
    con.close()


################################################################################################
#                   Natural Language Processing common used Functions                          #
################################################################################################


def preprocess_text(text, seq=False,lang='english', 
                  vocab_size=10000, print_first=0, 
                  pad=False, pad_type="post"):
    """
    function to preprocess a text and return a array of numbers
        Parameters :
            text : must be an array of texts
            seq  : bool, wether return an array of number representing the text index converted to array)
            lang : str, the language to use default is english
            vocab_size : int, the size of the vocabulary
            print_first : int, the words found in the text
            pad : bool(wether to pad the array or not)
            pad_type : str, the type of padding("post" or "pre")
    """
    import nltk
    import tensorflow as tf
    import string
    
    
    table = str.maketrans("", "", string.punctuation)
    stop_words = nltk.corpus.stopwords.words(lang)
    sentence = []
    for p in text:
        t = p.translate(table)
        splited = t.split()
        filetered = ""
        for w in splited:
            if w not in stop_words:
                filetered += w+ " "
        sentence.append(filetered)
    if seq:
        
        vocab_size = vocab_size
        tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size, oov_token="<OOV>")
        sequences = []
        tokenizer.fit_on_texts(sentence)
        indexes =  tokenizer.word_index
        i = -1
        for k in indexes:
            print(k)
            i +=1
            if i ==print_first:
                break
        sequences = tokenizer.texts_to_sequences(sentence)
        if pad:
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding=pad_type)
            return padded
        else:
            return sequences
    else:
        return sentence



def opener(file_name: str):
    """open a file and read the content of that file.

    Args:
        file_name (str): it takes the file path or the file if in the same directory.

    Returns:
        str: it returns a string containing the whole file.
    """
    text = open(file_name)
    content = text.read()
    text.close()
    return content

def vocabulary():
    """Initialize the vocabulary.

    Returns:
        Hash: It returns a hash table that is intented to map each word to its length
    """
    from collections import Counter
    vocabulary = Counter()
    return vocabulary

def tokenizer(text : str):
    """This function takes a text as an input and tokenize it by the help of nltk

    Args:
        text (str): A string file containing the read content of a file

    Returns:
        List: It returns a list of all the dinstinc words in the text.
    """
    import nltk
    import string
    tokens = text.split()
    stop_words = nltk.corpus.stopwords.words("english")
    table = str.maketrans("", "", string.punctuation)
    tokens = [w.translate(table) for w in tokens if (w not in stop_words and len(w)>1 and w.isalpha())]
    return tokens

def update_vocab(file_name: str, vocab):
    """It updates the vocabulary obtained from counter function

    Args:
        file_name (str): path to the file to open for updatingt the dict variable
        vocab (Count->sklearn counter): [description]
    """
    text = opener(file_name)
    tokens = tokenizer(text)
    vocab.update(tokens)
    
def put_all_together(directory, vocab):
    from os import listdir
    for file in listdir(directory):
        if not file.startswith("cv9"):
            path = directory+"/"+ file
            update_vocab(path, vocab)
            
def save_tokens_to_file(directory, vocab, min_length=2):
    tokens = [k for k,v in vocab.items() if v>min_length]
    l = "\n".join(sorted(tokens))
    f = open(directory, "w")
    f.write((l))
    f.close()

##########################################################################################
#                                                                                        #
##########################################################################################