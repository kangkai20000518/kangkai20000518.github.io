---
layout: post
title: Fake News Classification
---
Rampant misinformation—often called “fake news”—is one of the defining features of contemporary democratic life. In this Blog Post, I will develop and assess a fake news classifier using Tensorflow.


## Data Source
My data is from the article

Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).
I accessed it from Kaggle. I have done a small amount of data cleaning for you already, and performed a train-test split.

## §1. Acquire Training Data


```python
import pandas as pd
import re
import plotly.io as pio
import numpy as np
import string
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import warnings
import tensorflow as tf
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
warnings.filterwarnings("ignore")
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
df=pd.read_csv(train_url)
df.head()
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\kangk\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## §2. Make a Dataset

Write a function called make_dataset. This function should do two things:

Remove stopwords from the article text and title. A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but.”
Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), and the output consist only of the fake column. 


```python
stop = stopwords.words('english')
le = LabelEncoder()
df["fake"] = le.fit_transform(df["fake"])
num_title = len(df["fake"].unique())

def make_dataset(data):
    data['title'] = data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data=tf.data.Dataset.from_tensor_slices((
    {
     "title":data[["title"]],
     "text":data[["text"]]
    },{
     "fake":data["fake"]   
    }))
    return data
```


```python
data=make_dataset(df)
```

### Validation Data

split of 20% of it to use for validation.


```python
data = data.shuffle(buffer_size = len(data))

train_size = int(0.8*len(data))

train = data.take(train_size).batch(20)
val = data.skip(train_size).batch(20)
print(len(train), len(val))
```

    898 225
    

### Base Rate


```python
base_rate=len(df[df["fake"]==1])/len(df)
base_rate
```




    0.522963160942581



In this data set there are 0.5229 are fake news, which means our data set is pretty balanced.

### TextVectorization


```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"]))

```

    WARNING:tensorflow:AutoGraph could not transform <function <lambda> at 0x000001FC861623A0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module 'gast' has no attribute 'Index'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING: AutoGraph could not transform <function <lambda> at 0x000001FC861623A0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module 'gast' has no attribute 'Index'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    

## §3. Create Models

### 1. first model use only the article title as an input.


```python
title_input = keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)
```


```python
title_features = title_vectorize_layer(title_input) # apply this "function TextVectorization layer" to title_input
title_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)
```


```python
title_features= layers.Dense(32, activation='relu')(title_features)
output1 = layers.Dense(num_title , name="fake")(title_features) 
```


```python
model1 = keras.Model(
    inputs = title_input,
    outputs = output1
)
```


```python
model1.summary()
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    title (InputLayer)           [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization (TextVect (None, 500)               0         
    _________________________________________________________________
    embedding (Embedding)        (None, 500, 2)            4000      
    _________________________________________________________________
    dropout (Dropout)            (None, 500, 2)            0         
    _________________________________________________________________
    global_average_pooling1d (Gl (None, 2)                 0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 2)                 0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                96        
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    fake (Dense)                 (None, 2)                 66        
    =================================================================
    Total params: 5,218
    Trainable params: 5,218
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history1 = model1.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```


```python
from matplotlib import pyplot as plt
plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x1fc94ad5430>]




![output_23_1.png]({{ site.baseurl }}/images/output_23_1.png)     
    


### 2. In the second model use only the article text as an input.


```python
text_input = keras.Input(
    shape=(1,),
    name = "text", # same name as the dictionary key in the dataset
    dtype = "string"
)
```


```python
text_features = title_vectorize_layer(text_input) # apply this "function TextVectorization layer" to text_input
text_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
```


```python
text_features= layers.Dense(32, activation='relu')(text_features)
output2 = layers.Dense(num_title , name="fake")(text_features) 
```


```python
model2 = keras.Model(
    inputs = text_input,
    outputs = output2
)
```


```python
model2.summary()
```

    Model: "functional_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    text (InputLayer)            [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization (TextVect (None, 500)               0         
    _________________________________________________________________
    embedding (Embedding)        (None, 500, 2)            4000      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500, 2)            0         
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 2)                 0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2)                 0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                96        
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    fake (Dense)                 (None, 2)                 66        
    =================================================================
    Total params: 5,218
    Trainable params: 5,218
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history2 = model2.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```


```python
from matplotlib import pyplot as plt
plt.plot(history2.history["accuracy"])
plt.plot(history2.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x1fc9623d100>]




![output_32_1.png]({{ site.baseurl }}/images/output_32_1.png)     

    


### 3.In the third model use both the article title and the article text as input.


```python
title_input = keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)
text_input = keras.Input(
    shape=(1,),
    name = "text", # same name as the dictionary key in the dataset
    dtype = "string"
)
```


```python

title_features = title_vectorize_layer(title_input)# apply this "function TextVectorization layer" to title_input
title_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding1")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)

text_features = title_vectorize_layer(text_input)# apply this "function TextVectorization layer" to text_input
text_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding2")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
```


```python
main = layers.concatenate([title_features, text_features], axis = 1)
```


```python
main = layers.Dense(32, activation='relu')(main)
output = layers.Dense(num_title , name="fake")(main) 
```


```python
model3 = keras.Model(
    inputs = [title_input,text_input],
    outputs = output
)
```


```python
model3.summary()
```

    Model: "functional_5"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    title (InputLayer)              [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    text (InputLayer)               [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    text_vectorization (TextVectori (None, 500)          0           title[0][0]                      
                                                                     text[0][0]                       
    __________________________________________________________________________________________________
    embedding1 (Embedding)          (None, 500, 2)       4000        text_vectorization[2][0]         
    __________________________________________________________________________________________________
    embedding2 (Embedding)          (None, 500, 2)       4000        text_vectorization[3][0]         
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 500, 2)       0           embedding1[0][0]                 
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 500, 2)       0           embedding2[0][0]                 
    __________________________________________________________________________________________________
    global_average_pooling1d_2 (Glo (None, 2)            0           dropout_4[0][0]                  
    __________________________________________________________________________________________________
    global_average_pooling1d_3 (Glo (None, 2)            0           dropout_6[0][0]                  
    __________________________________________________________________________________________________
    dropout_5 (Dropout)             (None, 2)            0           global_average_pooling1d_2[0][0] 
    __________________________________________________________________________________________________
    dropout_7 (Dropout)             (None, 2)            0           global_average_pooling1d_3[0][0] 
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 32)           96          dropout_5[0][0]                  
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 32)           96          dropout_7[0][0]                  
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 64)           0           dense_4[0][0]                    
                                                                     dense_5[0][0]                    
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 32)           2080        concatenate[0][0]                
    __________________________________________________________________________________________________
    fake (Dense)                    (None, 2)            66          dense_6[0][0]                    
    ==================================================================================================
    Total params: 10,338
    Trainable params: 10,338
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history3 = model3.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```


```python
from matplotlib import pyplot as plt
plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x1fc99c57610>]




![output_42_1.png]({{ site.baseurl }}/images/output_42_1.png) 
    


After finished all three model I would  recommend us the title and text as input since it has the best peroformence without over-fitting.

## §4. Model Evaluation


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test=pd.read_csv(test_url)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>420</td>
      <td>CNN And MSNBC Destroy Trump, Black Out His Fa...</td>
      <td>Donald Trump practically does something to cri...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14902</td>
      <td>Exclusive: Kremlin tells companies to deliver ...</td>
      <td>The Kremlin wants good news.  The Russian lead...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>322</td>
      <td>Golden State Warriors Coach Just WRECKED Trum...</td>
      <td>On Saturday, the man we re forced to call  Pre...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16108</td>
      <td>Putin opens monument to Stalin's victims, diss...</td>
      <td>President Vladimir Putin inaugurated a monumen...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10304</td>
      <td>BREAKING: DNC HACKER FIRED For Bank Fraud…Blam...</td>
      <td>Apparently breaking the law and scamming the g...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test=make_dataset(test)
test=test.batch(20)
```


```python
model3.evaluate(test)
```

    1123/1123 [==============================] - 2s 2ms/step - loss: 0.0295 - accuracy: 0.9929
    




    [0.029513025656342506, 0.9929172992706299]



We can see that our final model accuracy is about 99%!

## §5. Embedding Visualization


```python
weights = model3.get_layer('embedding2').get_weights()[0] # get the weights from the embedding layer
vocab = title_vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```


```python
import plotly.express as px 
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                # size_max = 2,
                 hover_name = "word")

fig.show()
```

{% include figure_hw4.html %}

From above we can find out that there is only one cluster in this case. And the words show up in the center is the most common words in the news like **ality,wants, results.** but at the same time there are also some words like **gop,obamas and trumps**. Those words can be indicate is fake news or not.