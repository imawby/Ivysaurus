from tensorflow import range as tfrange
from tensorflow import newaxis, cast
from tensorflow import math as tfmath
from tensorflow import float32 as tffloat32
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Add, concatenate, Dense, Dropout, Embedding, Flatten, GlobalMaxPooling1D, Layer, LayerNormalization, MultiHeadAttention
import numpy as np

def TransformerModel(sequenceLength, nvocab, nclasses, embedDim):

    dedxInput = Input(shape=(sequenceLength, 1), dtype="int64")
    
    ################################
    # First, embedding
    ################################
    embedded = PositionalEmbedding(sequenceLength, nvocab, embedDim)(dedxInput)
    
    ################################
    # Now transformer
    ################################ 
    numHeads = 2
    denseDim = 32
    
    embedded = TransformerEncoder(embedDim, numHeads, denseDim)(embedded)
    
    ################################
    # Now classify the image
    ################################
    classify = GlobalMaxPooling1D()(embedded)
    classify = Dropout(0.5)(classify)
    
    outputs = Dense(nclasses, activation="softmax")(classify)
    
    ################################
    # Define model
    ################################    
    model = Model(inputs=dedxInput, outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################
# Positional Embedding

class PositionalEmbedding(Layer):
    def __init__(self, sequenceLength, vocabLength, embedDim):
        super().__init__()
        self.sequenceLength = sequenceLength
        self.vocabLength = vocabLength
        self.embedDim = embedDim
        self.tokenEmbeddings = Embedding(input_dim=vocabLength, output_dim=embedDim, mask_zero=True)
        self.positionEmbeddings = self.positional_encoding(sequenceLength, embedDim)

    def call(self, inputs):
        embeddedTokens = self.tokenEmbeddings(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        #embeddedTokens *= tfmath.sqrt(cast(self.embedDim, tffloat32))
        embeddedTokens = embeddedTokens + self.positionEmbeddings[newaxis, : self.sequenceLength, :]
        return embeddedTokens
 
    def compute_mask(self, *args, **kwargs):
        return self.tokenEmbeddings.compute_mask(*args, **kwargs)
    
    def positional_encoding(self, sequenceLength, embedDim):
        depthDim = embedDim/2

        positions = np.arange(sequenceLength).reshape(int(sequenceLength), 1)   # (seq, 1)
        depths = np.arange(depthDim).reshape(1, int(depthDim)) / depthDim       # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

        return cast(pos_encoding, dtype=tffloat32)

    
##########################################################################################################################
##########################################################################################################################
# Attention Block
    
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.attention = MultiHeadAttention(**kwargs)
        self.add = Add()
        self.layerNorm = LayerNormalization()
        
    def call(self, inputs):            
        attentionOutput = self.attention(inputs, inputs, inputs)
        output = self.add([inputs, attentionOutput])
        output = self.layerNorm(output)
        
        return output
    
##########################################################################################################################
##########################################################################################################################
# Transformer Encoder
    
class TransformerEncoder(Layer):
    def __init__(self, embedDim, numHeads, denseDim):
        super().__init__()
        self.embedDim = embedDim
        self.denseDim = denseDim
        self.numHeads = numHeads
        
        self.selfAttention = SelfAttention(num_heads=numHeads, key_dim=embedDim)
        self.denseLayers = Sequential(
            [Dense(denseDim, activation="relu"),
             Dense(embedDim)])
        self.add = Add()
        self.layerNorm = LayerNormalization()
        
    def call(self, inputs):
        attentionOutput = self.selfAttention(inputs)
        output = self.denseLayers(attentionOutput)
        output = self.add([attentionOutput, output])
        output = self.layerNorm(output)
        
        return output
