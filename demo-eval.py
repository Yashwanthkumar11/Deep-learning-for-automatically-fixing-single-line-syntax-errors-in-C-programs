import sys
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from vocabulary import Vocabulary

class Evaluate():
    def __init__(self):
        self._model_path = 'model'
        self._load_model()
        self._input_vocabulary = Vocabulary('')
        self._input_vocabulary.load_data('input_vocabulary.pickle')
        self._target_vocabulary = Vocabulary('')
        self._target_vocabulary.load_data('target_vocabulary.pickle')
        self._load_configs()

    def _load_configs(self):
        self.num_encoder_tokens = self._input_vocabulary.len_frequent_vocab()
        self.num_decoder_tokens = self._target_vocabulary.len_frequent_vocab()
        self.max_encoder_seq_length =  100 
        self.max_decoder_seq_length =  100

        

    def _load_model(self):
        self._model = keras.models.load_model(self._model_path)
        latent_dim = 256
        encoder_inputs = self._model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self._model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self._model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
        decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4_")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self._model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self._model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


    def _convert_to_list(self, tokens):
        tokens = [ast.literal_eval(src) for src in tokens]
        return list(tokens)
    
    def load_dataset(self, path):
        df  = pd.read_csv(path)
        sourceLineTokens = df['sourceLineTokens']
        input_texts = self._convert_to_list(sourceLineTokens)

        encoder_input_data = np.zeros((len(input_texts), 
                                    self.max_encoder_seq_length, self.num_encoder_tokens), 
                                    dtype="float32")
        for i, input_text in enumerate(input_texts):
            for t, char in enumerate(input_text[:self.max_encoder_seq_length]):
                encoder_input_data[i, t, self._input_vocabulary.to_index(char)] = 1.0
            
            if len(input_text) < self.max_encoder_seq_length:
                encoder_input_data[i, t + 1 :, self._input_vocabulary.PAD_token] = 1.0
        return (input_texts, encoder_input_data)

    
    
    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character. target_token_index["\t"]
        target_seq[0, 0, Vocabulary.SOS_token] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self._target_vocabulary.to_token(sampled_token_index) #reverse_target_char_index[sampled_token_index]
            decoded_sentence.append(sampled_char)#decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character. "\n"
            if sampled_char == self._input_vocabulary.to_token(Vocabulary.EOS_token) or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        decoded_sentence = list([token for token in decoded_sentence 
                         if token != self._target_vocabulary.to_token(Vocabulary.PAD_token)])
        return decoded_sentence
    
    def evaluate(self, encoded_data):
        decoded_tokens = []
        
        for i in range(len(encoded_data)):
            try:
                input_seq = encoded_data[i :  i + 1]
                decoded_token = self.decode_sequence(input_seq)
                decoded_tokens.append(decoded_token)
            except Exception as e:
                print(f'[EXCEPTION] --> {e}')
        
        return decoded_tokens






if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    evaluater = Evaluate()
    texts, encoded = evaluater.load_dataset(input_csv)
    print('Text Samples Found: ', len(texts))
    print('encoded shape: ', encoded.shape)
    print('Evaluation in progress - Please Wait')
    decoded = evaluater.evaluate(encoded)
    print('decoded length: ', len(decoded))

    texts = [str(src) for src in texts]
    decoded = [str(src) for src in decoded]
    output = {'sourceLineTokens':texts, 'targetLineTokens':decoded} 
    df = pd.DataFrame(output)
    df.to_csv(output_csv, index=False)