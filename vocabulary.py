import pickle



class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token
    OOV_Token = 3   # Out Of Vocabulary token
    SPACE = 4 # Vocabulary token for SPACE
    
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {Vocabulary.PAD_token: "PAD", Vocabulary.SOS_token: "SOS", Vocabulary.EOS_token: "EOS", 
                            Vocabulary.OOV_Token:'OOV', Vocabulary.SPACE : "SPACE"}
        self._num_constants = len(self.index2token) 
        
        self.num_tokens = len(self.index2token)
        self.num_sentences = 0
        self.longest_sentence = 0
        self.frequent = None

    def add_token(self, token):
        if token not in self.token2index:
            # First entry of word into vocabulary
            self.token2index[token] = self.num_tokens
            self.token2count[token] = 1
            self.index2token[self.num_tokens] = token
            self.num_tokens += 1
        else:
            # Word exists; increase word count
            self.token2count[token] += 1

    def add_token_list(self, token_list):
        sentence_len = 0
        for token in token_list:
            sentence_len += 1
            self.add_token(token)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_token(self, index):
        if self.frequent:
            return self.frequent['index2token'].get(index, None)
        else:
            return self.index2token.get(index, None)


    def to_index(self, word):
        if self.frequent:
            token =  self.frequent['token2index'].get(word, None)
            return token if token else self.OOV_Token
        else:
            token =  self.token2index.get(word, None)
            return token if token else self.OOV_Token

    def token_count(self, token):
        return self.token2count.get(token, None)

    def truncate_vocabulary(self, frequency=5):
        sorted_vocab = { k : v for k, v in sorted(self.token2count.items(), key= lambda item : item[1], reverse=True)}
        token2count =  {k: v for k, v in sorted_vocab.items() if v > frequency}
        token2index =  {}#{k: self.token2index[k]  for k, v in token2count.items() }
        index2token = {}
        i = self._num_constants
        for k, v in token2count.items():
            token2index[k] = i
            index2token[i] = k
            i += 1

        #index2token =  {v: k for k, v in token2index.items() }
        
        index2token[self.PAD_token] = "PAD"
        index2token[self.SOS_token] = "SOS"
        index2token[self.EOS_token] = "EOS"
        index2token[self.OOV_Token] = "OOV"
        index2token[self.SPACE] = "SPACE"

        self.frequent = {'token2index':token2index, 'index2token':index2token}
    
    def len_frequent_vocab(self):
        return len(self.frequent['index2token']) if self.frequent else None

    def __len__(self):
        return len(self.index2token)

    def save_data(self, path):
        data = {'frequent': self.frequent,
                'name': self.name,
                'token2index': self.token2index,
                'token2count': self.token2count,
                'index2token': self.index2token,
                'num_constants': self._num_constants,
                'num_tokens': self.num_tokens,
                'num_sentences': self.num_sentences,
                'longest_sentence': self.longest_sentence
            
        }
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()

    def load_data(self, path):
        file =open(path, 'rb')
        data = pickle.load(file)
        file.close()
        self.frequent = data['frequent']
        self.name = data['name']
        self.token2index = data['token2index']
        self.token2count = data['token2count']
        self.index2token = data['index2token']
        self._num_constants = data['num_constants']
        
        self.num_tokens = data['num_tokens']
        self.num_sentences = data['num_sentences']
        self.longest_sentence = data['longest_sentence']