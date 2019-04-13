special = {'!':' ', '@':' at ', '#':' ', '$':' ', 
           '%':' ', '^':' ', '&':' ', '*':' ', 
           '(':' ', ')':' ', '-':' ', '_':' ',
           '=':' ', '+':' ', '[':' ', ']':' ', 
           '{':' ', '}':' ', '|':' ', ';':' ',
           ':':' ', '"':' ', "'":' ', ',':' ', 
           '<':' ', '.':' ', '>':' ', '/':' ',
           '?':' ', '`':' ', '~':' ','\\':' ',
           '”':' ', '“':' ','…':' ','√':' sqrt ',
           '°':' ','£':' ','π':' pi ','€':' ',
           '−':' ','—':' ','℅':' ','•':' ',
           '±':' plus minus ','²':' two ','³':' three ',
           'µ':' mu ','¹':' one ','¼':' one fourth ', 
           '½':' one half ','¾':' three fourth ',
           'é':'e','ा':' ','í':'i','？':' ',
           'θ':' theta ','，':' ','α':' alpha ',
           '∞':' infinity ','е':'e','β':' beta ', 
           '∆':' delta ','ω':' omega ', 'τ':' tau ',
           'σ':' sigma ','∫':' integral ','ε':' epsilon ',
           'ρ':' rho ','1':' one ', '2':' two ', '3':' three ',
           '÷':' divide ','×':' multiply ','₹':' ',
           '–':' ','4':' four ', '5':' five ', '6':' six ',
           '7':' seven ', '8':' eight ', '9':' nine ',
           '0':' zero '}

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                       "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", 
                       "don't": "do not", "hadn't": "had not", 
                       "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", 
                       "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                       "how'll": "how will", "how's": "how is",  "I'd": "I would", 
                       "I'd've": "I would have", "I'll": "I will", 
                       "I'll've": "I will have","I'm": "I am", 
                       "I've": "I have", "i'd": "i would", 
                       "i'd've": "i would have", "i'll": "i will", 
                       "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not","where's": "where is",
                       "it'd": "it would", "it'd've": "it would have", 
                       "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "what's": "what is", 
                       "ma'am": "madam", "mayn't": "may not", 
                       "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", 
                       "mustn't": "must not", "mustn't've": "must not have", 
                       "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", 
                       "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", 
                       "she'd": "she would", "she'd've": "she would have", 
                       "she'll": "she will", "she'll've": "she will have",
                       "should've": "should have", "shouldn't": "should not", 
                       "shouldn't've": "should not have","she's": "she is",  
                       "so've": "so have","so's": "so as", "this's": "this is",
                       "that'd": "that would", "that'd've": "that would have", 
                       "that's": "that is", "there'd": "there would", 
                       "there'd've": "there would have", "there's": "there is", 
                       "they'd": "they would", "they'd've": "they would have", 
                       "they'll": "they will", "they'll've": "they will have", 
                       "they're": "they are", "they've": "they have", 
                       "to've": "to have", "wasn't": "was not", 
                       "we'd've": "we would have", "we'll": "we will", 
                       "we'll've": "we will have", "we're": "we are", 
                       "we've": "we have", "we'd": "we would", "here's": "here is",
                       "weren't": "were not", "what'll": "what will", 
                       "what'll've": "what will have", "what're": "what are", 
                       "what've": "what have", "when's": "when is", 
                       "when've": "when have", "where'd": "where did",  
                       "where've": "where have", "who'll": "who will", 
                       "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", 
                       "why've": "why have", "will've": "will have",
                       "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would","y'all'd've": "you all would have",
                       "y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", 
                       "you'll": "you will", "you'll've": "you will have", 
                       "you're": "you are", "you've": "you have"}

def build_vocab(text): 
    vocab = {} 
    for sentence in text: 
        sentence = sentence.split()
        for word in sentence: 
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def coverage(vocab, embedding): 
    knownWords = {}
    notKnownWords = {}
    countKnown = 0
    countNotKnown = 0
    for word in vocab.keys():
        if word in embedding:
            knownWords[word] = word
            countKnown += vocab[word]
        else: 
            notKnownWords[word] = vocab[word]
            countNotKnown += vocab[word]
    print('{:.3f} text coverage'.format(countKnown/(countKnown+countNotKnown) * 100))
    return knownWords, notKnownWords

def knownContractions(contraction_mapping, emb): 
    knownCont = []
    for cont in contraction_mapping: 
        if cont in emb: 
            knownCont.append(cont)
    for cont in knownCont:
        del contraction_mapping[cont]
    return knownCont

def cleanCont(text, emb): 
    knownContractions(contraction_mapping, emb)
    specials = ["’", "‘", "´", "`"]
    for s in specials: 
        text = text.replace(s, "'")
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(' ')])
    return text

def clean_characters(sentence, emb):
    for char in sentence: 
        if char in special.keys():
            sentence = sentence.replace(char, special[char])
    removed_spaces = ''
    count = 0
    for char in sentence:
        if char != ' ':
            removed_spaces += char
            count = 0
        else:
            if count < 1: 
                removed_spaces += char
                count = 1
    return removed_spaces.replace('  ', ' ')

mispell_dict = {}
def initworddict():
    with open('found.txt', encoding = 'latin') as f:
        temp = f.readline().split(',')
        for word in temp:
            try:
                word = word.replace('"','').replace('\'', '').split(':')
                mispell_dict[word[0].replace('\'','').lower()] = word[1].replace('\'','').lower()
            except:
                print(word)
                break

def correct_spelling(sent):
    for word in sent.split():
        if word in mispell_dict:
            sent = sent.replace(word, mispell_dict[word])
    return sent
