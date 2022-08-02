import re
import numpy as np
from collections import defaultdict
class bpe:
    def __init__(self,corpus,min_freq) -> None:
        self.init_vocab=defaultdict(int)
        for sent in corpus:
            sent=sent.strip()
            if sent!="":
                for w in sent.split(" "):
                    self.init_vocab[w]+=1 
        self.word_vocab = defaultdict(int)
        self.word_vocab = {k:v for k,v in self.init_vocab.items() if v>=min_freq}
        self.word_char_vocab = {" ".join(k):v for k,v in self.word_vocab.items()}
        self.sub_word_pair = defaultdict(int)
        self.char_vocab = defaultdict(int)
        for sent in corpus:
            if sent!="":
                for ch in list(sent):
                    self.char_vocab[ch]+=1
        print(self.word_char_vocab)
        print(self.char_vocab)
    

    def _find_top_subword(self):
        subword_dict=defaultdict(int)
        for wd,freq in self.word_char_vocab.items():
            subword=wd.split()
            for i in range(len(subword)-1):
                subword_dict[subword[i],subword[i+1]]+=freq
        top_subwd_pair = max(subword_dict,key=subword_dict.get)
        top_subword="".join(top_subwd_pair)
        self.sub_word_pair[top_subword]=subword_dict[top_subwd_pair]

        return top_subwd_pair

    def _merge(self,subw_pair):
        bigram = re.escape(" ".join(subw_pair))
       # print(bigram)
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        self.word_char_vocab = {p.sub("".join(subw_pair), w): cnt for w, cnt in self.word_char_vocab.items()}
        
            
        

    def build_subword(self,n_merge):
        for n in range(n_merge):
            top_subword_pair=self._find_top_subword()  
            self._merge(top_subword_pair)      


corpus=open("/home/local/ZOHOCORP/santha-11585/Documents/shakespeare.txt").readlines()
#b=bpe(corpus,1)
#b.build_subword(100)
#print(list(b.word_vocab.items())[:5])
#print(list(b.word_char_vocab.items())[:20])
#print(b.sub_word_pair)

b1=bpe(corpus,10)
b1.build_subword(1000)
print(b1.sub_word_pair)
print(b1.char_vocab)
subword_cnt={**b1.sub_word_pair,**b1.char_vocab}
print(subword_cnt)
print(len(subword_cnt))
total_cnt=np.sum(list(subword_cnt.values()))
print(total_cnt)
subword_ll={k:np.log(v/total_cnt) for k,v in subword_cnt.items()}
print(subword_ll)

        
def viterbi_forward(word,subword_ll):
    best_subw_slices=[None]*(len(word)+1)
    negll=np.zeros(len(word)+1)
    for eow in range(1,len(word)+1):
        negll[eow]=np.inf
        for bow in range(eow):
            t=word[bow:eow]
            print(t)
            if t in subword_ll:
                logl=subword_ll[t]
                print(logl)
                s=negll[bow]-logl
                print(s)
                if s<negll[eow]:
                    negll[eow]=s
                    best_subw_slices[eow]=(bow,eow)
    print(best_subw_slices)
    return best_subw_slices
            
subw_slices=viterbi_forward("whereby",subword_ll)

def viterbi_backward(subword_slice,word):
    subword=[]
    next_slice=subword_slice[-1]
    while next_slice is not None:
        subw=word[next_slice[0]:next_slice[1]]
        subword.append(subw)
        next_slice=subword_slice[next_slice[0]]
    subword.reverse()
    return subword

viterbi_backward(subw_slices,"whereby")

def viterbi(line,subword_ll):
    sub=[]
    for word in line.split():
        print(word)
        subw_slices=viterbi_forward(word,subword_ll)
        sub=sub+viterbi_backward(subw_slices,word)
    print(sub)
line="whereby they live: and though that all at once,"
viterbi(line,subword_ll)




