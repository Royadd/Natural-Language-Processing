import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
Trigram Language Models
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    Given a sequence, this function returns a list of n-grams, where each n-gram is a Python tuple.
    """
    sequence=["START"]*max(1,n-1)+sequence+["STOP"]
    if n==1:
        return [tuple([s]) for s in sequence]
    ngrams=[tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.totalwords=0
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populates dictionaries of unigram, bigram,
        and trigram counts. 
        """
        if len(self.lexicon)==0: return
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        for sentence in corpus:
            self.totalwords+=len(sentence)+1
        
            unigrams=get_ngrams(sentence,1)
            for gram in unigrams:
                self.unigramcounts[gram]=self.unigramcounts[gram]+1 if gram in self.unigramcounts else 1
            bigrams=get_ngrams(sentence,2)
            for gram in bigrams:
                self.bigramcounts[gram]=self.bigramcounts[gram]+1 if gram in self.bigramcounts else 1
            trigrams=get_ngrams(sentence,3)
            for gram in trigrams:
                self.trigramcounts[gram]=self.trigramcounts[gram]+1 if gram in self.trigramcounts else 1
            
        return

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[:2]==('START','START'):
            return self.trigramcounts[trigram]/self.unigramcounts[('START',)]
        if trigram[:2] not in self.bigramcounts:
            return 1.0/len(self.lexicon)
        return self.trigramcounts[trigram]/self.bigramcounts[tuple(list(trigram)[:2])] if trigram in self.trigramcounts else 0.0

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram]/self.unigramcounts[tuple(list(bigram)[:1])] if bigram in self.bigramcounts else 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """

        if unigram==('START',):
            return 1.0
        return self.unigramcounts[unigram]/self.totalwords if unigram in self.unigramcounts else 0.0

    def generate_sentence(self,t=20): 
        """
        Generates a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:3])+lambda3*self.raw_unigram_probability((trigram[2],))
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        trigrams=get_ngrams(sentence,3)
        return sum([math.log2(self.smoothed_trigram_probability(trigram)) for trigram in trigrams])

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        l=0
        M=0
        for sentence in corpus:
            l+=self.sentence_logprob(sentence)
            M+=len(sentence)+1
        l=l/M
        return math.pow(2,-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total+=1
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            # .. 
            if pp1<pp2: correct+=1
    
        for f in os.listdir(testdir2):
            total+=1
            pp2= model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            # .. 
            if pp2<pp1: correct+=1
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)
    
    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    print(acc)
