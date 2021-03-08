# Neural-Language-Model

A Language Model (LM) is the task of predicting what words come next. Informally, it is the probability distribution of the next word when the sequence of words is given.

There are primarily two types of language models: 
1)  Statistical Language Models (N-Gram, Exponential, Continuous Space)
2)  Neural Language Models (RNN, LSTM)


Perplexity: 
In Language model, perplexity is used to measure how well that language model is predicting the next word. It is the normalized inverse probability of the test set.
In mathematically, the equation of perplexity is,
PP(p) = 2H(p) = 2**(−Σ𝑝(𝑥)log2𝑝(𝑥))


Some Common Examples of Language Models
1) Machine Translation
2) Speech Recognization
3) Sentiment Analysis
4) Text Suggestions


In this example I built a Gated Recurrent Unit (GRU) Language Model.
1) Programming Language: Python
2) Libraries: PyTorch
3) LM Type: GRU RNN 
4) Dataset used: Wikitext-2


Inspired by:
1) https://medium.com/@florijan.stamenkovic_99541/rnn-language-modelling-with-pytorch-packed-batching-and-tied-weights-9d8952db35a9
2) https://github.com/florijanstamenkovic/PytorchRnnLM
