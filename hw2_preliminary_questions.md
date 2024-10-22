- [General summary](#general-summary)
  - [Part 1: Word2Vec](#part-1-word2vec)
  - [Part 2: Text Classification](#part-2-text-classification)
- [Implementation decisions](#implementation-decisions)
  - [Hyperparameter choice](#hyperparameter-choice)
- [Conclusion](#conclusion)

1. a) It makes sense to maximize $c \cdot w$ as it represents the similarity between $c$ and $v$. Hence, this will maximize 
$\sigma(c \cdot w)$. This is consistent with minimizing  the loss function which contains the term: 
$\mathbf{1}_{c \in C^+}\log\sigma(c \cdot w)$

    b) When $c \in C^-$, we want to $\bold{minimize}$ $\sigma(c \cdot w)$ for the same reasons

    Geometrically, maximizing $\sigma(c \cdot w)$ pushes the embeddings of $c$ closer to the embedding of $w$ while minimizing it pushes its embedding away. This is desirable since words with a similar meaning often appear together and their embeddings will therefore be close to each other.  

2. a) The main idea of the paper is to map the inputs into another space where the distance in the new space is large if the original distance was large and small if the original distance was small. Mapping the inputs to a new space allow to control to which transformation the input should be invariant.

    b) In our case, we use indicator functions instead of $Y$. The case when $\mathbf{1}_{c \in C^+} = 1$ corresponds to the case when $Y=0$ (genuine pair) and the case when $\mathbf{1}_{c \in C^-} = 1$ corresponds to the case when $Y=1$ (impostor pair)

    c) $E_W(X_1, X_2)$ would be equivalent to $\sigma(c\cdot w)$ which measures similarity. However, it is different in our case because when $c$ and $w$ are similar, so is $\sigma(c\cdot w)$. This is the reason for the introduction of the $-$ signs in the equation.

    d) Finally, the analogs of $L_G$ and $L_I$ are respectively the $x \mapsto \log x$ and the $x \mapsto \log (1-x)$ functions


## General summary

We hereby implement the Word2Vec algorithm from scratch. These are the main steps we followed:

### Part 1: Word2Vec

1. Load the data
2. Preprocess + tokenize
3. Split into train + test set
4. Build the contexts-target words pairs and create a dataset from them (Q5)
5. Create a DataLoader adding negative context to each sample of the dataset (Q6-8)
6. Create our Word2Vec class model. We implement a predict method that we will use to predict the target word given its context (Q9)
7. Train the model (Q10)
8. Test the model on word prediction. We get 4.5% accuracy, which is in the range of what we could expect with little data and little training, according to the literature (Q11)

### Part 2: Text Classification
1. Reuse the learned embeddings to train a basic Conv model for text classification (Q1-2)
2. Train the same model from scratch, without using the learned embeddings. Unfortunately, we haven't observed any impro


## Implementation decisions

### Hyperparameter choice
All the hyperparameters were set based on existing literature. We tried to tweak them a little bit but found no genuine increase in performance so left them as is.
We chose 80/20% train test split which is a usual thing to do.

For building the negative contexts, although it was indicated to just randomly sample from the vocabulary, we decided to improve this to not sample words from the positive context. This arguably won't impact the model performance much but it's more rigorous

In the Word2Vec model, we use two different embeddings:
- in_embedding learns the embeddings of words when used as context words. At test time, when using the context to predict the target word, we average the embeddings of the context words.
- out_embedding learns the embeddings of words when used as context words. 

We trained the model for only 15 epochs, which is arguably a bit low but still sufficient for our validation loss to start to plateau. Our goal was not to reach peak performance. We just want to go fast.

We then tested our model on the word prediction task. To perform inference, we average the context embeddings and take output the target embedding that is the most similar to the context.

Note that it would also have made sense to train a text classifier and train then test it on sentiment analysis (downstream task) using the labels in IMDB dataset. But this task is already done in part 2 so it's more fun to do word prediction instead.


## Conclusion

During this HW, I built a very solid understanding of the Word2Vec model. Implementing it from scratch allowed me to fully understand how it works. Also, I also learn the methodology to reimplement models from scratch and how to design the classes in a sensible way so I can now reuse this framework in the future.


