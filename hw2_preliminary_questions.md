1. a) It makes sense to maximize $c \cdot w$ as it represents the similarity between $c$ and $v$. Hence, this will maximize 
$\sigma(c \cdot w)$. This is consistent with minimizing  the loss function which contains the term: 
$\mathbf{1}_{c \in C^+}\log\sigma(c \cdot w)$

    b) When $c \in C^-$, we want to $\bold{minimize}$ $\sigma(c \cdot w)$ for the same reasons

    Geometrically, maximizing $\sigma(c \cdot w)$ pushes the embeddings of $c$ closer to the embedding of $w$ while minimizing it pushes its embedding away. This is desirable since words with a similar meaning often appear together and their embeddings will therefore be close to each other.  

2. a) The main idea of the paper is to map the inputs into another space where the distance in the new space is large if the original distance was large and small if the original distance was small. Mapping the inputs to a new space allow to control to which transformation the input should be invariant.

    b) In our case, we use indicator functions instead of $Y$. The case when $\mathbf{1}_{c \in C^+} = 1$ corresponds to the case when $Y=0$ (genuine pair) and the case when $\mathbf{1}_{c \in C^-} = 1$ corresponds to the case when $Y=1$ (impostor pair)

    c) $E_W(X_1, X_2)$ would be equivalent to $\sigma(c\cdot w)$ which measures similarity. However, it is different in our case because when $c$ and $w$ are similar, so is $\sigma(c\cdot w)$. This is the reason for the introduction of the $-$ signs in the equation.

    d) Finally, the analogs of $L_G$ and $L_I$ are respectively the $x \mapsto \log x$ and the $x \mapsto \log (1-x)$ functions


