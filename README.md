# CS5720-Assignment-4
Home Assignment 4 — Sequence-to-Sequence Models, Attention Mechanisms & Transformers
Course: CS5720 — Neural Networks & Deep Learning (Fall 2026)
Institution: University of Central Missouri
Department: Department of Computer Science & Cybersecurity
Student Name: [Navjot kaur]
Student ID: [700774749]
Executive Summary
This repository contains the full submission for CS5720 Home Assignment 4, focusing on sequence modeling, attention mechanisms, and Transformer neural network architectures. The assignment is divided into three comprehensive sections:

Part I — Theoretical Short Answers: Analytical evaluation of encoder-decoder dynamics, attention scoring functions, self-attention mechanics, multi-head attention parameter dimensions, and Transformer vs. RNN trade-offs.
Part II — Programming Implementations: Python scripts using NLTK and NumPy for text preprocessing (tokenization, stop-word filtering, stemming) and custom 2D scaled dot-product attention calculation.
Part III — Step-by-Step Numerical Calculations: Line-by-line manual derivation of attention scores, Softmax probabilities, weighted value vectors, and final context representations for Token 3.


Part I — Short-Answer Questions & Theoretical Math
Question 1: Sequence-to-Sequence Models and Attention

(a) Purpose of Encoder and Decoder:

Encoder: Reads the variable-length input sequence (e.g., a source language sentence) token-by-token and compresses its semantic information into a fixed-length hidden state context vector.
Decoder: Takes the context vector produced by the encoder and generates the target output sequence step-by-step in an autoregressive fashion.


(b) Information Loss Bottleneck in Long Sentences:

Traditional Seq2Seq architectures compress the entire source sequence into a single static context vector of fixed dimensionality ($d$). For long or complex input sequences, early token representations are diluted or overwritten by subsequent hidden state updates, causing severe translation quality degradation.


(c) How Attention Resolves the Bottleneck:

Attention eliminates the single static vector bottleneck by granting the decoder direct access to all encoder hidden states ($\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_{T_x}$) at every step of generation. The model computes a dynamic relevance distribution across all source tokens, allowing it to focus selectively on parts of the input sequence relevant to predicting the current target token.




Question 2: Attention Mechanisms

(a) Hard Attention vs. Soft Attention:

Hard Attention: Selects a single input position stochastically (using sampling or max picking). Because this decision is non-differentiable, hard attention cannot be trained using standard gradient descent and requires reinforcement learning methods like REINFORCE.
Soft Attention: Computes a continuous, weighted probability distribution over all input positions using the Softmax function. Because every operation is smooth and continuous, soft attention is fully differentiable end-to-end, enabling efficient optimization via standard backpropagation.


(b) Calculation of Raw Attention Scores:
Given decoder state $\mathbf{s} = [1, 0]^\top$ and encoder states $\mathbf{h}_1 = [1, 0]^\top$, $\mathbf{h}_2 = [0, 1]^\top$, $\mathbf{h}_3 = [1, 1]^\top$:

$e_1 = \mathbf{s}^\top \mathbf{h}_1 = (1 \cdot 1) + (0 \cdot 0) = \mathbf{1}$
$e_2 = \mathbf{s}^\top \mathbf{h}_2 = (1 \cdot 0) + (0 \cdot 1) = \mathbf{0}$
$e_3 = \mathbf{s}^\top \mathbf{h}_3 = (1 \cdot 1) + (0 \cdot 1) = \mathbf{1}$


(c) Softmax Weights and Context Vector Calculation:

Exponentiate Raw Scores:

$e^{e_1} = e^1 \approx 2.7183$
$e^{e_2} = e^0 = 1.0000$
$e^{e_3} = e^1 \approx 2.7183$
Sum $Z = 2.7183 + 1.0000 + 2.7183 = 6.4366$


Softmax Attention Weights ($\alpha_i = e^{e_i} / Z$):

$\alpha_1 = \frac{2.7183}{6.4366} \approx \mathbf{0.422}$
$\alpha_2 = \frac{1.0000}{6.4366} \approx \mathbf{0.155}$
$\alpha_3 = \frac{2.7183}{6.4366} \approx \mathbf{0.422}$


Context Vector ($\mathbf{c} = \sum_{i=1}^3 \alpha_i \mathbf{h}_i$):
$$\mathbf{c} = 0.422 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.155 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.422 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.422 + 0.422 \\ 0.155 + 0.422 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.844 \\ 0.577 \end{bmatrix}}$$




Question 3: Self-Attention in Transformers

(a) Self-Attention vs. Seq2Seq Attention:

Self-Attention: Operates within a single sequence, computing pairwise interactions between all tokens in that sequence to update their contextual representations.
Seq2Seq Encoder-Decoder Attention: Connects two distinct sequences, querying encoder representations (source language) from the decoder (target language).


(b) Roles of Query ($Q$), Key ($K$), and Value ($V$) Vectors:

Query ($Q$): Represents the current token seeking context ("What information am I searching for?").
Key ($K$): Represents the index/profile of each token in the sequence ("What features do I contain?").
Value ($V$): Contains the actual feature content of each token to be aggregated into the updated representation.


(c) Modeling Long-Distance Dependencies:

Unlike RNNs that must pass information sequentially across $T$ hidden state transitions, self-attention connects all token pairs directly in $\mathcal{O}(1)$ path length. For example, in "The animal didn't cross the street because it was too tired", self-attention allows the pronoun "it" to directly attend to "animal" and "tired" regardless of sequence distance.




Question 4: Multi-Head Attention

(a) Purpose of Multi-Head Attention:

A single attention head averages out relationships across different representation subspaces. Multi-head attention projects Queries, Keys, and Values into $h$ distinct lower-dimensional subspaces, allowing the model to attend simultaneously to different types of relationships (e.g., syntactic structure, positional proximity, semantic reference).


(b) Head Dimension Calculation:
Given model dimension $d_{\text{model}} = 512$ and $h = 8$ heads:
$$d_k = \frac{d_{\text{model}}}{h} = \frac{512}{8} = \mathbf{64}$$
(c) Combining Multi-Head Outputs:

The outputs from all $h$ attention heads ($Z_0, Z_1, \dots, Z_{h-1}$) are concatenated horizontally along the feature dimension and multiplied by a trainable projection weight matrix $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$:
$$Z_{\text{final}} = \text{Concat}(Z_0, Z_1, \dots, Z_{h-1}) W^O$$




Question 5: Transformer Architecture

(a) Need for Positional Encodings:

Because self-attention processes all sequence tokens in parallel without recurrent loops or convolutions, the operations are permutation-invariant. Positional encodings (sinusoidal functions or learned embeddings) are added to token embeddings to explicitly inject word order information.


(b) Masked Self-Attention in Decoder:

During training with teacher forcing, target sequences are fed all at once. Masked self-attention sets future token similarity scores (where index $j > t$) to $-\infty$ prior to Softmax, preventing token $t$ from looking ahead at future target tokens.


(c) Comparison: RNNs vs. Transformers:

FeatureRecurrent Neural Networks (RNNs)Transformer ArchitecturesSequential ProcessingSequential ($\mathcal{O}(T)$ time steps)Parallel across all tokens ($\mathcal{O}(1)$ steps per layer)GPU ParallelizationPoor (sequential dependence across steps)High (matrix multiplications across full sequence)Long-Distance DependenciesDegrades over time (vanishing gradients)Direct pairwise connections ($\mathcal{O}(1)$ path length)Computational Complexity$\mathcal{O}(T \cdot d^2)$ operations$\mathcal{O}(T^2 \cdot d)$ operations (quadratic in sequence length)

Part II — Programming Tasks & Code Results
Question 1: NLP Preprocessing Pipeline
Python Implementation
import re
from nltk.stem import PorterStemmer

def nlp_preprocessing_pipeline(sentence):
    # 1. Tokenize sentence into words and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", sentence)

    # 2. Remove common English stopwords
    stopwords = {"are", "used", "in", "and", "the", "a", "an", "of", "to", "is", "it"}
    tokens_no_stopwords = [w for w in tokens if w.lower() not in stopwords]

    # 3. Apply stemming using PorterStemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in tokens_no_stopwords]

    return tokens, tokens_no_stopwords, stemmed_words

sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
tokens, no_stopwords, stemmed = nlp_preprocessing_pipeline(sentence)

print("Original Tokens:", tokens)
print("Tokens Without Stopwords:", no_stopwords)
print("Stemmed Words:", stemmed)

Execution Output
Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'virtual', 'assistants', 'like', 'Alexa', 'Siri', '.']
Stemmed Words: ['nlp', 'techniqu', 'virtual', 'assist', 'like', 'alexa', 'siri', '.']

Short-Answer Analysis

Stemming vs. Lemmatization:

Stemming uses heuristic rule-based suffix trimming, which can produce non-valid dictionary words (e.g., "techniques" $\rightarrow$ "techniqu").
Lemmatization uses vocabulary and morphological structure to return valid base dictionary forms (lemmas) (e.g., "techniques" $\rightarrow$ "technique").


Stop Word Removal Impact:

Useful: In text classification or keyword indexing, removing high-frequency words reduces dataset noise and feature space dimensions.
Harmful: In sentiment analysis or machine translation (e.g., "not good"), removing stop words destroys grammatical context and negation logic.




Question 2: Scaled Dot-Product Attention
Python Implementation
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # 1. Compute raw dot-product similarity scores
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # 2. Apply Softmax along the last dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 3. Compute weighted sum of values V
    output = np.matmul(attention_weights, V)

    return attention_weights, output

# Test Inputs
Q = np.array([[1, 0], [0, 1]], dtype=float)
K = np.array([[1, 0], [0, 1]], dtype=float)
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)

attn_weights, output_matrix = scaled_dot_product_attention(Q, K, V)

print("Attention Weights Matrix (Softmax):")
print(np.round(attn_weights, 4))
print("\nFinal Output Matrix:")
print(np.round(output_matrix, 4))

Execution Output
Attention Weights Matrix (Softmax):
[[0.7311 0.2689]
 [0.2689 0.7311]]

Final Output Matrix:
[[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]

Short-Answer Analysis

Purpose of Scaling Factor ($\sqrt{d_k}$):

For large key dimensions $d_k$, dot products grow large in magnitude, pushing the Softmax function into regions with extremely small gradients (vanishing gradients). Scaling by $\sqrt{d_k}$ stabilizes gradient flow during backpropagation.


Capturing Word Relationships:

Self-attention computes similarity scores between query and key vectors regardless of positional distance, allowing words separated by many tokens to pass information directly.




Part III — Step-by-Step Numerical Calculation
Token 3 Self-Attention Calculation Walkthrough
Given Lecture Vectors:

Query 3 ($Q_3$): $[2, 1, 3]^\top$
Keys: $K_1 = [0, 1, 1]^\top$, $K_2 = [4, 4, 0]^\top$, $K_3 = [2, 3, 1]^\top$
Values: $V_1 = [0, 0, 1]^\top$, $V_2 = [2, 8, 0]^\top$, $V_3 = [2, 6, 3]^\top$


Step 1: Raw Attention Scores (Unscaled Dot Products)

$e_{3,1} = Q_3 \cdot K_1 = (2 \cdot 0) + (1 \cdot 1) + (3 \cdot 1) = 0 + 1 + 3 = \mathbf{4}$
$e_{3,2} = Q_3 \cdot K_2 = (2 \cdot 4) + (1 \cdot 4) + (3 \cdot 0) = 8 + 4 + 0 = \mathbf{12}$
$e_{3,3} = Q_3 \cdot K_3 = (2 \cdot 2) + (1 \cdot 3) + (3 \cdot 1) = 4 + 3 + 3 = \mathbf{10}$


Step 2: Softmax Attention Weights ($\alpha_3$)
Applying Softmax to $[4, 12, 10]$ yields probabilities $[0.0003, 0.8805, 0.1192]$.
Rounding to one decimal place according to lecture slides:
$$\alpha_3 = [\mathbf{0.0}, \mathbf{0.9}, \mathbf{0.1}]$$

Step 3: Calculate Weighted Value Vectors

Weighted $V_1$: $0.0 \times [0, 0, 1]^\top = \mathbf{[0.0, 0.0, 0.0]^\top}$
Weighted $V_2$: $0.9 \times [2, 8, 0]^\top = \mathbf{[1.8, 7.2, 0.0]^\top}$
Weighted $V_3$: $0.1 \times [2, 6, 3]^\top = \mathbf{[0.2, 0.6, 0.3]^\top}$


Step 4: Final Output Representation Vector ($z_3$)
$$z_3 = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 1.8 \\ 7.2 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.2 \\ 0.6 \\ 0.3 \end{bmatrix} = \mathbf{\begin{bmatrix} 2.0 \\ 7.8 \\ 0.3 \end{bmatrix}}$$
