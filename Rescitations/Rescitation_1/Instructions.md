Pure Python Perceptron

Download, install, run and study the pure python perceptron code (examples.py, perceptron.py)

1. Similar to the given majority and xor examples, define 8 training examples to train a one-wire-not function which returns 1 if x1 is 0, 0 if x1 is 1. All other inputs with combinations of 1 and 0 on x2 and x3 do not change the output. Train 100 epochs with the activation function hyperbolic tangent. What happens to weights w2 and w3 and why?

2. Explore alternative activation functions (ReLU, sigmoid, tanh) and initial weight distributions (uniform over 0-1, uniform over -1 to 1, zeros). Which converge the fastest for majority and one-wire-not? Which degenerate and why? 

XOR

3. Do the problem on slide 9 of lecture #2.

Basic Torch

4. Initialize a tensor A of 40 x 256 x 1000 elements with random values between 0 and 1 of type 16 byte floats and another tensor x of 1000 elements or random values between 1.0 and 10 and calculate the tensor product Ax. What is the shape of the result?

5. Port the pure perceptron example to pytorch. Assume batch sizes of 8. 

CoLA

Download, install, run and study the CoLA pytorch example (CoLA.tsv,
Download (CoLA.tsv, CoLA.py

Download CoLA.py)

6. Instead of a running this as a single hidden layer of 500 units, run it as a deeper neural network of two fully connected hidden layers of 100 units each. Does accuracy improve, degrade or stay about the same?

7. Technically the program should be based on BCELoss rather than CrossEntropyLoss. Make the necessary changes. 

8. Instead of running this as a feed forward network represent this as an RNN

9. Repeat 8 with an LSTM (or GRU)

10. Thought provoking question. Ever since the 1950s theoretical linguists have been developing theories of syntax. This has progressed through initial deep generative syntax, Government and Binding, X-Bar theory to Minimalism. For computer scientists an easy, though grossly simplifying way to get insight into this is to model syntax via context free grammars such as S->NP VP, VP-> V NP, VP -> V, NP -> D N, D-> 'a'|'the', N->'dog'|'boy', V->'bites'|'sleeps'. With such a grammar one could, from S,  generate "the boy bites the dog" as well as, unfortunately "*a boy sleeps the dog". In any case, even if a linguistic theory over or under generates, it does represent at least some type of knowledge. How could we use such knowledge to improve our results over the CoLA corpus?