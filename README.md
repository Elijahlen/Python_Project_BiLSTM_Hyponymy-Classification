# Python_BiLSTM_hyponymy-classification
* Build a system that can extract [hyponym and hypernym](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) from a sentence. 
* Use [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme to represent the results (i.e., encode the tags), as both hypernyms and hyponyms can be phrases.
For example, in sentence *Stephen Hawking is a physicist .*, phrase *Stephen Hawking* is the **hyponym** of *physicist*, and *physicist* is the **hypernym** of *Stephen Hawking*.
 * `[Stephen, Hawking, is, a, physicist, .]` is the input word list 
 * `[B-TAR, I-TAR, O, O, B-HYP, O]` is the corresponding output tag sequence, where `TAR` corresponds to hyponyms and `HYP` corresponds to hypernyms.
# Data Preprocessing
+ `tags.txt` lists all the tags.
+ `train.txt`, `dev.txt`, `test.txt` refers to training set, development set, and test set, individually. Each of them is in the CoNLL format, with each line having two columns for word and tag. Sentences are separated by blank lines.
+ `word_embeddings.txt` is the pretrained word embedding file, with word in the first column and embeddings (a vector with 50-dimensions) in the rest columns.    
# A Baseline BiLSTM Model

+ In this project, you are required to use Bidirectional LSTM (BiLSTM) and SoftMax in your model.
+ The input tensor will go through a *BiLSTM layer*, followed by a *softmax* function to determine the output tags.
+ An example of the model is shown as follows:<img src="./workflow.png" width="25%">.

