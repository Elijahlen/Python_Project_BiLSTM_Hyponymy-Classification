# Python_BiLSTM_hyponymy-classification
* Build a system that can extract [hyponym and hypernym](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) from a sentence. 
* Use [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme to represent the results (i.e., encode the tags), as both hypernyms and hyponyms can be phrases.
For example, in sentence *Stephen Hawking is a physicist .*, phrase *Stephen Hawking* is the **hyponym** of *physicist*, and *physicist* is the **hypernym** of *Stephen Hawking*.
 * `[Stephen, Hawking, is, a, physicist, .]` is the input word list 
 * `[B-TAR, I-TAR, O, O, B-HYP, O]` is the corresponding output tag sequence, where `TAR` corresponds to hyponyms and `HYP` corresponds to hypernyms.
