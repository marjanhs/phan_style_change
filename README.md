# PAN2018: PHAN Style Change Detection
**A Parallel Hierarchical Attention Network for Style Change Detection**

Authors : **Marjan Hosseinia** and **Arjun Mukherjee**

We propose a model for the new problem of style change detection.
Given a document, we verify if it contains at least one style change. In other
words, the task is to investigate whether it is written by one or multiple authors.
The model is composed of two parallel attention networks. Unlike the conventional
recurrent neural networks that use the character or word sequences to learn
the underlying language model of documents, our model focuses on the hierarchical
structure of the language and observes the parse tree features of a sentence
using a pre-trained statistical parser. Besides, our model is independent of style
change positions although they are given during the training phase. The reason
is to have a more applicable approach to the real world problems where such information
is not available. PAN 2018 results show that it achieves 82% accuracy
and stays at the second rank.


### prerequisits:
[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/),
[pytorch 0.4](https://anaconda.org/soumith/pytorch),
[nltk](https://anaconda.org/anaconda/nltk), [tqdm](https://anaconda.org/conda-forge/tqdm),
[pycorenlp](https://github.com/smilli/py-corenlp)

### Dataset:
[PAN 2018 Style Change Detection](https://pan.webis.de/clef18/pan18-web/author-identification.html).
The csv files of the dataset can be found in data folder.

### Stanford CoreNLP:

Dowanload [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

Unzip the file:

`Unzip stanford-corenlp-full-2018-02-27.zip`

Run the server:

`cd stanford-corenlp-full-2018-02-27`

`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`

### train:
`sypt_train.py -c csv_files_path -o ptf_files_path`  

csv_file_path: path to the csv train and validation files

ptf_files_path: path to ptf_train and ptf_validation files

e.g. `sypt_train.py -c data/ -o data/`

### test:
`sypt_test.py -c csv_files_path -o ptf_files_path`

csv_file_path: path to the csv test file

ptf_files_path: path to ptf_train and ptf_test files and the json output file 

e.g. `sypt_test.py -c data/ -o data/`

