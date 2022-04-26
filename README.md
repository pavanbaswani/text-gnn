# Text classification
The practise of classify natural language documents with applicable categories from a preset collection is known as text classification. It can help you identify client behaviour by classifying conversations on various social networks, feedback, and other web sources. Text classification is used in a variety of domains, including document filtering, automated metadata generation, word sense disambiguation, a populace of hierarchical catalogues of Web resources, and, more broadly, any application that requires document organisation or selective and adaptive document send-off. Text classification begins with feature extraction, in which we attempt to extract the key features/words from the text in order to differentiate them. As a result, rather than obtaining all of the characteristics from the corpus, it is critical to obtain usable features.

# GCN
The text is typically processed in linear order by the majority of Recurrent Neural Network designs (left-to-right or right-to-left). It has the potential to vanish the long-distance links between entities/words. Recently, for retaining contextual information at a long distance, Graph neural network methods have evolved, and this GNN design is extremely strong, capable of capturing input from very distant neighbours. One such text categorization approach employs a single big graph drawn from a complete corpus of words and phrases.

# GraphSAGE
Embeddings for nodes present in a graph are useful for almost all kind of application but it requires the information from all the nodes available in the graph which is transductive approach as they can not work neatly if the nodes are not already seen and the approach that work on unseen nodes is known as inductive approach. GraphSAGE is a technique that uses inductive approach to efficiently generate the node embeddings. Here, instead of training each node a function is used that generates embeddings by sampling and aggregration. Like GCN, GraphSAGE can also be used for the task of text classification.

# GAT 
As attention mechanism has become powerful tool in almost all the sequence based task Dealing with variable sized inputs, focusing on the most relevant parts of the input to make decisions are some benefits of using attention mechanism. When an attention mechanism is used to compute a representation of a single sequence is known as self-attention. Graph Model Based on Attention Mechanism for Text Classification is also proved helpful as we are getting better accuracy than the LSTM and RNN.

# Dataset
The dataset can be downloaded from this link: https://github.com/yao8839836/text_gcn/tree/master/data
1. The 20NG dataset contains 18,846 documents evenly categorized into 20 different categories.
2. The Ohsumed corpus is from the MEDLINE database, contains 23 disease categories.
3. R52 and R8 are two subsets of the Reuters 21578 dataset and contains 52, 8 categories respectively.
4. MR is a movie review dataset f or binary sentiment classification. It contains 5,331 positive and 5,331
negative reviews.

| Dataset 	| #Docs 	| #Training 	| #Test 	| #Words 	| #Nodes 	| #Classes 	| Avg Length 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| 20NG 	| 18,846 	| 11,314 	| 7,532 	| 42,757 	| 61,603 	| 20 	| 221.26 	|
| R8 	| 7,674 	| 5,485 	| 2,189 	| 7,688 	| 15,362 	| 8 	| 65.72 	|
| R52 	| 9,100 	| 6,532 	| 2,568 	| 8,892 	| 17,992 	| 52 	| 69.82 	|
| Ohsumed 	| 7,400 	| 3,357 	| 4,043 	| 14,157 	| 21,557 	| 23 	| 135.82 	|
| MR 	| 10,662 	| 7,108 	| 3,554 	| 18,764 	| 29,426 	| 2 	| 20.39 	|

# Accuracy for SAGE and GAT
| Datasets | GraphSAGE Accuracy | GAT Accuracy | GraphSAGE F1 score | GAT F1 score |
|----------|--------------------|--------------|--------------------|--------------|
| 20NG     | 79.14              | 46.42        | 7914               | 46.41        |
| R8       | 95.20              | 91.96        | 95.20              | 91.95        |
| R52      | 97.54              | 91.74        | 97.54              | 91.74        |
| Ohsumed  | 98.56              | 89.71        | 98.56              | 89.71        |
| MR       | 76.45              | 76.25        | 76.44              | 76.25        |

# Visualization

<div align="center">    
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/tsne_plots.jpg?raw=true" width="500px" height="400px" alt="R8_gcn_test" align=center />
</div>

Plot showing Movie Review dataset classification.

Note: All the datasets used in this repository will be availabel at: https://tinyurl.com/3b4fafp3
