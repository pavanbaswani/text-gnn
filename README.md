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
| 20NG     | 79.14              | 46.42        | 79.14              | 46.41        |
| R8       | 95.20              | 91.96        | 95.20              | 91.95        |
| R52      | 97.54              | 91.74        | 97.54              | 91.74        |
| Ohsumed  | 98.56              | 89.71        | 98.56              | 89.71        |
| MR       | 76.45              | 76.25        | 76.44              | 76.25        |

# Experimental Results [F1 Measure (weighted avg)]

| **Dataset** | **LSTM** | **Bi-LSTM** |  **GCN**  |   **GAT**  | **SAGE** | **BERT** |
|-------------|:--------:|:-----------:|:---------:|:----------:|:--------:|:--------:|
| 20ng        |   0.62   |     0.74    |  **0.85** |    0.42    |   0.74   |   0.16   |
| oshumed     |     -    |     0.44    |    0.68   |    0.87    | **0.97** |   0.21   |
| R8          |   0.89   |     0.95    |  **0.96** |    0.90    |   0.94   |   0.93   |
| R52         |     -    |      -      |    0.93   |    0.87    | **0.96** |     -    |
| MR          |   0.72   |     0.72    |  **0.77** |    0.76    |   0.76   |   0.62   |
| Hindi       |   0.92   |     0.94    |  **0.89** |  **0.84**  |   0.54   | **0.94** |
| Telugu      |   0.85   |   **0.86**  |    0.63   |    0.73    |   0.69   |   0.54   |
| Kannada     | **0.85** |     0.84    |    0.78   |    0.76    | **0.78** |   0.76   |
| Bengali     |   0.53   |     0.54    |    0.68   |    0.73    |   0.67   | **0.76** |
| Marathi     |   0.53   |     0.42    | **0.699** |    0.695   |   0.68   |   0.68   |
| Tamil       |   0.85   |   **0.86**  |    0.81   |    0.68    |   0.43   |   0.82   |
| Malyalam    |   0.61   |     0.62    |     -     | **0.6269** |     -    |   0.62   |
| Gujrathi    |   0.92   |     0.92    |     -     |      -     |     -    |     -    |

# Visualization

<div align="center">    
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/tsne_plots.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
 
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/bn_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
 
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/kn_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center /> 
 
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/mar_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
</div>
</br>

<div align="center">    
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/ml_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
  
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/ta_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
 
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/tel_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
   
<img src="https://github.com/pavanbaswani/text-gnn/blob/main/plots/kn_gcn_doc_test.jpg?raw=true" width="200px" height="200px" alt="R8_gcn_test" align=center />
</div>

Plot showing dataset classification.

Note: All the datasets used in this repository will be availabel at: https://tinyurl.com/3b4fafp3
