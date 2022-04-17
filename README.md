# Text classification
The practise of classify natural language documents with applicable categories from a preset collection is known as text classification. It can help you identify client behaviour by classifying conversations on various social networks, feedback, and other web sources. Text classification is used in a variety of domains, including document filtering, automated metadata generation, word sense disambiguation, a populace of hierarchical catalogues of Web resources, and, more broadly, any application that requires document organisation or selective and adaptive document send-off. Text classification begins with feature extraction, in which we attempt to extract the key features/words from the text in order to differentiate them. As a result, rather than obtaining all of the characteristics from the corpus, it is critical to obtain usable features.

# GCN
The text is typically processed in linear order by the majority of Recurrent Neural Network designs (left-to-right or right-to-left). It has the potential to vanish the long-distance links between entities/words. Recently, for retaining contextual information at a long distance, Graph neural network methods have evolved, and this GNN design is extremely strong, capable of capturing input from very distant neighbours. One such text categorization approach employs a single big graph drawn from a complete corpus of words and phrases.

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
