# Text classification
The practise of classify natural language documents with applicable categories from a preset collection is known as text classification. It can help you identify client behaviour by classifying conversations on various social networks, feedback, and other web sources. Text classification is used in a variety of domains, including document filtering, automated metadata generation, word sense disambiguation, a populace of hierarchical catalogues of Web resources, and, more broadly, any application that requires document organisation or selective and adaptive document send-off. Text classification begins with feature extraction, in which we attempt to extract the key features/words from the text in order to differentiate them. As a result, rather than obtaining all of the characteristics from the corpus, it is critical to obtain usable features.

# GCN
The text is typically processed in linear order by the majority of Recurrent Neural Network designs (left-to-right or right-to-left). It has the potential to vanish the long-distance links between entities/words. Recently, for retaining contextual information at a long distance, Graph neural network methods have evolved, and this GNN design is extremely strong, capable of capturing input from very distant neighbours. One such text categorization approach employs a single big graph drawn from a complete corpus of words and phrases.

# GraphSAGE
Embeddings for nodes present in a graph are useful for almost all kind of application but it requires the information from all the nodes available in the graph which is transductive approach as they can not work neatly if the nodes are not already seen and the approach that work on unseen nodes is known as inductive approach. GraphSAGE is a technique that uses inductive approach to efficiently generate the node embeddings. Here, instead of training each node a function is used that generates embeddings by sampling and aggregration. Like GCN, GraphSAGE can also be used for the task of text classification.

# GAT 
As attention mechanism has become powerful tool in almost all the sequence based task Dealing with variable sized inputs, focusing on the most relevant parts of the input to make decisions are some benefits of using attention mechanism. When an attention mechanism is used to compute a representation of a single sequence is known as self-attention. Graph Model Based on Attention Mechanism for Text Classification is also proved helpful as we are getting better accuracy than the LSTM and RNN.

# How to RUN
1. Download the data files for the required model (NLP or GNN). Extract and place it in the data folder.
2. For NLP models replace the train, dev, test csv paths with appropriate language file paths. Whereas the GNN models don't required to specify any path.
3. To run the NLP models, use the jupyter notebook and description added for your reference.
4. To run the GNN models, make sure you have the cuda enabled and run the below commands for any language.

Run the remove_words.py to preprocess the data and remove the stopwords and less frequent words (<5)

**List of dataset-names:** ('20ng', 'R8', 'R52', 'ohsumed', 'mr', 'hin', 'tel', 'bn', 'gu', 'kn', 'ml', 'ta','mar')
```
>>> python remove_words.py <dataset-name>
```
Run the build_graph.py to prepare the graph using the entire corpus.
```
>>> python build_graph.py <dataset-name>
```
Run the run.py to run the experiments.

**List of model-names:** ("GCN", "SAGE", "GAT")
```
>>> python run.py --model <model-name> --cuda True --dataset <dataset-name>
```

# Dataset
**NLP Models data donwload link:** https://iiitaphyd-my.sharepoint.com/:f:/g/personal/pavan_baswani_research_iiit_ac_in/EhHbL4vdAXpPnpO7RdIx410BuyMr8exZK7uGFlsrE6iJEg?e=dTLoxb

**GNN Models data download link:** https://iiitaphyd-my.sharepoint.com/:f:/g/personal/pavan_baswani_research_iiit_ac_in/Eq5FZEVwNRFDlNhQ0cxXEDkBPpl2hUDfkQj10eRxH6K5IQ?e=S5oGqO

The english raw dataset can be downloaded from this link: https://github.com/yao8839836/text_gcn/tree/master/data
1. The 20NG dataset contains 18,846 documents evenly categorized into 20 different categories.
2. The Ohsumed corpus is from the MEDLINE database, contains 23 disease categories.
3. R52 and R8 are two subsets of the Reuters 21578 dataset and contains 52, 8 categories respectively.
4. MR is a movie review dataset f or binary sentiment classification. It contains 5,331 positive and 5,331
negative reviews.

## Multilingual-Indic Dataset
To train the LSTM based Neural Language model, we used the scraped news articles from various news websites for corresponding languages tabulated below. Some of these websites have the pagination which requires the button click to navigate to the next page. To achieve this, we have used the Selenium (https://www.selenium.dev/) library. Using BeautifulSoup (https://beautiful-soup-4.readthedocs.io/en/latest/) python library. Initially, all the URLs of the news articles are extracted and used as the input to the content crawler (which scrap the html content from the given URL with required delay mentioned in website metadata.). This content crawler, uses the requests and BeautifulSoup libraries to extract the HTML content of the articles and save them in the local drive.

| SNo 	| Language 	| Website 	|
|---	|---	|---	|
| 1 	| Telugu (te) 	| https://www.vaartha.com/ 	|
| 2 	| Hindi (hi) 	| https://www.indiatv.in/ 	|
| 3 	| Tamil (ta) 	| https://www.updatenews360.com/ 	|
| 4 	| Kannada (kn) 	| https://kannadanews.com/ 	|
| 5 	| Malayalam (ml) 	| https://dailyindianherald.com/ 	|
| 6 	| Bengali (bn) 	| https://www.abplive.com/ 	|
| 7 	| Gujarathi (gu) 	| https://www.gujaratsamachar.com/ 	|
| 8 	| Marathi (mr) 	| https://www.abplive.com/ 	|


Here, the required details about each dataset presented in the below table. For GCN, GAT and SAGE models, we have used the limited data for some languages (25% of hindi, 38% of telugu, 44% of kannada and 41% of gujarathi). Remaining are trained on entire corpus for each language.


| **Datasets** 	| **Docs** 	| **Training** 	| **Test** 	| **Words** 	| **Nodes** 	| **Classes** 	| **Average length** 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| 20NG 	| 18846 	| 11314 	| 7532 	| 42757 	| 61603 	| 20 	| 221.26 	|
| R8 	| 7674 	| 5485 	| 2189 	| 7688 	| 15362 	| 8 	| 65.72 	|
| R52 	| 9100 	| 6532 	| 2568 	| 8892 	| 17992 	| 52 	| 69.82 	|
| Ohsumed 	| 7400 	| 3357 	| 4043 	| 14157 	| 21557 	| 23 	| 135.82 	|
| MR 	| 10662 	| 7108 	| 3554 	| 18764 	| 29426 	| 2 	| 20.39 	|
| Hindi 	| 38121 	| 36214 	| 1907 	| 17383 	| 42661 	| 5 	| 471.18 	|
| Marathi 	| 15884 	| 15099 	| 785 	| 26125 	| 23306 	| 7 	| 742.10 	|
| Bengali 	| 20541 	| 19500 	| 1041 	| 43725 	| 30876 	| 8 	| 561.05 	|
| Telugu 	| 38526 	| 36599 	| 1927 	| 20674 	| 43231 	| 9 	| 310.98 	|
| Tamil 	| 35591 	| 33811 	| 1780 	| 34694 	| 42461 	| 12 	| 543.46 	|
| Malayalam 	| 38104 	| 36201 	| 1903 	| 31793 	| 45975 	| 5 	| 708.96 	|
| Kannada 	| 36647 	| 34814 	| 1833 	| 35006 	| 44660 	| 9 	| 469.17 	|
| Gujarathi 	| 36922 	| 35079 	| 1843 	| 50028 	| 47871 	| 11 	| 545.40 	|


# Experimental Results [F1 Measure (weighted avg)]
**NLP Models and results download link:** https://iiitaphyd-my.sharepoint.com/:f:/g/personal/pavan_baswani_research_iiit_ac_in/EmLpO8LUr-dHhEjQEmhcUQ4Bel2hVoCDyn_cIJA3yDtAJA?e=m0DwO1

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
| Gujrathi    |   0.92   |     0.92    |     -     |      -     |     -    | **0.93** |

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
</div>
