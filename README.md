# Named Entity Recognition


![ner diagram](./images/NER_Diagram.png)


## Run

*test.sh* is a simple bash script. To run it:

```
conda create -n nlp2020-hw1 python=3.7
conda activate nlp2020-hw1
pip install -r requirements.txt
bash test.sh data/dev.tsv
```

Actually, you can replace *data/dev.tsv* to point to a different file, as far as the target file has the same format.
# Results
| Model          | Dev F1 (%) | Test F1 (%) |
| -------------- |:------:|:------:|
| Word emb (100d) + BiLSTM  | 78.02  | 78.73 |
| Word2Vec + BiLSTM | 80.25   | 81.04 | 
| Word2Vec + BiLSTM + CRF | 88.12   | 89.76 | 
| Pos Emb + W2v + CRF | 92.11   | 92.37 | 
| Pos Emb + W2v | 92.20   | 92.66 | 