# Text Similarity Using BERT
- End to End NLP text similarity project , served as a REST API via Flask App.
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/c/quora-question-pairs/data)


####  Steps to run the project [Click Here](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/develop/backend/services/text_similarity/README.md)

### Dataset Description


- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

### Goal

The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning.


### Following are the screenshots for the sample **request** and sample **response.**

- Request sample

![Sample request](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/main/msc/sample_request.png)
  <br>
  <br>
- Response Sample

![Sample response](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/main/msc/sample_response.png)
<br>
<br>

![sample request and response](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/main/msc/sample_request_response.png)
