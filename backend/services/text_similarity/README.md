# Text Similarity Using BERT
 

- End to End NLP text similarity project , served as a REST API via Flask
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/c/quora-question-pairs/data)
- My kaggle Notebook can be found [here](https://www.kaggle.com/raryan/quora-text-similarity-using-bert)


## Steps to Run the Project:
- What is [**Virtual Environment in python ?**](https://www.geeksforgeeks.org/python-virtual-environment/)
- [Create virtual environment in python](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/)
- [Create virtual environment Anaconda](https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/)
- create a virtual environment and install [requirements.txt](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/develop/requirements.txt)

> pip install -r requirements.txt


### For Training/Fine-Tuning
- After Setting up the environment go to [**backend/services/text_similarity/application/ai/training/**](https://github.com/R-aryan/Text-Similarity-Using-BERT/tree/develop/backend/services/text_similarity/application/ai/training) and run **main.py** and the training will start.
- After training is complete the weights of the model will be saved in weights directory, and this weights can be used for inference.

> python main.py


### For Prediction/Inference
- Download the pre-trained weights(file is Zipped) from [here](https://drive.google.com/drive/folders/1CwlhY4BkhyU6vAtFnM_W75cSvTsJC4n-?usp=sharing) and place it inside the weights' folder(**backend/services/text_similarity/application/ai/weights/**) after unzipping it.
- After setting up the environment: go to [**backend/services/text_similarity/api**](https://github.com/R-aryan/Text-Similarity-Using-BERT/tree/develop/backend/services/text_similarity/api) and run **app.py**.
- After running the above step the server will start(Endpoint- **localhost:8080**).  
- You can send the **POST/GET** request at this URL - **localhost:8080/text_similarity/api/v1/predict** (you can find the declaration of endpoint under [**backend/services/text_similarity/api/__init__.py**](https://github.com/R-aryan/Text-Similarity-Using-BERT/blob/develop/backend/services/text_similarity/api/__init__.py) )
- You can also see the logs under [**(backend/services/text_similarity/logs)**](https://github.com/R-aryan/Text-Similarity-Using-BERT/tree/develop/backend/services/text_similarity/logs) directory.

> python app.py


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
