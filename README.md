# movie_rs
Episerver movie recommendation test

## System requirements:
   - RAM: 16GB
   - CPU: core i3 7th gen
   - GPU: good to have

## Description
   - model can get RMSE ~0.87 on val_set and test_set
   - if the user is not in the train set, recommend top movies for the user:
       - top movies is defined as popular and high rated movie

## Possible improvement
   - using more information from movie metadata
   - try more complex model:
       - more hidden layers
       - try different method (concatenate, sum) to combine MF and movie metadata
   - using pretrained word embedding to embed keywords and other movie metadata

## how to run
#### Step 1:
   Download dataset:
       - Can directly download dataset via http at: https://www.kaggle.com/rounakbanik/the-movies-dataset/data
       - Or can download using CLI via tool at: https://github.com/Kaggle/kaggle-api
   Save downloaded zip file at ./data folder, then extract the zip file
#### Step 2:
   Preprocess data:
   
       - python main.py --action preprocess
       
   The data will be process and split train/test/val and stored in the folder ./model_data

#### Step 3:
   Train model:
   
       - python main.py --action train   

#### Step 4:
   Export http API:
   
       - python main.py --action http_demo

