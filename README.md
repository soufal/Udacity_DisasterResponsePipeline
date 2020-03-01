# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Note:
1. The details of the Project:
    1. 搭建一个ETL管道：
        1. 搭建一个ETL管道处理来自`Figure 8`的灾害消息；
        2. 并将`CSV`文件中的数据分类；
        3. 把数据加载到**SQLite数据库**中。
    2. 搭建一个机器学习管道：
        1. 从数据库中读取数据；
        2. 创建和保存**多输出的监督学习模型**；
    3. 搭建一个网络应用程序：
        1. 从数据库中提取数据进行**数据可视化**；
        2. 使用模型将消息分类成**36类**；

2. The Note of myself:
[DisasterResponsePipeline_note](https://github.com/soufal/Udacity_DisasterResponsePipeline/blob/master/DisasterResponsePipeline_note.md)

3. The solution of `juypter notebook`:
    - [ETL Pipeline Preparation-zh.ipynb](https://github.com/soufal/Udacity_DisasterResponsePipeline/blob/master/jupyter_files/ETL%20Pipeline%20Preparation-zh.ipynb)    

    - [ML Pipeline Preparation-zh.ipynb](https://github.com/soufal/Udacity_DisasterResponsePipeline/blob/master/jupyter_files/ML%20Pipeline%20Preparation-zh.ipynb)
