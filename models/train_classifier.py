import sys
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine

#NLTK libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


#skicit-learn libraries
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
    Description:
        从数据库中导出数据。
        并将其分为X，Y和category_names
    Input:
        数据库文件名。
    Output:
        X：输入特征数据；
        Y：标签。
        category_names: 标签名，即类别名。
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponsePipeline_table',engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
    Description:
        分词函数。对文本进行处理，分词，并去掉停用词。
    Input: 
        需要分词的文本。
    Output：
        分词后得到的词列表。
    """
    # 使用正则表达式清理数据
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    
    # lemmatizer并去掉停用词
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in nltk.corpus.stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """   
    Description:    
        构建一个基于网格搜索的ML模型。
    Input:
        None
    Output:
        model: ML模型。
    """
    #创建一个机器学习管道
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    #设置网格搜索参数
    parameters = {
                'vect__ngram_range': ((1, 1), (1, 2)),
                'clf__estimator__min_samples_split': (2, 4),
                'tfidf__norm': ['l1', 'l2'],
                'tfidf__sublinear_tf': [True, False]}

    model = GridSearchCV(pipeline, param_grid=parameters,cv=5, verbose=3, n_jobs=-1)    

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """   
    Description:    
        评估模型的质量。
    Input:
        model：ML模型。
        X_test：测试集特征数据。
        Y_test：测试集标签。
        category_names：类别名。
    Output:
        输出基于classification_report得到的相关质量评估值。
    """
    y_preds_test = model.predict(X_test)
    y_preds_test = pd.DataFrame(data=y_preds_test, columns=Y_test.columns, index=Y_test.index)
    for col in Y_test.columns:
        print(classification_report(y_preds_test[col],Y_test[col]))


def save_model(model, model_filepath):
    """   
    Description:    
        将模型保存为pkl文件。
    Input:
        model：ML模型。
        model_filepath：保存的文件名。
    Output:
        None。
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()