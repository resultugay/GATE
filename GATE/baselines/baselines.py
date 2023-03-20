import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig,NodeConfig,TabNetModelConfig,CategoryEmbeddingMDNConfig,MixtureDensityHeadConfig,NODEMDNConfig,AutoIntConfig,TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
import random
import sys


dataset = ['football','nba','synthetic']

football_cat_columns = ['club_name','league_name','club_name1','league_name1']

all_cat_cols = {}
all_cat_cols['football'] = ['club_name','league_name','club_name1','league_name1']
all_cat_cols['nba'] = ['team_abbreviation','college','country','team_abbreviation1','college1','country1']
all_cat_cols['synthetic'] = ['LN','status','job','city','country','marital_status','LN1','status1','job1','city1','country1','marital_status1']
original_stdout = sys.stdout


for data in dataset:
    training = pd.read_csv('../data_preprocessed/' + data + '/training_5_percent_binary_classification.csv')
    validation = pd.read_csv('../data_preprocessed/' + data + '/validation_binary_classification.csv')
    test = pd.read_csv('../data_preprocessed/' + data + '/test_binary_classification.csv')

    all_columns = list(training.columns)
    cat_columns = all_cat_cols[data]
    num_columns = [elem for elem in all_columns if elem not in cat_columns]
    remove_cols = ['id','row_id','entity_id','timestamp',
                  'id1','row_id1','entity_id1','timestamp1']
    
    if data == 'football':
        num_columns.remove('sofifa_id')
        num_columns.remove('sofifa_id1')
        
    for col in remove_cols:
        try:
            num_columns.remove(col)
        except:
            pass

    num_columns.remove('target')    
    
    training.drop(remove_cols,axis=1,inplace=True)
    validation.drop(remove_cols,axis=1,inplace=True)
    test.drop(remove_cols,axis=1,inplace=True)


    config = {}
    config['tabnet_config'] = TabNetModelConfig(
                task="classification",
                learning_rate = 1e-3,
                n_d = 16, n_a=16, n_steps=4,
                metrics = ["accuracy","f1"],
                metrics_params = [{},{"num_classes":2}]
            )

    config['TabTransformer_config'] = TabTransformerConfig(
                task="classification",
                learning_rate = 1e-3,
                metrics = ["accuracy","f1"],
                metrics_params = [{},{"num_classes":2}]
            )   
    config['Node'] = NodeConfig(
                task="classification",
                num_layers=2, # Number of Dense Layers
                num_trees=128, #Number of Trees in each layer
                depth=5, #Depth of each Tree
                embed_categorical=False, #If True, will use a learned embedding, else it will use LeaveOneOutEncoding for categorical columns
                learning_rate = 1e-3,
                metrics = ["accuracy","f1"],
                metrics_params = [{},{"num_classes":2}]
            )
    config['CatEmb'] = CategoryEmbeddingModelConfig(
                task="classification",
                layers="1024-512-512",  # Number of nodes in each layer
                activation="LeakyReLU", # Activation between each layers
                learning_rate = 1e-3,
                metrics = ["accuracy","f1"],
                metrics_params = [{},{"num_classes":2}]
            )
    results = {}
    seeds = {}
    for model_name,model_config in config.items():
        results[model_name] = []
        seeds[model_name] = []
        for i in range(5):
            seed = random.randint(0,10000)
            data_config = DataConfig(
                target=['target'], 
                continuous_cols=num_columns,
                categorical_cols=cat_columns,
            )

            trainer_config = TrainerConfig(
                auto_lr_find=False, # Runs the LRFinder to automatically derive a learning rate
                batch_size=1024,
                max_epochs=20,
            )

            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config="./optimizer_config.yml", 
                trainer_config=trainer_config,
            )

            tabular_model.fit(train=training, validation=validation,seed=seed)
            result = tabular_model.evaluate(test)
            results[model_name].append(result)
            seeds[model_name].append(seed)
            #pred_df = tabular_model.predict(test)
            #pred_df.to_csv('../data/football/result.csv')


    with open('../results/'+ data +'_baseline_results.txt', 'w') as f:
        sys.stdout = f
        for model_name,res in results.items():
            print('Global seed for ',data ,' and model ', model_name,seeds[model_name])
            print(model_name,res)
        sys.stdout = original_stdout 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    