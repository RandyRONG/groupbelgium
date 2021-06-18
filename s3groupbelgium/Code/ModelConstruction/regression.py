
import os
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("qt4agg")
import matplotlib.pyplot as plt

def LGBMRegression(df,target_name,drop_cols,record_indicators,test_portion,val_portion,cv_search,dict_hw_importance):

    record_indicators[target_name] = {}
    df_test = df.sample(frac=test_portion)
    df_train = df.drop(list(df_test.index))
    df_val = df_train.sample(frac=val_portion)
    df_train = df.drop(list(df_val.index))
    # df_train = df_train.reset_index(drop = True)
    # df_val = df_val.reset_index(drop = True)
    # df_test = df_test.reset_index(dro p = True)

    drop_cols.append(target_name)

    def SplitLabels(sub_df):
        
        y = list(sub_df[target_name])
        X = sub_df.drop(drop_cols, 1)
        return X,y

    train_data,label_train = SplitLabels(df_train)
    val_data,label_val = SplitLabels(df_val)
    test_data,label_test = SplitLabels(df_test)

    X_train = train_data
    y_train = label_train 
    X_test = val_data  
    y_test = label_val

    lgb_train = lgb.Dataset(X_train, y_train)  
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  

    params = {  
        'objective':'regression',
        'boosting_type': 'gbdt',  
        'metric':  'mean_squared_error',  
        'verbose':-1,
        # 'num_leaves': 16,  ### could change but be careful about overfitting
        # 'max_depth': 8,  ### could change but be careful about overfitting
        # 'min_data_in_leaf': 450,  
        # 'learning_rate': 0.01,  ### could change but be careful about local optimization
        # 'feature_fraction': 0.6,  ### like random forest for its features to sample
        # 'bagging_fraction': 0.6,  ### like random forest for its samples to sample
        # 'bagging_freq': 200,  ### how many times for sample
        # 'lambda_l1': 0.01,    ### L1 norm (lead to more zero coeff)
        # 'lambda_l2': 0.01,    ### L2 norm
        # 'weight_column':'name:claim_amount',
        # 'is_unbalance': False # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
        }  

    # train  
    # print ('Start training...'+str(i))  
    
    rds_params = {
        'bagging_freq': range(100, 500, 100),
        'min_child_weight': range(3, 20, 2),
        'colsample_bytree': np.arange(0.4, 1.0),
        'max_depth': range(4, 32, 2),
        'num_leaves':range(16, 64, 4),
        # 'subsample': np.arange(0.5, 1.0, 0.1),
        'feature_fraction': np.arange(0.5, 1.0, 0.1),
        'bagging_fraction': np.arange(0.5, 1.0, 0.1),
        'lambda_l1': np.arange(0.01, 0.1, 0.01),
        'lambda_l2': np.arange(0.01, 0.1, 0.01),
        'min_child_samples': range(10, 30),
        'learning_rate': np.arange(0.01, 0.04, 0.01)}
    
    model = lgb.LGBMRegressor(**params)
    optimized_GBM = model_selection.RandomizedSearchCV(model, rds_params, n_iter=50, cv=cv_search, n_jobs=4)
    optimized_GBM.fit(X_train, y_train) 
    print('best parameters:{0}'.format(optimized_GBM.best_params_))
    print('best score:{0}'.format(optimized_GBM.best_score_))
    params.update(optimized_GBM.best_params_)
    record_indicators[target_name]['best_params'] = params
    record_indicators[target_name]['best_score'] = optimized_GBM.best_score_
    print (params)
    gbm = lgb.train(params,  
                    lgb_train,  
                    num_boost_round=500,   # max training epoches
                    valid_sets=lgb_eval, 
                    early_stopping_rounds=100) # to which epoch to check early_stopping)  

    # print('Start predicting...')  
    importance = gbm.feature_importance()
    importance_dict = {}
    names = gbm.feature_name()  
    # to collect the importance of features
    for index, im in enumerate(importance):  
        if names[index] not in importance_dict.keys():
            importance_dict[names[index]] = [im]
        else:
            importance_dict[names[index]].append(im) 
    importance_sort = sorted(importance_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
    importance_dict_dict = {i[0]:float(i[1][0]) for i in importance_sort}
    total_importance = sum([i[1][0] for i in importance_sort])
    print (target_name)
    print (importance_dict_dict)
    record_indicators[target_name]['importance'] = importance_dict_dict
    
    # lgb.plot_importance(gbm, max_num_features=30)
    # plt.title("Featurertances")
    # plt.show()

    values_train = gbm.predict(train_data, num_iteration=gbm.best_iteration) 
    values_test = gbm.predict(test_data, num_iteration=gbm.best_iteration) 
    mse_train = mean_squared_error(label_train,values_train)
    mse_test = mean_squared_error(label_test, values_test)
    print (target_name,mse_train,mse_test)
    record_indicators[target_name]['MSE_train'] = mse_train
    record_indicators[target_name]['MSE_test'] = mse_test
    dict_hw_importance[target_name] = {"importance":importance_dict['heatwaves'][0],
        "importance_portion":round(importance_dict['heatwaves'][0]/total_importance,4),
        "expected_portion":round(1/len(importance_sort),4),
        "mse_train":round(mse_train,4),
        "mse_test":round(mse_test,4)}
    record_indicators[target_name]['importance_heatwaves'] = {"importance":float(importance_dict['heatwaves'][0]),
        "importance_portion":round(importance_dict['heatwaves'][0]/total_importance,4),
        "expected_portion":round(1/len(importance_sort),4)}
    x = test_data['heatwaves']
    plt.title(target_name) 
    plt.scatter(x, label_test, label='real') 
    plt.scatter(x, values_test, color = 'red', label='predict')
    plt.legend() 
    # plt.show() 
    plt.close()

    return record_indicators


def Final2Preprocess(df_final_2,population_dict,hw_dict,country_col,year_col,country_dict):
    countries_2 = df_final_2[country_col]
    years_2 = df_final_2[year_col]
    
    populations = []
    heatwaves = []
    enocoded_country = []
    for idx_2,country in enumerate(countries_2):
        if country not in population_dict.keys() or country not in hw_dict.keys():
            df_final_2 = df_final_2.drop(idx_2,axis=0)
            continue
        if years_2[idx_2] not in population_dict[country].keys() or years_2[idx_2] not in hw_dict[country].keys():
            df_final_2 = df_final_2.drop(idx_2,axis=0)
            continue
        populations.append(population_dict[country][years_2[idx_2]])
        heatwaves.append(hw_dict[country][years_2[idx_2]])
        enocoded_country.append(country_dict[country])
        
    df_final_2 = df_final_2.reset_index(drop=True)
    df_final_2['populations'] = populations
    df_final_2['heatwaves'] = heatwaves
    df_final_2['enocoded_country'] = enocoded_country

    return df_final_2


def EncodeCountry(df):
    countries = list(df['Alpha-3 code'])
    country_dict = {}
    count_ = 1
    for country in countries:
        country_dict[country] = count_
        count_ += 1

    return country_dict

if __name__ =='__main__':
    root_dir = '../../Data/PublicHealth/'
    hw_dir = '../../Data/HeatWaves/'
    population_df = pd.read_csv(root_dir+'population.csv',index_col='Country Code')
    hw_df = pd.read_csv(hw_dir+'hot_waves.csv',index_col='alpha_3_code')
    country_code_df = pd.read_csv(hw_dir+'country_code.csv')
    out_json_path = root_dir+'indicators_regression.json'
    df_final = pd.read_csv(root_dir+'final_data.csv')
    # df_final_2 = pd.read_csv('final_data_2.csv')
    test_portion = 0.2
    val_portion = 0.15
    cv_search = 5

    def Df2Dict(trans_df):
        trans_dict = {}
        poplulation_years = [int(i) for i in trans_df.columns if i.startswith('19') or i.startswith('20')]
        for country in list(trans_df.index):
            trans_dict[country] = {}
            for year in poplulation_years:
                try:
                    trans_dict[country][year] = float(trans_df.loc[country,str(year)])
                except:
                    continue
        return trans_dict
    
    population_dict = Df2Dict(population_df)
    hw_dict = Df2Dict(hw_df)

    country_dict = EncodeCountry(country_code_df)

    dict_hw_importance = {}

    # df_final_2 = Final2Preprocess(df_final_2,population_dict,hw_dict,'ISO','Year',country_dict)
    df_final = Final2Preprocess(df_final,population_dict,hw_dict,'COUNTRY','YEAR',country_dict)
    record_indicators = {}

    death_indicators = ['death_diabetes_65','death_cerebrovascular_65','death_respiratory_65','death_external','death_internal','death_all']
    drop_cols_ = ['COUNTRY']
    drop_cols_.extend(death_indicators)

    
    for death_indicator in death_indicators:
        record_indicators = LGBMRegression(df_final,death_indicator,drop_cols_,record_indicators,test_portion,val_portion,cv_search,dict_hw_importance)
    
    print (dict_hw_importance)
    print (record_indicators)
    out_json_dict = json.dumps(record_indicators,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()


    
    

    
    

