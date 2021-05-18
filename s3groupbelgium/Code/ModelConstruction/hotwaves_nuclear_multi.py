import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("qt4agg")
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
import eli5
from eli5.sklearn import PermutationImportance


def TimeSeriesNN(epochs,record_indicators,target_name,stat_country,df_time_series,split_date):
    record_indicators[target_name][stat_country] = {}
    selected_df = df_time_series[[i for i in df_time_series.columns if i.startswith(stat_country)]]
    train = selected_df.loc[:split_date]
    test = selected_df.loc[split_date:] 
    plt.figure(figsize=(10, 6))
    ax = train['_'.join([stat_country,target_name])].plot()
    test['_'.join([stat_country,target_name])].plot(ax=ax)
    plt.legend(['train', 'test']);     
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    X_train = np.asarray(train_sc[:-1])
    y_train = np.asarray([i[1] for i in train_sc[1:]])
    X_test = np.asarray(test_sc[:-1])
    y_test = np.asarray([i[1] for i in test_sc[1:]])

    nn_model = Sequential()
    nn_model.add(Dense(16, input_dim=2, activation='relu'))
    nn_model.add(Dense(1))
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

    def GetImportances(record_indicators,train_model,X_train,model_name):
        perm = PermutationImportance(train_model, scoring="neg_mean_squared_error",random_state=0).fit(X_train, y_train)
        print(perm.feature_importances_)
        feature_importances_ = perm.feature_importances_
        record_indicators[target_name][stat_country][model_name+'_importance_heatwaves'] = feature_importances_[0]
        record_indicators[target_name][stat_country][model_name+'_importance_timeseries'] = feature_importances_[1]
        eli5.show_weights(perm, feature_names = ['heatwaves',target_name])
        return record_indicators
    
    record_indicators = GetImportances(record_indicators,nn_model,X_train,'NN')

    y_pred_test_nn = nn_model.predict(X_test)
    y_train_pred_nn = nn_model.predict(X_train)
    # record_indicators[target_name][stat_country]['NN_train_R2'] = r2_score(y_train, y_train_pred_nn)
    # record_indicators[target_name][stat_country]['NN_test_R2'] = r2_score(y_test, y_pred_test_nn)
    # print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
    # print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))


    X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    lstm_model = Sequential()
    lstm_model.add(LSTM(8, input_shape=(X_train_lmse.shape[1],1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')

    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train_lmse, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    
    # record_indicators = GetImportances(record_indicators,lstm_model,X_train_lmse,'LSTM')
    
    y_pred_test_lstm = lstm_model.predict(X_test_lmse)
    y_train_pred_lstm = lstm_model.predict(X_train_lmse)

    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
    # record_indicators[target_name][stat_country]['LSTM_train_R2'] = r2_score(y_train, y_train_pred_lstm)
    # record_indicators[target_name][stat_country]['LSTM_test_R2'] = r2_score(y_test, y_pred_test_lstm)

    nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
    lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)

    print('NN: %f'%nn_test_mse)
    print('LSTM: %f'%lstm_test_mse)
    record_indicators[target_name][stat_country]['NN_test_mse'] = nn_test_mse
    record_indicators[target_name][stat_country]['LSTM_test_mse'] = lstm_test_mse

    nn_y_pred_test = nn_model.predict(X_test)
    lstm_y_pred_test = lstm_model.predict(X_test_lmse)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_nn, label='NN')
    plt.title("NN's Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Adj Close Scaled')
    plt.legend()
    # plt.show();
    plt.close() 

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_lstm, label='LSTM')
    plt.title("LSTM's Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Adj Close Scaled')
    plt.legend()
    # plt.show();
    plt.close() 
    return record_indicators,[list(test.index),y_test,[i[0] for i in y_pred_test_nn]]

if __name__ == '__main__':
    root_dir = '../../Data/LivelihoodEconomy/'
    hw_dir = '../../Data/HeatWaves/'
    df_hw = pd.read_csv(hw_dir+"hot_waves.csv",index_col='alpha_3_code')
    df_np = pd.read_csv(root_dir+"Electricity_production_from_nuclear sources.csv",index_col='Country Code')
    df_nc = pd.read_csv(root_dir+"Alternative_nuclear_energy.csv",index_col='Country Code')
    df_fish = pd.read_csv(root_dir+"fish_catches.csv",index_col='country_3code')
    df_infla = pd.read_csv(root_dir+"inflation.csv",index_col='Country Code')

    out_df_path = root_dir+'time_series_{}.csv'
    out_json_path = root_dir+'indicators_time_series.json'
    epochs = 10
    country_list = list(df_hw.index)
    record_indicators = {}
    # chosen_country_code = input('please print the country you wanna see (3-letter country code):')
    for df_target,target_name in [[df_np,'nuclear_production_rate'],[df_nc,'nuclear_consumption_rate'],[df_fish,'fish_catches'],[df_infla,'inflation_consumer_price']]:
        record_country = {}
        record_time_series = {}
        years = [int(i) for i in df_hw.columns if i.startswith('19') or i.startswith('20')]
        for year in range(min(years),max(years)+1):
            record_time_series[year] = {}
        for country in country_list:
            record_country[country] = {'heatwaves':[],target_name:[]}
            for year in range(min(years),max(years)+1):
                try:
                    if str(df_hw.loc[country,str(year)])=='nan'  or str(df_target.loc[country,str(year)])=='nan'or str(df_hw.loc[country,str(year)])=='[]' or str(df_target.loc[country,str(year)])=='0.0':
                        continue
                    record_country[country]['heatwaves'].append(float(df_hw.loc[country,str(year)]))
                    record_country[country][target_name].append(df_target.loc[country,str(year)])
                    record_time_series[year]['_'.join([country,'heatwaves'])] = float(df_hw.loc[country,str(year)])
                    record_time_series[year]['_'.join([country,target_name])] = float(df_target.loc[country,str(year)])
                except:
                    continue
            
            if record_country[country] == {'heatwaves':[],target_name:[]} or len(record_country[country]['heatwaves']) != len(record_country[country][target_name]) :
                del record_country[country]  
        
            
        select_len = max([len(record_country[country][target_name]) for country in country_list if country in record_country.keys() ])
        print (select_len)
        for country in country_list:
            if country not in record_country.keys():
                continue
            if len(record_country[country][target_name])<select_len:
                del record_country[country]
        stat_countries = list(record_country.keys())
        record_time_series_2 = deepcopy(record_time_series)
        for year in range(min(years),max(years)+1):
            for key_ in record_time_series[year]:
                if key_.split('_')[0] not in stat_countries:
                    del record_time_series_2[year][key_]
        df_time_series = pd.DataFrame(record_time_series_2)
        df_time_series=df_time_series.T
        df_time_series = df_time_series.dropna(axis=0,how='all').dropna(axis=1,how="all")
        df_time_series.to_csv(out_df_path.format(target_name))
        df_time_series['Date'] = df_time_series.index
        print (df_time_series)
        split_date = df_time_series.index[-5]
        print (split_date)
        record_indicators[target_name] = {}
        for stat_country in stat_countries:
            # if stat_country != chosen_country_code:
            #     continue
            record_indicators=TimeSeriesNN(epochs,record_indicators,target_name,stat_country,df_time_series,split_date)

    print (record_indicators)

    out_json_dict = json.dumps(record_indicators,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()


