import os
import json
# from DataPreprocessing import visualization_hot_waves
from ModelConstruction import regression
from ModelConstruction import hotwaves_nuclear_multi
from ModelConstruction import WordCloud_LDA
from flask import Flask
from  flask import  render_template
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from urllib.request import urlopen
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from copy import deepcopy
import base64
from jupyter_dash import JupyterDash
import time

hw_root = '../Data/HeatWaves'
ph_root = '../Data/PublicHealth'
le_root = '../Data/LivelihoodEconomy'
sc_root = '../Data/SocialCulture'

indicator_file_dict = {'fish_catches':'fish_catches.csv',
    'nuclear_production':'Electricity_production_from_nuclear sources.csv',
    'nuclear_consumption':'Alternative_nuclear_energy.csv',
    'inflation':'inflation.csv'}

hw_df_path = os.path.join(hw_root,'hot_waves.csv')
ph_df_path = os.path.join(ph_root,'final_data.csv')


with open(os.path.join(hw_root,'countries.geojson')) as response:    
    counties = json.load(response)

def VisualizeMap(choice_year):
    global hw_df_path,counties
    hw_df = pd.read_csv(hw_df_path)
    df = hw_df[['alpha_3_code',choice_year]]
    max_value = int(np.max(df[choice_year]))+1
    fig = px.choropleth_mapbox(df, 
        geojson=counties,
        featureidkey ='properties.ISO_A3',
        locations='alpha_3_code', 
        color=choice_year,
        color_continuous_scale=px.colors.diverging.RdYlGn[::-1], 
        # color_continuous_scale="Viridis",
        range_color=(1, max_value), 
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 50.8503, "lon": 4.3517},
        opacity=0.5,
        labels={'alpha_3_code':'country'})
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()
    return  fig

def VisualizePH(PH_indicator):
    global ph_df_path
    con_data = pd.read_csv(ph_df_path)
    scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'],  [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds
    data_slider = []
    for year in con_data.YEAR.unique():
    
        # I select the year 
        sel_data=con_data[con_data['YEAR']== year]
        
        for col in sel_data.columns:  # I transform the columns into string type so I can:
            sel_data[col] = sel_data[col].astype(str)

        ### I create the text for mouse-hover for each state, for the current year    
        sel_data['text'] = sel_data['COUNTRY']

        ### create the dictionary with the data for the current year
        data_one_year = dict(
                            type='choropleth',
                            locations = sel_data['COUNTRY'],
                            z=sel_data[PH_indicator].astype(float),
                            locationmode='ISO-3',
                            colorscale = scl,
                            text = sel_data['text'],
                            )

        data_slider.append(data_one_year)  # I add the dictionary to the list of dictionaries for the slider
    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='Year {}'.format(i + 1986)) # label to be displayed for each step (year)
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]  
    layout = dict(geo=dict(scope='europe',
                       projection={'type': 'natural earth'}),
              sliders=sliders,title=PH_indicator)
    fig = go.Figure(data=data_slider, layout=layout) 
    # fig = plotly.offline.iplot(pic)
    return fig


def VisualizeLE(indicator,country):
    
    global le_root,indicator_file_dict,hw_df_path
    hw_df = pd.read_csv(hw_df_path,index_col='alpha_3_code')
    if indicator == 'fish_catches':
        le_df = pd.read_csv(os.path.join(le_root,indicator_file_dict[indicator]),index_col='country_3code')
    else:
        le_df = pd.read_csv(os.path.join(le_root,indicator_file_dict[indicator]),index_col='Country Code')
    years = [int(i) for i in hw_df.columns if i.startswith('19') or i.startswith('20')]
    hws_ = []
    indicators_ = []
    years_ = []
    for year in range(min(years),max(years)+1):
        try:
            if str(hw_df.loc[country,str(year)])=='nan'  or str(le_df.loc[country,str(year)])=='nan'or str(hw_df.loc[country,str(year)])=='[]' or str(le_df.loc[country,str(year)])=='0.0':
                continue
            years_.append(year)
            hws_.append(float(hw_df.loc[country,str(year)]))
            indicators_.append(float(le_df.loc[country,str(year)]))
        except:
            continue
    fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=years_, y=indicators_,
                        mode='lines',
                        name='indicator'),secondary_y=False,)
    fig.add_trace(go.Scatter(x=years_, y=hws_,
                        mode='lines+markers',
                        name='heat wave'),secondary_y=True,)
    fig.update_layout(
        title_text="Indicators of {} and heat waves for {}".format(indicator,country)
        )
    
    return fig
    


# dcc.Slider(value=2000, min=1986, max=2020, step=1, 
#            marks={-5: 'Year 1986 Degrees', 0: '0', 10: 'Year 2020'})

app = dash.Dash(__name__,title='MDA_Belgium',external_stylesheets = [dbc.themes.BOOTSTRAP])

# app = JupyterDash(__name__, title='MDA_Belgium', external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
colors = {
    'background': '#111111',
    'bodyColor':'#F2DFCE',
    'text': '#7FDBFF'
}

def get_page_heading_style():
    return {'backgroundColor': colors['background']}

def get_page_heading_title():
    return html.H1(children='MDA by Group Belgium',
                    style={
                    'textAlign': 'center',
                    # 'color': colors['text']
                })

def get_page_heading_subtitle():
    return html.Div(children='Impact of Heat Wave',
                    style={
                        'textAlign':'center',
                        # 'color':colors['text']
                    })


def generate_page_header():
    main_header =  dbc.Row(
                    [
                        dbc.Col(get_page_heading_title(),md=12)
                    ],
                    align="center",
                    # style=get_page_heading_style()
                )
    subtitle_header = dbc.Row(
                    [
                        dbc.Col(get_page_heading_subtitle(),md=12)
                    ],
                    align="center",
                    # style=get_page_heading_style()
                )
    header = (main_header,subtitle_header)
    return header

def get_title(title,position):
    return html.Div(children=title,
                    style={
                        'textAlign':position,
                        # 'color':colors['text']
                    })

def Macrograph():
    return dcc.Graph(id="Macrograph", figure=VisualizeMap('2000'))

def Mapslider():
    return html.Div([dcc.Slider(value=2000, min=1986, max=2020, step=1, id='year_slider',
        marks={1986: 'Year 1986', 2000: 'Year 2000', 2020: 'Year 2020'})])

def PHGraph():
    return html.Div([dcc.Graph(id='PHGraph',figure=VisualizePH('death_all'))], style={"height": "50%", "width": "80%"})


def LEGraph():
    return html.Div(dcc.Graph(id="LEGraph",figure=VisualizeLE('fish_catches','BEL')), style={"height": "50%", "width": "80%"})

def create_dropdown_PH_list():
    global ph_df_path
    df = pd.read_csv(ph_df_path)
    original_list = [i for i in df.columns if i.startswith('death_')]
    dropdown_list = []
    for cntry in sorted(original_list):
        tmp_dict = {'label':cntry,'value':cntry}
        dropdown_list.append(tmp_dict)
    return dropdown_list

def create_dropdown_LE_list():
    original_list = ['fish_catches','nuclear_production','nuclear_consumption','inflation']
    dropdown_list = []
    for cntry in sorted(original_list):
        tmp_dict = {'label':cntry,'value':cntry}
        dropdown_list.append(tmp_dict)
    return dropdown_list

def get_dropdown(mode):
    if mode == 'PH':
        return html.Div(children=[
                            dcc.Dropdown(id='myPHindicator',
                                options=create_dropdown_PH_list(),
                                value='death_all', style={"height": "50%", "width": "50%"}
                            )
                        ])
    else:
        return html.Div(children=[
                            dcc.Dropdown(id='myLEindicator',
                                options=create_dropdown_LE_list(),
                                value='fish_catches', style={"height": "50%", "width": "50%"}
                            )
                        ])


def InputRegression():
    input_groups = html.Div([
        dbc.InputGroup([
            dbc.InputGroupAddon("Portion of Test Dataset (0.10-0.40)", addon_type="prepend"),
            dbc.Input(id='reg_portion_test',value=0.2, type="number")],),
        dbc.InputGroup([
            dbc.InputGroupAddon("Portion of Valid. Dataset (0.10-0.40)", addon_type="prepend"),
            dbc.Input(id='reg_portion_valid',value=0.1,type="number")]),
        dbc.InputGroup([
            dbc.InputGroupAddon("K-fold CV for parameters tuning (2-10)",addon_type="prepend"),
            dbc.Input(id='reg_k_cv',value=3,type='number')])
        ], style={"height": "50%", "width": "50%"})
    return input_groups

def InputTimeSeries(indicator):
    input_groups = html.Div(['you could choose country from: '+' '.join(GetLECountries(indicator)[-1]) +'\n',
        dbc.InputGroup([
            dbc.InputGroupAddon("Epochs for training", addon_type="prepend"),
            dbc.Input(id='epochs',value=10, type="number")],),
        dbc.InputGroup([
            dbc.InputGroupAddon("Terms of Test Dataset", addon_type="prepend"),
            dbc.Input(id='terms_test',value=5,type="number")],),
        dbc.InputGroup([
            dbc.InputGroupAddon("Country", addon_type="prepend"),
            dbc.Input(id='LE_country',value='BEL',type="text")],),
        # dbc.InputGroup([
        #     dcc.Dropdown(id='LE_country',
        #                         options=create_dropdown_LE_country_list(indicator),
        #                         value='BEL', style={"height": "50%", "width": "50%"}
        #                     )
        #     ])
        ], style={"height": "50%", "width": "50%"})
    return input_groups

def InputNLP():
    input_groups = html.Div([
        dbc.InputGroup([
            dbc.InputGroupAddon("Number of Words", addon_type="prepend"),
            dbc.Input(id='words_range',value=200, type="number")],),
        dbc.InputGroup([
            dbc.InputGroupAddon("Number of Topics", addon_type="prepend"),
            dbc.Input(id='num_topics',value=3,type="number")]),
        dbc.InputGroup([
            dbc.InputGroupAddon("Min count for words vector", addon_type="prepend"),
            dbc.Input(id='min_count',value=3,type="number")]),
        ], style={"height": "50%", "width": "50%"})
    return input_groups

def create_dropdown_LE_country_list(indicator):
    _,_,original_list = GetLECountries(indicator)
    dropdown_list = []
    for cntry in sorted(original_list):
        tmp_dict = {'label':cntry,'value':cntry}
        dropdown_list.append(tmp_dict)
    return dropdown_list

def GetLECountries(indicator):
    global le_root,indicator_file_dict,hw_df_path
    hw_df = pd.read_csv(hw_df_path,index_col='alpha_3_code')
    country_list = list(hw_df.index)
    if indicator == 'fish_catches':
        le_df = pd.read_csv(os.path.join(le_root,indicator_file_dict[indicator]),index_col='country_3code')
    else:
        le_df = pd.read_csv(os.path.join(le_root,indicator_file_dict[indicator]),index_col='Country Code')
    
    record_country = {}
    record_time_series = {}
    years = [int(i) for i in hw_df.columns if i.startswith('19') or i.startswith('20')]
    for year in range(min(years),max(years)+1):
        record_time_series[year] = {}
    for country in country_list:
        record_country[country] = {'heatwaves':[],indicator:[]}
        for year in range(min(years),max(years)+1):
            try:
                if str(hw_df.loc[country,str(year)])=='nan'  or str(le_df.loc[country,str(year)])=='nan'or str(hw_df.loc[country,str(year)])=='[]' or str(le_df.loc[country,str(year)])=='0.0':
                    continue
                record_country[country]['heatwaves'].append(float(hw_df.loc[country,str(year)]))
                record_country[country][indicator].append(le_df.loc[country,str(year)])
                record_time_series[year]['_'.join([country,'heatwaves'])] = float(hw_df.loc[country,str(year)])
                record_time_series[year]['_'.join([country,indicator])] = float(le_df.loc[country,str(year)])
            except:
                continue
        
        if record_country[country] == {'heatwaves':[],indicator:[]} or len(record_country[country]['heatwaves']) != len(record_country[country][indicator]) :
            del record_country[country]  
    
        
    select_len = max([len(record_country[country][indicator]) for country in country_list if country in record_country.keys() ])
    # print (select_len)
    for country in country_list:
        if country not in record_country.keys():
            continue
        if len(record_country[country][indicator])<select_len:
            del record_country[country]
    stat_countries = list(record_country.keys())
    return record_time_series,years,stat_countries

def RunRegression(test_portion,val_portion,cv_search,death_indicator):
    population_df = pd.read_csv(os.path.join(ph_root,'population.csv'),index_col='Country Code')
    hw_df = pd.read_csv(os.path.join(hw_root,'hot_waves.csv'),index_col='alpha_3_code')
    country_code_df = pd.read_csv(os.path.join(hw_root,'country_code.csv'))
    df_final = pd.read_csv(os.path.join(ph_root,'final_data.csv'))
    # df_final_2 = pd.read_csv('final_data_2.csv')

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

    country_dict = regression.EncodeCountry(country_code_df)

    dict_hw_importance = {}

    # df_final_2 = Final2Preprocess(df_final_2,population_dict,hw_dict,'ISO','Year',country_dict)
    df_final = regression.Final2Preprocess(df_final,population_dict,hw_dict,'COUNTRY','YEAR',country_dict)
    record_indicators = {}
    # record_indicators = LGBMRegression(df_final_2,'Total Deaths',['Country','ISO'],record_indicators)

    death_indicators =  [i for i in df_final.columns if i.startswith('death_')]
    drop_cols_ = ['COUNTRY']
    drop_cols_.extend(death_indicators)
    
    record_indicators = regression.LGBMRegression(df_final,death_indicator,drop_cols_,record_indicators,test_portion,val_portion,cv_search,dict_hw_importance)

    return record_indicators[death_indicator]

def generate_card_content(card_header,card_value,overall_value,ad_title):
    card_head_style = {'textAlign':'center','fontSize':'150%'}
    card_body_style = {'textAlign':'center','fontSize':'200%'}
    card_header = dbc.CardHeader(card_header,style=card_head_style)
    card_body = dbc.CardBody(
        [
            html.H5(f"{(card_value):,}", className="card-title",style=card_body_style),
            html.P(
                "{}: {:,}".format(ad_title,overall_value),
                className="card-text",style={'textAlign':'center'}
            ),
        ]
    )
    card = [card_header,card_body]
    return card

def ResultsRegression(portion_test,portion_valid,cv_search,indicator):
    results = RunRegression(portion_test,portion_valid,cv_search,indicator)
    MSE_train = round(results['MSE_train'],4)
    MSE_test = round(results['MSE_test'],4)
    importance_portion = round(results['importance_heatwaves']['importance_portion'],4)
    expected_portion = round(results['importance_heatwaves']['expected_portion'],4)
    x = list(results['importance'].keys())
    y = [results['importance'][i] for i in x]
    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(generate_card_content("Importance of heat wave",importance_portion,expected_portion,'average importance'), color="success", inverse=True), width=3),
                    dbc.Col(dbc.Card(generate_card_content("MSE on test dataset",MSE_test,MSE_train,'MSE on train dataset'), color="warning", inverse=True), width=3),
                    dbc.Col(dcc.Graph(figure=get_importance_bar(x,y,indicator)))
                ],
                className="mb-4",
            ),
        ],id='results_regression'
    )
    return cards

def get_importance_bar(x,y,indicator):
    fig = go.Figure(
        data=[go.Bar(x=x,
        y=y,
        # orientation='h',
        )],
        layout_title_text="A importance bar plot concerning {}".format(indicator)
    )
    return fig


def RunTimeSeries(epochs,test_terms,country,indicator):
    record_time_series,years,stat_countries = GetLECountries(indicator)
    record_time_series_2 = deepcopy(record_time_series)
    for year in range(min(years),max(years)+1):
        for key_ in record_time_series[year]:
            if key_.split('_')[0] not in stat_countries:
                del record_time_series_2[year][key_]
    df_time_series = pd.DataFrame(record_time_series_2)
    df_time_series=df_time_series.T
    df_time_series = df_time_series.dropna(axis=0,how='all').dropna(axis=1,how="all")
    df_time_series['Date'] = df_time_series.index
    # print (df_time_series)
    split_date = df_time_series.index[-test_terms]
    record_indicators = {}
    record_indicators[indicator] = {}
    record_indicators,results_list = hotwaves_nuclear_multi.TimeSeriesNN(epochs,record_indicators,indicator,country,df_time_series,split_date)
    return record_indicators[indicator][country],results_list

def ResultsTimeSeries(epochs,test_terms,country,indicator):
    results,results_list = RunTimeSeries(epochs,test_terms,country,indicator)
    # print (results_list)
    MSE_NN = round(results['NN_test_mse'],4)
    MSE_LSTM = round(results['LSTM_test_mse'],4)
    importance_hw= round(results['NN_importance_heatwaves'],4)
    importance_ts = round(results['NN_importance_timeseries'],4)

    def TimeSeriesPreditcGraph(x,ture_y,predict_y,indicator,country):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ture_y,
                            mode='lines',
                            name='True Value'))
        fig.add_trace(go.Scatter(x=x, y=predict_y,
                            mode='lines+markers',
                            name='Predict Value'))
        fig.update_layout(
        title_text="Preditction of {} by lag term and heat waves for {}".format(indicator,country)
        )
        return fig

    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(generate_card_content("Importance of heat wave",importance_hw,importance_ts,'Importance of lag term'), color="success", inverse=True), width=3),
                    dbc.Col(dbc.Card(generate_card_content("MSE on test dataset by NN",MSE_NN,MSE_LSTM,'MSE on test dataset by LSTM'), color="warning", inverse=True), width=3),
                    dbc.Col(dcc.Graph(id="predict_timeseries",figure=TimeSeriesPreditcGraph(*results_list,indicator,country)))
                ],
                className="mb-4",
            ),
        ],id='results_timeseries'
    )

    return cards



picture_path = os.path.join(sc_root,'heatwaves.jpg')
output_pic_path = os.path.join(sc_root,'output.png')
df = pd.read_csv(os.path.join(sc_root,'heatwaves_twitter.csv'))
text_list = df['text']
rejected_words = ['heatwave','heatwaves','http','https','isnt','im','tco','dont','amp','ive','thats','didnt','havent','george',
                    'get','take','make','come','go','gonna','say','think','know','start','people','want','cant','thing']
not_related_words = ['dnf','dream','fcu']

words_dict = {}
text_list = [WordCloud_LDA.text_prepare(x,rejected_words,not_related_words,words_dict) for x in text_list]
text_list = [x for x in text_list if len(x)>0 ]


def ResultsNLP(words_range,num_topics,min_count):
    global text_list,picture_path,output_pic_path
    df_tfidf = WordCloud_LDA.GetTFIDF(text_list,words_range,min_count)
    WordCloud_LDA.WordCloudDraw(df_tfidf,words_range,picture_path,output_pic_path,sc_root,'noshow')
    encoded_image = base64.b64encode(open(output_pic_path, 'rb').read()).decode('ascii')
    # print (text_list[:10])
    topic_df = WordCloud_LDA.LDAModel([' '.join(i) for i in text_list],num_topics,min_count)
    print (topic_df)
    nlp_feed = html.Div(
        [
        dbc.Row(
            [
        dbc.Col((dash_table.DataTable(
            columns=[{"name": i, "id": i} 
                    for i in topic_df.columns],
            data=topic_df.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="paleturquoise"),
            style_data=dict(backgroundColor="lavender")))),
        dbc.Col(html.Img(src='data:image/png;base64,{}'.format(encoded_image),style={'width':'90%'}))
        ])],id='results_nlp'
    )

    return nlp_feed 


def generate_layout():
    page_header = generate_page_header()
    layout = dbc.Container(
        [
            page_header[0],
            page_header[1],
            html.Hr(),
            dbc.Row(
                    [
                        dbc.Col(get_title('Trend of Duration of heat wave in Europe','left'),md=12)
                    ]),
            dbc.Row(
                [
                    dbc.Col(Macrograph())                    
                ]
            ),
            dbc.Row(
                [                  
                    dbc.Col(Mapslider())
                ],
                align="center",
            ),
            html.Hr(),
            dbc.Row(
                    [
                        dbc.Col(get_title('Indicators for Public Health','left'),md=12)
                    ]),
            html.Hr(),
            dbc.Row(
                    [
                        dbc.Col(get_dropdown('PH')),                  
                    ],
                align="left",
                ),
            dbc.Row(
                    [
                        dbc.Col(PHGraph()),
                        dbc.Col(InputRegression()),                   
                    ],
                align="center",
                ),
            dbc.Row(
                    [
                        dbc.Col(get_title('Results of regression concerning the impact on Public Health','left'),md=12),
                                          
                    ],
                align="left",
                ),
            dbc.Row(
                    [
                        dbc.Col(ResultsRegression(0.2,0.1,3,'death_all')),                    
                    ],
                align="center",
                ),
            
       
        html.Hr(),
            dbc.Row(
                    [
                        dbc.Col(get_title('Indicators for Livelihood Economy','left'),md=12)
                    ]),
            html.Hr(),
        dbc.Row(
                    [
                        dbc.Col(get_dropdown('LE')),                  
                    ],
                align="left",
                ),
        dbc.Row(
                    [
                        dbc.Col(LEGraph()),
                        dbc.Col(InputTimeSeries('fish_catches')),                   
                    ],
                align="center",
                ),
        dbc.Row(
                    [
                        dbc.Col(get_title('Results of time series model concerning the impact on Livelihood Economy','left'),md=12),
                                          
                    ],
                align="left",
                ),
        dbc.Row(
                    [
                        dbc.Col(ResultsTimeSeries(10,5,'BEL','fish_catches')),                    
                    ],
                align="center",
                ), 
        html.Hr(),
        dbc.Row(
                [
                    dbc.Col(get_title('Indicators for Social Culture','left'),md=12)
                ]),
        html.Hr(),
        dbc.Row(
                    [
                        dbc.Col(InputNLP()),                   
                    ],
                align="center",
                ),
        dbc.Row(
                    [
                        dbc.Col(get_title('LDA Topics and Word Cloud: ','left'),md=12),
                                          
                    ],
                align="left",
                ),
        dbc.Row(
                    [
                        dbc.Col(ResultsNLP(200,3,3)),                    
                    ],
                align="center",
                ),         
        ],
        fluid=True
        )
    return layout
app.layout = generate_layout()

@app.callback(
    
    Output('Macrograph','figure'),
    
    [
        Input('year_slider', 'value'),
    ]
)
def update_chart(choice_year):
    if not choice_year:
        choice_year = 2000
    return VisualizeMap(str(choice_year))


@app.callback(
    
    Output('PHGraph','figure'),
    Output('results_regression','children'),
    
    
    [
        Input('myPHindicator', 'value'),
        Input('reg_portion_test', 'value'),
        Input('reg_portion_valid', 'value'),
        Input('reg_k_cv', 'value')
    ]
)

def update_chart(changed_indicator,reg_portion_test,reg_portion_valid,reg_k_cv):
    if not changed_indicator:
        changed_indicator = 'death_all'
    if not reg_portion_test:
        reg_portion_test = 0.2
    if not reg_portion_valid:
        reg_portion_valid = 0.1
    if not reg_k_cv:
        reg_k_cv = 3
    return VisualizePH(changed_indicator),  \
        ResultsRegression(reg_portion_test,reg_portion_valid,reg_k_cv,changed_indicator)

@app.callback(

    Output('LEGraph','figure'),
    Output('results_timeseries','children'),
   
    [
        Input('myLEindicator', 'value'),
        Input('epochs', 'value'),
        Input('terms_test', 'value'),
        Input('LE_country', 'value')
    ]
)

def update_chart(ind_LE,epochs,terms_test,LE_country):
    if not ind_LE:
        ind_LE = 'fish_catches'
    if not epochs:
        epochs = 10
    if not terms_test:
        terms_test = 5
    if not LE_country:
        LE_country = 'BEL'
    return VisualizeLE(ind_LE,LE_country),ResultsTimeSeries(epochs,terms_test,LE_country,ind_LE)

@app.callback(
    Output('results_nlp','children'),
    [
        Input('words_range', 'value'),
        Input('num_topics', 'value'),
        Input('min_count', 'value'),
    ]
)

def update_chart(words_range,num_topics,min_count):
    if not words_range:
        words_range = 200
    if not num_topics:
        num_topics = 3
    if not min_count:
        min_count = 3
    return ResultsNLP(words_range,num_topics,min_count)

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False)
    
