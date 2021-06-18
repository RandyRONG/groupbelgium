import os
import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.charts import Timeline
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/"
## cd pyecharts-assets 
## python -m http.server



if __name__ == '__main__':

    root_dir = '../../Data/HeatWaves/'

    data = pd.read_csv(root_dir+'hot_waves.csv',index_col='country_name') 
    output_filename = 'HotWavesforEU.html'
    drop_index = [list(data.index)[i] for i in range(len(list(data['2020']))) if (list(data['2020'])[i]) == '[]']
    data = data.drop(drop_index, axis=0)

    attr = list(data.index)
    print (attr)
    times = [i for i in data.columns if i.startswith('19') or i.startswith('20')]
    max_ = int(max([float(i) for i in list(data['2020'])]))+1
    print (times)
    def map_visualmap(sequence, date) -> Map:
        c = (
            Map()
            .add(date, sequence, maptype="world")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="HotWavesforEU"),
                visualmap_opts=opts.VisualMapOpts(max_=max_),
            )
        )
        return c
    timeline = Timeline()

    for i in range(len(times)):
        time =times[i]
        row = data[time].tolist()
        sequence_temp = list(zip(attr,row))
        map_temp = map_visualmap(sequence_temp,time)
        timeline.add(map_temp,time).add_schema(play_interval=360)

    timeline.render(output_filename)
    os.system("start {}".format(output_filename))
