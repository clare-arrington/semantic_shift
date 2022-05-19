import plotly.express as px
import pandas as pd

def print_results(result_dict, output_path, anchor_name, source_names):    
    with open(f'{output_path}/neighbors.txt', 'w') as f:
        print(f'=========== Targets from {anchor_name} ===========', file=f)
    
        for target in sorted(list(result_dict.keys())):
            print(f'{target} ', file=f, end='')

            source_neighbors = result_dict[target]

            print(f'\n\n\t=== Target neighbors from {anchor_name} ===\n\t\t', file=f, end='')
            print(', '.join(source_neighbors[anchor_name]), file=f)

            for source in source_names:
                print(f'\n\t=== {source} neighbors ===\n\t\t', file=f, end='')
                print(', '.join(source_neighbors[source]), file=f)

            print('\n', file=f)

def make_plot(x, words, categories, 
              wv_names, path, flip=False):
    data = []
    for x_i, word, category in zip(x, words, categories):
        data.append([
            x_i[0], x_i[1], word, category, 12
        ])

    df = pd.DataFrame(data, columns=['x','y','name','label','size'])

    layout = {
        "paper_bgcolor": "#FAFAFA",
        "plot_bgcolor": "#DDDDDD",
        "dragmode": "pan",
        'font': {
            'family': "Courier New, monospace",
            'size': 13
        },
        'margin': {
            'l': 60,
            'r': 40,
            'b': 40,
            't': 40,
            'pad': 4
        },
        'xaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'yaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'legend': {
            "title":''
        },
        'title': {
        }
    }

    symbols = ["square", "circle"]
    # green  : #7D8E4D, #5E6A3D
    # purple : #764e80, #62395F
    colors = ["#7D8E4D", "#764e80"]
    if flip:
        symbols.reverse()
        colors.reverse()

    # colors.append("#4D243D")

    fig = px.scatter(df, x='x', y='y', 
        color='label', symbol='label', text='name',
        hover_name="name", size="size",
        hover_data={"label":False,
                    "name":False,
                    "x":False, 
                    "y":False},
        symbol_map={wv_names[0]: symbols[0], 
                    wv_names[1]: symbols[1],
                    f'Target from {wv_names[0]}': 'x',
                    f'Target from {wv_names[1]}': 'x'},
        color_discrete_map={wv_names[0]: colors[0], 
                            wv_names[1]: colors[1],
                            f'Target from {wv_names[0]}': colors[0],
                            f'Target from {wv_names[1]}': colors[1]
                            }
    )
    fig.update_layout(**layout)
    fig.update_traces(textposition='top center',
    textfont={'family': "Raleway, sans-serif" }
    )
    # fig.show()
    fig.write_html(path)

