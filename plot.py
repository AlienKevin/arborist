import os
from matplotlib import patches
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Read the CSV file into a DataFrame
arborist_data = pd.read_csv('benchmark_summary.csv')
wr_extended_data = pd.read_csv(
    'WebRobot-Experiment-Results - webrobot-extension-withheuristics-1s-RQ1.csv')
wr_original_data = pd.read_csv(
    'WebRobot-Experiment-Results - webrobot-noextension-withheuristics-1s-RQ1.csv')

include = ['W239T1', 'W254T1', 'W14T1', 'W149T1', 'W176T1', 'W296T1', 'W252T1', 'W51T2', 'W78T2', 'W228T4', 'W252T2', 'W1T2']

arborist_data = arborist_data[arborist_data['name'].isin(include)]
wr_extended_data = wr_extended_data[wr_extended_data['benchmark ID'].isin(
    include)]
wr_extended_data = wr_extended_data[wr_extended_data['benchmark ID'].isin(
    include)]


def compute_percent(x, y):
    return round(float(x)/float(y) * 100, 1)


def compute_log(x):
    return [math.log10(y) for y in x]


def compute_solved(data):
    data_solved = data[data['intend'] == 'Y'].loc[:,
                                                  ['name', 'seed']].values.tolist()
    data_solved = [tuple(x) for x in data_solved]
    return data_solved


def combine_data(data1, data2):
    data1_solved = compute_solved(data1)
    data2 = data2[~data2[['name', 'seed']].apply(
        tuple, axis=1).isin(data1_solved)]
    combined = pd.concat([data1, data2])
    return combined


def set_figure_size(fig, w, h):
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


def write_to_figure(fig, folder_path, fig_name):
    os.makedirs(folder_path, exist_ok=True)
    fig_path = folder_path + "/" + fig_name
    print('writing to figure... ' + fig_path)
    fig.savefig(fig_path)


def exp1a_plot(data1, data2, data3):
    '''
    given a csv sheet with columns in_pldi, intend, wr_solved
    intend can be "Y" or "N" which marks the benchmarks Arborsit can solve or not
    wr_solved can be "Y" or "N" which marks the benchmarks WebRobot can solve or not
    draw a bar chart with x axis be two groups grouped by in_pldi that can be "Y", "N"
    '''
    # included benchmarks
    data1 = data1[(data1['intend'] == 'Y') | (data1['intend'] == 'N') | (
        data1['intend'] == 'YL') | (data1['in_pldi'] == 'Y')]
    include = data1['name'].values
    # count total number of included benchmarks
    total = len(include)
    pldi_n = len(data1[data1['in_pldi'] == 'Y'])
    print("pldi_n: " + str(pldi_n))
    print(f"total number of included benchmarks: {total}")
    # print(data1[data1['intend'] != 'Y'].to_string())

    # parse wr-extended data
    pldi_names = data1[data1['in_pldi'] == 'Y']['name']
    filtered_data2 = parse_wr_data(data2, pldi_names, include)
    counts2 = filtered_data2.groupby(
        'in_pldi').size().reset_index(name='count')
    webrobot_extended_counts = [compute_percent(x, y) for (x, y) in
                                [
        (counts2['count'].sum(), total),
        (counts2['count'][1], pldi_n),
        (counts2['count'][0], total - pldi_n)
    ]
    ]
    # parse wr data
    filtered_data3 = parse_wr_data(data3, pldi_names, include)
    counts3 = filtered_data3.groupby(
        'in_pldi').size().reset_index(name='count')
    webrobot_counts = [compute_percent(x, y) for (x, y) in
                       [
        (counts3['count'].sum(), total),
        (counts3['count'][1], pldi_n),
        (counts3['count'][0], total - pldi_n)
    ]
    ]

    # Filter the DataFrame to include only rows where intend or wr_solved is "Y"
    # print(len(data1))
    filtered_data = data1[(data1['intend'] == 'Y')]
    # print(len(filtered_data))
    counts1 = filtered_data.groupby('in_pldi').size().reset_index(name='count')
    # print(counts1)
    # print(total - 76)
    arborist_counts = [compute_percent(x, y) for (x, y) in
                       [
        (counts1['count'].sum(), total),
        (counts1['count'][1], pldi_n),
        (counts1['count'][0], total - pldi_n)
    ]
    ]

    # Extract the values for plotting
    x_labels = ['All', 'In PLDI', 'New']

    # Set the positions and width of the bars
    bar_width = 0.25
    r1 = range(len(x_labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the bar chart
    plt.bar(r1, arborist_counts, color='black', edgecolor='black',
            hatch='', width=bar_width, label='Arborist')
    plt.bar(r2, webrobot_counts, color='none', edgecolor='black',
            hatch='..', width=bar_width, label='WR')
    plt.bar(r3, webrobot_extended_counts, color='white', edgecolor='black',
            hatch='', width=bar_width, label='WR-extended')
    # Remove the x-axis label
    plt.xlabel('')

    # Set the x-axis tick positions and labels
    x_tick_positions = [r for r in r2]
    x_tick_labels = [f'All ({total})', f'Prior({pldi_n})', f'New ({total - pldi_n})']
    plt.xticks(x_tick_positions, x_tick_labels, fontsize=16)

    # Set the y-axis label
    # Position y-axis label at the top
    plt.gca().yaxis.set_label_coords(-.1, 1.04)
    plt.ylabel('% of benchmarks solved',
               fontsize=16, rotation=0, ha='left')
    y_tick_positions = list(range(25, 101, 25))
    y_tick_labels = [str(x) for x in y_tick_positions]
    y_tick_labels[-1] = '100%'
    plt.ylim(30, plt.ylim()[1])
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=15)

    # Customize the legend
    legend = plt.legend(loc='upper center', bbox_to_anchor=(
        0.45, 1.26), ncol=3, fontsize=16)
    legend.get_frame().set_linewidth(0)  # Remove legend border lines

    # Remove right and top border lines of the main graph
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Show the numbers on the bars
    for i, v in enumerate(arborist_counts):
        plt.text(i - 0.11, v + 1, str(v), fontsize=13)
    for i, v in enumerate(webrobot_counts):
        plt.text(i + bar_width - 0.11, v + 1, str(v), fontsize=13)
    for i, v in enumerate(webrobot_extended_counts):
        plt.text(i + 2*bar_width - 0.11, v + 1, str(v), fontsize=13)

    # crop the figure
    plt.tight_layout()
    # fig, ax = plt.subplots()
    # set_figure_size(fig, 8, 5.5)
    plt.subplots_adjust(top=.8, bottom=0.06, right=0.98, left=0.165)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/RQ1-benchmarks-solved.pdf', bbox_inches='tight')

    # Show the plot
    # plt.show()
    plt.close()


def exp1b_plot(data1, data2, data3):
    '''
    given the same data with column "max" having the max time cost each benchmarks takes
    For each category draw a box plot with x axis be the category and y axis be the max time cost
    if max time is larger that 1.0 cap it to 1.0
    The plot should be grouped as before by in_pldi, and each group should have two boxes for
    Arborist and WebRobot.
    '''
    # parse wr data
    pldi_names = data1[data1['in_pldi'] == 'Y']['name']
    data1 = data1[(data1['intend'] == 'Y') | (data1['intend'] == 'N') | (
        data1['intend'] == 'YL') | (data1['in_pldi'] == 'Y')]
    include = data1['name'].values
    total = len(include)
    pldi_n = len(data1[data1['in_pldi'] == 'Y'])
    print("pldi_n: " + str(pldi_n))
    # pldi_n = 76
    print(f"total number of included benchmarks: {total}")
    # print(data2['intended?'])
    filtered_data2 = parse_wr_data(data2, pldi_names, include)
    filtered_data3 = parse_wr_data(data3, pldi_names, include)

    # Cap max time cost at 1.0
    data1['max'] = data1['max'].clip(upper=1.0)
    data1['timeout'] = data1['timeout'].fillna('Y')
    data1.loc[data1['timeout'] != 'Y', 'max'] = 1.0
    filtered_data = data1[(data1['intend'] == 'Y') |
                          (data1['wr_solved'] == 'Y')]

    # Filter the data for each group
    in_pldi_data = filtered_data[filtered_data['in_pldi'] == 'Y']
    new_data = filtered_data[filtered_data['in_pldi'] == 'N']
    all_data = filtered_data
    data2_in_pldi = filtered_data2[filtered_data2['in_pldi'] == 'Y']
    data2_new = filtered_data2[filtered_data2['in_pldi'] == 'N']
    data2_all = filtered_data2
    data3_in_pldi = filtered_data3[filtered_data3['in_pldi'] == 'Y']
    data3_new = filtered_data3[filtered_data3['in_pldi'] == 'N']
    data3_all = filtered_data3

    # Define the groups
    groups = ['All', 'Prior', 'New']
    # Initialize lists to store the box plot data for Arborist and WebRobot
    arborist_data = []
    webrobot_data = []
    webrobot_extended_data = []

    # Iterate over each group
    for group_name in groups:
        if group_name == 'Prior':
            arborist_data.append(
                compute_log(in_pldi_data[in_pldi_data['intend'] == 'Y']['max']))
            webrobot_data.append(compute_log(data3_in_pldi['longest time']))
            webrobot_extended_data.append(
                compute_log(data2_in_pldi['longest time']))
        elif group_name == 'New':
            arborist_data.append(compute_log(
                new_data[new_data['intend'] == 'Y']['max']))
            webrobot_data.append(compute_log(data3_new['longest time']))
            webrobot_extended_data.append(
                compute_log(data2_new['longest time']))
        elif group_name == 'All':
            arborist_data.append(compute_log(
                all_data[all_data['intend'] == 'Y']['max']))
            webrobot_data.append(compute_log(data3_all['longest time']))
            webrobot_extended_data.append(
                compute_log(data2_all['longest time']))
   # Create the figure and axes
    # fig, ax = plt.subplots()
    # print(ax)

    # Set the positions for the box plots
    positions = [i * 2 for i in range(len(groups))]
    # print(arborist_data)
    # webrobot_extended_boxes = plt.boxplot(webrobot_extended_data, positions=[p + 1 for p in positions], widths=0.4, patch_artist=True,
    #                                      boxprops=dict(facecolor='lightcoral', edgecolor='black'), medianprops=dict(color='black'), flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4))
    arborist_boxes = plt.boxplot(arborist_data, positions=positions, widths=0.4, patch_artist=True,
                                 boxprops=dict(facecolor='black', color='black'), medianprops=dict(color='red', linewidth=2),
                                 flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='none', markersize=4))

    # Plot the box plots for WebRobot
    webrobot_boxes = plt.boxplot(webrobot_data, positions=[p + 0.5 for p in positions], widths=0.4, patch_artist=True,
                                 boxprops=dict(facecolor='none', edgecolor='black', hatch='..'), medianprops=dict(color='red', linewidth=2),
                                 flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='none', markersize=4))

    webrobot_extended_boxes = plt.boxplot(webrobot_extended_data, positions=[p + 1 for p in positions], widths=0.4, patch_artist=True,
                                          boxprops=dict(
                                              facecolor='none', edgecolor='black', hatch=''),
                                          medianprops=dict(color='red', linewidth=2), flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='none', markersize=4))

    # outlier_style = dict(marker='.', markerfacecolor='black', markersize=3, linestyle='none')
    # for boxes in [arborist_boxes, webrobot_boxes]:
    #     for flier in boxes['fliers']:
    #         flier.set(**outlier_style)

    font_size = 18

    # Set the x-axis limits and labels
    x_tick_positions = [p + 0.5 for p in positions]
    x_tick_labels = [f'All ({total})', f'Prior({pldi_n})', f'New ({total - pldi_n})']
    plt.xticks(x_tick_positions, x_tick_labels, fontsize=font_size)

    # Adjust the position of the y-axis label
    plt.gca().yaxis.set_label_coords(-0.05, 1.04)
    plt.ylabel('Solving time (seconds)', fontsize=font_size, rotation=0, ha='left')

    # Format y-axis tick labels in scientific notation
    # y_formatter = LogFormatterSciNotation(base=2)
    # plt.gca().yaxis.set_major_formatter(y_formatter)
    # Format y-axis tick labels as power of 2
    y_ticks = plt.gca().get_yticks()
    y_ticks = np.linspace(-3.0, 0.0, 4)
    y_labels = [r'$10^{{{}}}$'.format(int(y_tick)) for y_tick in y_ticks]
    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels(y_labels, fontsize=font_size)

    # webrobot_extended_legend = patches.Patch(color='lightcoral', label='WebRobot-extended')
    arborist_legend = patches.Patch(
        facecolor='black', edgecolor='black', hatch='', label='Arborist')
    webrobot_legend = patches.Patch(
        facecolor='white', edgecolor='black', hatch='..', label='WR')
    webrobot_extended_legend = patches.Patch(
        facecolor='white', edgecolor='black', hatch='', label='WR-extended')
    legend = plt.legend(handles=[arborist_legend, webrobot_legend, webrobot_extended_legend],
                        bbox_to_anchor=(1.07, 1.25), ncol=3, fontsize=font_size,
                        edgecolor='black')
    # Customize the legend
    # legend = plt.legend(loc='upper center', bbox_to_anchor=(
    #         0.55, 1.17), ncol=3, fontsize=16)
    legend.get_frame().set_linewidth(0)  # Remove legend border lines

    # Remove right and top border lines of the main graph
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # crop the figure
    plt.subplots_adjust(top=.905, bottom=0.06, right=0.96, left=0.105)

    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/RQ1-synthesis-times.pdf', bbox_inches='tight')
    # Show the plot
    # plt.show()
    plt.close()


def ablation1_plot(data):
    # Filter out rows with timeout = "Y"
    data_len = len(data['name'].unique())
    print(f"total number of benchmarks: {data_len}")
    print(len(list(data['name'].unique())))

    data.fillna(value={'timeout': 1}, inplace=True)
    data['timeout'] = data['timeout'].apply(lambda x: 1 if x == 1 else 0)
    # data = data[data['timeout'] == 'N']
    print(data[(data['name'] == "W8T3")]
          [["sample_selectors", "timeout"]].to_string())
    # print(filtered_data)
    data['timeout'] = data.groupby(['seed', 'sample_selectors'])[
        'timeout'].transform('sum')
    print(data)
    # Looping through each group and its rows
    # for _, group_data in grouped_data:
    #     if not all(x == group_data['timeout'].values[0] for x in group_data['timeout'].values):
    #         print(group_data[['name', 'sample_selectors', 'timeout']].to_string())

    # best_data = data.groupby(['name', 'sample_selectors']).apply(lambda x: 1 if 'N' in x['timeout'].values else 0).reset_index(name='best')
    # worst_data = data.groupby(['name', 'sample_selectors']).apply(lambda x: 0 if 'Y' in x['timeout'].values else 1).reset_index(name='worst')
    # mean_data = data.groupby(['name', 'sample_selectors'])['timeout'].apply(lambda x: (x == 'N').mean()).reset_index(name='mean')
    # data = best_data.merge(worst_data, on=['name', 'sample_selectors'])
    # data = data.merge(mean_data, on=['name', 'sample_selectors'])

    # Group the data by sample_selectors and count the number of benchmarks
    grouped_data = data.groupby(['sample_selectors']).agg(
        {'timeout': ['max', 'min', 'mean']}).reset_index()
    grouped_data.columns = ['sample_selectors', 'max', 'min', 'mean']
    print(grouped_data)

    # Sort the grouped data by sample_selectors in ascending order
    sorted_data = grouped_data.sort_values('sample_selectors')

    print(sorted_data)
    # change best worst and mean to percentage (/ data_len)
    sorted_data['max'] = sorted_data['max'].apply(
        lambda x: compute_percent(x, data_len))
    sorted_data['min'] = sorted_data['min'].apply(
        lambda x: compute_percent(x, data_len))
    sorted_data['mean'] = sorted_data['mean'].apply(
        lambda x: compute_percent(x, data_len))
    
    # print the first data
    # print(f"first max: {sorted_data['max'][0]}")
    # print(f"first min: {sorted_data['min'][0]}")
    # print(f"first mean: {sorted_data['mean'][0]}")
    sorted_data['max'][0] = 100.0
    sorted_data['min'][0] = 100.0
    sorted_data['mean'][0] = 100.0

    print("percentage table:")
    print(sorted_data)

    fig, ax = plt.subplots()
    set_figure_size(fig, 10, 5)
    # Generate the line chart
    # plt.plot(sorted_data['sample_selectors'],
    #          sorted_data['count'], color='black')
   # Adding vertical lines
    for i in range(len(sorted_data['sample_selectors'])):
        plt.plot([sorted_data['sample_selectors'][i], sorted_data['sample_selectors'][i]],
                 [sorted_data['min'][i], sorted_data['max'][i]], color='black', linestyle='-', linewidth=1.0)

        # add horizontal lines
        plt.plot([sorted_data['sample_selectors'][i] - 120, sorted_data['sample_selectors'][i] + 120],
                 [sorted_data['max'][i], sorted_data['max'][i]], color='black', linestyle='-', linewidth=1.2)
        plt.plot([sorted_data['sample_selectors'][i] - 120, sorted_data['sample_selectors'][i] + 120],
                 [sorted_data['min'][i], sorted_data['min'][i]], color='black', linestyle='-', linewidth=1.2)

    font_size = 21

    # plt.plot(sorted_data['sample_selectors'],
    #          sorted_data['best'], color='black')
    # plt.plot(sorted_data['sample_selectors'],
    #          sorted_data['worst'], color='black')
    plt.plot(sorted_data['sample_selectors'],
             sorted_data['mean'], color='black')
    plt.scatter(sorted_data['sample_selectors'],
                sorted_data['mean'], color='black', marker='.', s=80)

    plt.xlabel('# of candidate selectors', fontsize=font_size+2)
    plt.gca().xaxis.set_label_coords(0.5, -0.1)

    plt.gca().yaxis.set_label_coords(-0.05, 1.04)
    plt.ylabel('% of benchmarks exhausted (total 131)',
               fontsize=font_size+2, rotation=0, ha='left')

    y_tick_positions = list(range(50, 101, 10))
    y_tick_labels = [str(x) for x in y_tick_positions]
    y_tick_labels[-1] = '100%'
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=font_size+2)
    plt.xticks(fontsize=font_size-2)

    # plt.xticks(range(0.0, 1.0, 0.1))
    plt.xticks([0] + [x * 1000 for x in range(1, 11)])
    plt.grid(False)
    # Remove right and top border lines of the main graph
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # crop the figure
    plt.tight_layout()
    plt.subplots_adjust(top=.93, bottom=0.1, right=0.92, left=0.19)
    plt.savefig('./figures/RQ3-exhausted-count.pdf', bbox_inches='tight')
    # plt.show()


def ablation2_plot(data, data2, data3, data4):
    for_data_benchmarks = ["W7T1",
                           "W7T2",
                           "W9T1",
                           "W18T1",
                           "W46T1",
                           "W50T1",
                           "W52T1",
                           "W81T1",
                           "W111T1",
                           "W111T2",
                           "W120T1",
                           "W124T1",
                           "W125T1",
                           "W127T1",
                           "W141T1",
                           "W146T1",
                           "W146T2",
                           "W157T1",
                           "W157T2",
                           "W173T1",
                           "W177T1",
                           "W188T1",
                           "W190T1",
                           "W213T1",
                           "W233T1",
                           "W237T1",
                           "W239T1",
                           "W239T2",
                           "W253T1",
                           "W254T1",
                           "W276T1",
                           "W285T1",
                           "W287T1",
                           "W287T2"]

    # filter all rows
    single_loop_fordata = data[(data['name'].isin(for_data_benchmarks)) & (
        data['loop_depth'] == 1)]['name'].unique().tolist()
    print(f"{data[(data['name'].isin(for_data_benchmarks)) & (data['loop_depth'] == 1)]['name'].unique().tolist()}")
    # combine the segments of data
    # data2 = data2[data2['sample_selectors'] != 41]
    data_1 = data[data['sample_selectors'] == 1]
    data_51 = data[data['sample_selectors'] == 51]
    combined_1 = combine_data(data_1, data2)
    combined_2 = combine_data(combined_1, data_51)
    combined_3 = combine_data(combined_2, data3)
    combined_4 = combine_data(combined_3, data)
    combined_5 = combine_data(combined_4, data4)
    # concatinate data and data2
    data = combined_5

    print(f"total number of benchmarks: {len(data['name'].unique())}")
    print("depth of loops > 1: ", len(
        data[data['loop_depth'] > 1]['name'].unique()))
    print("depth of loops == 1: ", len(
        data[data['loop_depth'] == 1]['name'].unique()))
    print("number of parametriazable instructions > 1 and loop depth > 1: ", len(
        data[(data['n_parametrizable'] > 1) & (data['loop_depth'] > 1)]['name'].unique()))
    print("number of parametriazable instructions > 1: ", len(
        data[data['n_parametrizable'] > 1]['name'].unique()))
    print("number of parametriazable instructions > 2: ", len(
        data[data['n_parametrizable'] > 2]['name'].unique()))

    # filter out benchmarks that are not for data benchmarks
    # data = data[(~data['name'].isin(for_data_benchmarks))]
    # data = data[(~data['name'].isin(single_loop_fordata))]
    # data = data[data['n_parametrizable'] > 1]
    # filter out data with NGT or NDSL
    # data = data[(data['intend'] != 'NGT') & (data['intend'] != 'NDSL')]

    print(
        f"number of benchmarks without send_data: {len(data['name'].unique())}")

    data_len = len(data['name'].unique())
    print(f"total number of benchmarks: {data_len}")
    # print all time out benchmarks
    data.fillna(value={'timeout': 'N'}, inplace=True)
    data['timeout'] = data['timeout'].apply(lambda x: 'N' if x == 'N' else 'Y')
    print(
        f"time out benchmarks: {data[data['timeout'] == 'Y']['name'].unique().tolist()}")
    print("----")

    # for _, row in data.iterrows():
    #     if row['intend'] == 'Y' or row['timeout'] == 'Y':
    #         for i in range(row['sample_selectors'] + 50, 1001, 50):
    #             print(f"sample_selectors: {i}, name: {row['name']}")
    #             for j in range(0, 8):
    #                 new_row = pd.DataFrame({'name': row['name'], 'sample_selectors': [i], 'intend': row['intend']})
    #                 data = pd.concat([data, new_row], ignore_index=True)
    data['intend'] = data['intend'].apply(lambda x: 1 if x == 'Y' else 0)
    data = data.groupby(['sample_selectors', 'seed']).agg({'intend': 'sum'})
    data = data.sort_values('sample_selectors')
    print(data.to_string())
    data = data.groupby(['seed']).agg({'intend': 'cumsum'}).reset_index()
    print(data)

    # group by sample_selectors and compute sum of best and worst and mean
    grouped_data = data.groupby(['sample_selectors']).agg(
        {'intend': ['max', 'min', 'mean']}).reset_index()
    grouped_data.columns = ['sample_selectors', 'max', 'min', 'mean']
    print(grouped_data)
    # Sort the grouped data by n_selectors in ascending order
    sorted_data = grouped_data.sort_values('sample_selectors')

    # Create a new row with sample_rate = 0 and count = 0
    # new_row = pd.DataFrame({'sample_selectors': [0.0], 'count': [0]})

    print(sorted_data)

    # change best worst and mean to percentage (/ data_len)
    sorted_data['max'] = sorted_data['max'].apply(
        lambda x: compute_percent(x, data_len))
    sorted_data['min'] = sorted_data['min'].apply(
        lambda x: compute_percent(x, data_len))
    sorted_data['mean'] = sorted_data['mean'].apply(
        lambda x: compute_percent(x, data_len))

    print("percentage table:")
    print(sorted_data)

    fig, ax = plt.subplots()
    set_figure_size(fig, 15, 5)

    # Generate the line chart
    for i in range(len(sorted_data['sample_selectors'])):
        plt.plot([sorted_data['sample_selectors'][i], sorted_data['sample_selectors'][i]],
                 [sorted_data['min'][i], sorted_data['max'][i]], color='black', linestyle='-', linewidth=1.0)

        # add horizontal lines
        plt.plot([sorted_data['sample_selectors'][i] - 8, sorted_data['sample_selectors'][i] + 8],
                 [sorted_data['max'][i], sorted_data['max'][i]], color='black', linestyle='-', linewidth=1.2)
        plt.plot([sorted_data['sample_selectors'][i] - 8, sorted_data['sample_selectors'][i] + 8],
                 [sorted_data['min'][i], sorted_data['min'][i]], color='black', linestyle='-', linewidth=1.2)

    # plt.plot(sorted_data['sample_selectors'],
    #          sorted_data['best'], color='black')
    # plt.plot(sorted_data['sample_selectors'],
    #          sorted_data['worst'], color='black')
    plt.plot(sorted_data['sample_selectors'],
             sorted_data['mean'], color='black')
    plt.scatter(sorted_data['sample_selectors'],
                sorted_data['mean'], color='black', marker='.', s=80)

    font_size = 22
    plt.xlabel('# of candidate selectors', fontsize=font_size)
    # plt.ylabel('Solve Rate', fontsize=14)
    plt.gca().yaxis.set_label_coords(-0.02, 1.05)
    plt.ylabel('% of benchmarks solved (total 131)',
               fontsize=font_size, rotation=0, ha='left')
    y_tick_positions = list(range(0, 101, 10))
    y_tick_labels = [str(x) for x in y_tick_positions]
    y_tick_labels[-1] = '100%'
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=font_size)
    plt.xticks(range(0, 1501, 100), fontsize=font_size)
    plt.gca().xaxis.set_label_coords(0.5, -0.12)
    # plt.xticks([x * 10 for x in range(0, 11)])
    plt.grid(False)
    # Remove right and top border lines of the main graph
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig('./figures/RQ2-solved-benchmarks.pdf', bbox_inches='tight')
    plt.subplots_adjust(top=.915, bottom=0.14, right=0.94, left=0.115)

    # print statistics

    # plt.show()


def parse_wr_data(data, pldi_names, include):
    # print(data2['intended?'])
    # print(include)
    filtered_data = data[(data['intended?'] == True) &
                         (data['benchmark ID'].isin(include))]
    filtered_data['in_pldi'] = [
        "Y" if name in pldi_names.values else "N" for name in filtered_data["benchmark ID"]]
    filtered_data['longest time'] = filtered_data['longest time'].apply(
        lambda x: x / 1000.0).clip(upper=1.0)
    return filtered_data


def ablation1b_plot(data):
    # Filter out rows with timeout = "Y"
    data_len = len(data['name'].unique())
    print(f"total number of benchmarks: {data_len}")
    print(len(list(data['name'].unique())))

    data.fillna(value={'timeout': 1}, inplace=True)
    data['timeout'] = data['timeout'].apply(lambda x: 1 if x == 1 else 0)
    data = data[data['timeout'] == 1]
    data = data[data['sample_selectors'] > 1]
    # data['max_time'] = data.groupby(['seed', 'sample_selectors'])['max'].transform('sum')
    print(data)

    # Group the data by column x
    # data['max'] = data['max'].apply(lambda x: math.log2(x))
    grouped_data = data.groupby('sample_selectors')['max']

    # Create an empty list to store the box plot artists
    boxplot_artists = []

    fig, ax = plt.subplots()
    set_figure_size(fig, 10, 5)

    # Iterate over the groups and create a box plot for each group
    for group, group_data in grouped_data:
        boxplot_artist = plt.boxplot(group_data, positions=[group], widths=360, patch_artist=True,
                                     boxprops=dict(facecolor='none', color='black'), medianprops=dict(color='black'),
                                     showfliers=False)

        boxplot_artists.append(boxplot_artist)

    # Customize the appearance of the plot
    # plt.xticks(range(1, len(grouped_data.groups) + 1), grouped_data.groups)
    plt.grid(True)

    font_size = 25

    plt.xlabel('# of candidate selectors', fontsize=font_size)
    ticks = [1000 * x for x in range(1, 11)]
    x_tick_positions = [x * 1000 for x in range(1, 11)]
    plt.xticks(x_tick_positions, ticks, fontsize=font_size-2)
    plt.gca().xaxis.set_label_coords(0.5, -0.1)

    y_formatter = LogFormatterSciNotation(base=10)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(y_formatter)

    plt.ylabel('Exhaustion time (seconds)', fontsize=font_size, rotation=0, ha='left')
    plt.gca().yaxis.set_label_coords(-0.05, 1.05)
    # y_tick_positions = list(range(0, 8, 1))
    y_tick_positions = [0.001, 0.01, 0.1, 1,  5, 10]
    # y_tick_labels = [str(float(x)) for x in y_tick_positions]
    y_tick_labels = y_tick_positions
    y_tick_labels = [r'$10^{{{}}}$'.format(int(-3)), r'$10^{{{}}}$'.format(int(-2)), r'$10^{{{}}}$'.format(
        int(-1)), r'${{{}}}$'.format(int(1)), r'${{{}}}$'.format(int(5)), r'${{{}}}$'.format(int(10))]
    # y_tick_labels = [r'$10^{{{}}}$'.format(int(y_tick)) for y_tick in [-3, -2, -1, 0, 1, 2]]
    # y_tick_labels.append(["1", "5", "10"])
    # ax.set_yticks(y_tick_positions)
    # ax.set_yticklabels(y_tick_labels)
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=font_size)

    plt.grid(False)
    # Remove right and top border lines of the main graph
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # crop the figure
    plt.tight_layout()
    # fig, ax = plt.subplots()
    # set_figure_size(fig, 8, 5.5)
    plt.subplots_adjust(top=.928, bottom=0.087, right=0.95, left=0.125)
    plt.savefig('./figures/RQ3-exhausted-time.pdf', bbox_inches='tight')
    # plt.show()


# RQ1
exp1a_plot(arborist_data, wr_extended_data, wr_original_data)
exp1b_plot(arborist_data, wr_extended_data, wr_original_data)

# RQ2
# ablation2_plot(arborist_sparsity_data, arborist_sparsity_data2,
#                arborist_sparsity_data3, arborist_sparsity_data4)

# RQ3
# ablation1_plot(arborist_scale_data)
# ablation1b_plot(arborist_scale_data)
