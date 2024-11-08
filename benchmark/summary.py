import numpy as np
import pandas as pd


def calulcate_distance(df):
    print(df['pos_x'])


def consolidate_results(df):
    results = {}
    not_stopped_percent = df[df['speed'] > 0.1].count()[0] / len(df.index) * 100

    # Is successful
    results['successfull'] = df['reached_destination'].values[-1]

    # Collisions
    results['n_predistrain_collision'] = sum(
        df['collision_predistrain'].diff().dropna() > 0
    )
    results['n_vehicle_collisions'] = sum(df['collision_vehicle'].diff().dropna() > 0)
    results['n_other_collisions'] = sum(df['collision_other'].diff().dropna() > 0)
    results['n_collisions'] = sum(((df['n_collisions'] > 0) * 1).diff().dropna() > 0)
    # results['n_collisions'] = df['n_collisions'].values[-1]

    if (not_stopped_percent < 70.0) and not results['successfull']:
        results['not_stalled'] = False
    else:
        results['not_stalled'] = True

    # Lane invasions
    results['n_lane_invasion'] = sum(df['lane_invasion'].diff().dropna() > 0)

    # Percentage completed
    results['route_completed'] = (
        (df['len_path_points'].values[0] - df['len_path_points'].values[-1] + 1)
        / df['len_path_points'].values[0]
        * 100
    )

    # Distance travelled

    squared_sum = df['pos_x'].diff().dropna() ** 2 + df['pos_y'].diff().dropna() ** 2
    results['distance_travelled'] = np.sqrt(squared_sum).sum()

    infractions = (
        +results['n_vehicle_collisions']
        + results['n_predistrain_collision']
        # + results['n_other_collisions']
        + results['n_lane_invasion']
    )
    results['infractions'] = infractions * (1000 / results['distance_travelled'])

    return results


def summarize(read_path):
    df = pd.read_csv(read_path)

    new_labels = ['iteration', 'exp_id', 'navigation_type']
    df = df.set_index(keys=new_labels)
    indices = df.index.unique()

    results = []
    for index in indices:
        df_temp = df.loc[index]
        results.append(consolidate_results(df_temp))

    final_summary = pd.DataFrame.from_dict(results, orient='columns')

    # Drop stalled episodes and clean up some
    final_summary = final_summary[final_summary['not_stalled']]

    print(final_summary.describe())
    print('-' * 32)
    sucessful_episodes = np.sum(final_summary['successfull'] * 1) / len(
        final_summary.index
    )
    print(f'% of successfully completed episodes={sucessful_episodes}')
