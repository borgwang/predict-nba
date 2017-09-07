# author: wangguibo <borgwang@126.com>
# date: 2017-8-28
# Copyright (c) 2017 by wangguibo
#
# file: preprocess.py
# desc: Prepare dataset for predicting task
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import runtime_path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.utils import *

# CONFIG
WARM_START = 20
K = 12  # last k games


def merge_teams(d, old_name, new_name):
    d[new_name] = dict(d[old_name], **d[new_name])
    d.pop(old_name)
    return d


def reduce_features(features, reduce_dim=8):
    # reduce features dimensions
    # print('reducing features dimensions...')
    features = (features.T - np.mean(features, 0)[:, None]).T
    features = (features.T / (np.std(features, 0) + 1e-6)[:, None]).T
    # return PCA(n_components=reduce_dim).fit_transform(features)
    return features


def get_record_stats():
    team_schedules_path = DATASET_BASE_PATH + 'team_schedules.p'
    team_schedules = load_from_file(team_schedules_path, 'team_schedules')

    # handle teams that change names
    for old_name, new_name in NAME_CHANGES.items():
        merge_teams(team_schedules, old_name, new_name)

    # game result features
    w_pct, last_k_w_pct, home_w_pct, raod_w_pct = {}, {}, {}, {}
    pts_diff = {}
    for team, seasons in team_schedules.items():
        _w_pct, _last_k_w_pct, _home_w_pct, _raod_w_pct = {}, {}, {}, {}
        _pts_diff = {}
        years = sorted(seasons.keys())
        for year_idx in range(1, len(years)):
            # last 20 games of last season as warmstart_games
            warmstart_games = seasons[years[year_idx - 1]][-WARM_START:]
            games = np.vstack((warmstart_games, seasons[years[year_idx]]))
            for i, g in enumerate(games[WARM_START:]):
                s_idx = 0 if i < WARM_START else WARM_START
                e_idx = i + WARM_START
                num_games = e_idx - s_idx
                game_date = games[e_idx][0]
                prev_games = games[s_idx:e_idx]
                # overall win percentage
                _w_pct[game_date] = \
                    len(np.where(prev_games[:, 3] == 'W')[0]) / num_games
                # win percentage at home
                home_games = np.where(prev_games[:, 2] == 'home')[0]
                home_w_games = np.where(np.logical_and(
                    prev_games[:, 2] == 'home',
                    prev_games[:, 3] == 'W'))[0]
                _home_w_pct[game_date] = len(home_w_games) / len(home_games)
                # win percentage on the raod
                raod_game = np.where(prev_games[:, 2] == 'visit')[0]
                raod_w_games = np.where(np.logical_and(
                    prev_games[:, 2] == 'visit',
                    prev_games[:, 3] == 'W'))[0]
                _raod_w_pct[game_date] = len(raod_w_games) / len(raod_game)
                # mean point differential
                ht_pts = prev_games[:, 4].astype(float)
                rt_pts = prev_games[:, 5].astype(float)
                _pts_diff[game_date] = np.mean(ht_pts - rt_pts)
                # win percentage of last k games
                last_k_games = games[e_idx - K:e_idx]
                _last_k_w_pct[game_date] = \
                    len(np.where(last_k_games[:, 3] == 'W')[0]) / K

        w_pct[team] = _w_pct
        last_k_w_pct[team] = _last_k_w_pct
        home_w_pct[team] = _home_w_pct
        raod_w_pct[team] = _raod_w_pct
        pts_diff[team] = _pts_diff

    return [w_pct, last_k_w_pct, home_w_pct, raod_w_pct, pts_diff]


def get_game_stats():
    # team stats features
    team_stats_path = DATASET_BASE_PATH + 'team_performance.p'
    team_stats = load_from_file(team_stats_path, 'team_performance')
    for game_id, stat in team_stats.items():
        date = stat[0]
        season = get_season_by_date(date)
        team_stats[game_id].insert(1, season)
    team_stats = np.asarray(team_stats.values(), dtype=object)
    info_part = team_stats[:, :5].astype(str)

    # handle teams that change names
    for old_name, new_name in NAME_CHANGES.items():
        info_part[info_part == old_name] = new_name

    h_stats = np.asarray(
        [np.asarray(s) for s in team_stats[:, 6]])
    r_stats = np.asarray(
        [np.asarray(s) for s in team_stats[:, 5]])
    recent_stats, cumulated_stats = {}, {}
    for team in TEAM_NAMES:
        if team in NAME_CHANGES.keys():
            continue
        h_idx = np.where(info_part[:, 3] == team)[0]
        date = info_part[h_idx, 0][:, None]
        season = info_part[h_idx, 1][:, None]
        h_stat = h_stats[h_idx]
        h_concat = np.hstack((
            date, season, np.array(['home'] * len(h_idx))[:, None],
            h_stat))

        r_idx = np.where(info_part[:, 4] == team)[0]
        date = info_part[r_idx, 0][:, None]
        season = info_part[r_idx, 1][:, None]
        r_stat = r_stats[r_idx]
        r_concat = np.hstack((
            date, season, np.array(['road'] * len(r_idx))[:, None], r_stat))

        # concat date, game position and stats (and then sort by date)
        arr = np.vstack((h_concat, r_concat)).astype(str)
        arr = arr[np.argsort(arr[:, 0])]

        h_idx = np.where(arr[:, 2] == 'home')[0]
        r_idx = np.where(arr[:, 2] == 'road')[0]

        date_to_recent_stat = {}
        for i in range(len(arr))[::-1]:
            date = arr[i][0]
            if i < 20:
                continue
            last_k_h_game_idx = h_idx[h_idx < i][-3:]
            last_k_h_game_feature = np.mean(
                arr[last_k_h_game_idx, 3:].astype(float), 0)

            last_k_r_game_idx = r_idx[r_idx < i][-3:]
            last_k_r_game_feature = np.mean(
                arr[last_k_r_game_idx, 3:].astype(float), 0)

            date_to_recent_stat[date] = {
                'home': last_k_h_game_feature,
                'road': last_k_r_game_feature}

        date_to_cumulated_stat = {}
        for i in range(len(arr))[::-1]:
            date = arr[i][0]
            if i < 20:
                continue
            season = arr[i][1]
            season_game_idx = np.where(arr[:, 1] == season)[0]
            prev_game_idx = season_game_idx[season_game_idx < i]
            if len(prev_game_idx) < 10:
                cumulated_stat = np.mean(arr[i - 10:i, 3:].astype(float), 0)
            else:
                cumulated_stat = np.mean(
                    arr[prev_game_idx, 3:].astype(float), 0)
            date_to_cumulated_stat[date] = cumulated_stat

        recent_stats[team] = date_to_recent_stat
        cumulated_stats[team] = date_to_cumulated_stat

    return recent_stats, cumulated_stats


def get_odds_stats():
    game_odds_path = DATASET_BASE_PATH + 'game_odds.p'
    game_odds = load_from_file(game_odds_path, 'game_odds', dtype='list')

    return game_odds


def laod_games():
    # load game info
    game_info_path = DATASET_BASE_PATH + 'game_info.p'
    game_info = load_from_file(game_info_path, 'game_info')
    # handle teams that change names
    for year_g in game_info.values():
        for month_g in year_g.values():
            for old_name, new_name in NAME_CHANGES.items():
                month_g[month_g == old_name] = new_name

    games = []
    for year_g in game_info.values():
        for month_g in year_g.values():
            games.append(month_g)
    games = np.vstack(games)

    # filter games
    games = games[np.where(games[:, 1] == 'regular')[0]]
    games = games[np.where(games[:, 0] > '20080900')[0]]
    print('load %d games in total' % len(games))

    return games


def construct_dataset(record_stats, recent_stats, cumulated_stats, odds_stats):
    games = laod_games()
    dataset = []
    w_pct, last_k_w_pct, home_w_pct, raod_w_pct, pts_diff = record_stats
    record_features, stat_features, odds_features, labels = [], [], [], []

    # construct features
    for g in games:
        date, ht, rt, ht_pts, rt_pts = g[0], g[2], g[3], g[4], g[5]
        ht_w_pct = w_pct[ht][date]
        rt_w_pct = w_pct[rt][date]
        ht_pts_diff = pts_diff[ht][date]
        rt_pts_diff = pts_diff[rt][date]
        ht_last_k_w_pct = last_k_w_pct[ht][date]
        rt_last_k_w_pct = last_k_w_pct[rt][date]
        ht_home_c_pct = home_w_pct[ht][date]
        rt_road_c_pct = raod_w_pct[rt][date]
        # record_feature
        record_feature = [ht_w_pct, rt_w_pct, ht_pts_diff, rt_pts_diff,
                          ht_last_k_w_pct, rt_last_k_w_pct, ht_home_c_pct,
                          rt_road_c_pct]

        # recent_stat_feature
        ht_last_k_stat = recent_stats[ht][date]['home']
        rt_last_k_stat = recent_stats[rt][date]['road']

        # cumulated_stat_feature
        ht_cum_stat = cumulated_stats[ht][date]
        rt_cum_stat = cumulated_stats[rt][date]

        recent_feature = ht_last_k_stat - rt_last_k_stat
        cum_feature = ht_cum_stat - rt_cum_stat

        # stat_feature(combine recent and cumulate)
        stat_feature = recent_feature + cum_feature

        # odds_features
        game_idx = np.where(np.logical_and(
            odds_stats[:, 0] == date, odds_stats[:, 1] == ht))[0]
        game_idx = game_idx[0]

        if odds_stats[game_idx, 3] == 'NL' or odds_stats[game_idx, 4] == 'NL' \
                or math.isnan(float(odds_stats[game_idx, 3])):
            print(date, ht, rt)
            continue
        odds_features.append(odds_stats[game_idx, [3, 4]])
        record_features.append(record_feature)
        stat_features.append(stat_feature)

        # labels
        label = 1.0 if float(ht_pts) > float(rt_pts) else 0.0
        labels.append(label)

    # construct dataset
    labels = np.asarray(labels).reshape((-1, 1))
    record_features = np.asarray(record_features, dtype=float)
    stat_features = np.asarray(stat_features, dtype=float)
    odds_features = np.asarray(odds_features, dtype=float)

    stat_features = reduce_features(stat_features)
    
    # # feature correlate coefficient
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(34, 34))
    # corr = np.corrcoef(cum_features, rowvar=False)
    # sns.heatmap(corr, annot=True, square=True)
    # plt.show()
    # import pdb; pdb.set_trace()

    dataset = np.hstack(
        [record_features, stat_features, odds_features, labels])
    record_feature_fields = \
        ['ht_w_pct', 'rt_w_pct', 'ht_pts_diff', 'rt_pts_diff',
         'ht_last_k_w_pct', 'rt_last_k_w_pct', 'ht_home_c_pct',
         'rt_road_c_pct']
    stat_feature_fields = \
        ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
         'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
         'MP', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
         'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg']
    # stat_feature_fields = ['stat_features'] * stat_features.shape[1]
    odds_features = ['ht_odds', 'rt_odds']
    filed_names = record_feature_fields + stat_feature_fields + odds_features
    return np.asarray(dataset), filed_names


def main():
    odds_stats = get_odds_stats()
    record_stats = get_record_stats()
    recent_stats, cumulated_stats = get_game_stats()
    dataset, filed_names = construct_dataset(
        record_stats, recent_stats, cumulated_stats, odds_stats)
    print('----------')
    print('construct dataset finished.')
    print('dataset shape: (%d, %d)' % (dataset.shape[0], dataset.shape[1]))
    # dump tp file
    data_path = './dataset/dataset.p'
    dump_to_file([dataset, filed_names], data_path)


if __name__ == '__main__':
    main()
