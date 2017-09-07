# author: wangguibo <borgwang@126.com>
# date: 2017-8-23
# Copyright (c) 2017 by wangguibo
#
# file: fetch_game_odds.py
# desc:
#     Fetch game odds from 'http://www.oddsportal.com'.
#     Data save to './data/game_odds.p'
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import pandas as pd
import numpy as np
import runtime_path
from utils.utils import *


def date_format(date, season):
    front_year, laatter_year = season.split('-')
    if len(date) < 4:
        formated_date = laatter_year + '0' + date
    else:
        formated_date = front_year + date

    return formated_date


# load excel file
odds_data_path = DATASET_BASE_PATH + 'odds'
odds_data = {}
for filename in os.listdir(odds_data_path):
    if filename.split('.')[-1] != 'xlsx':
        continue
    season = filename.split('.')[0]
    file_path = os.path.join(odds_data_path, filename)
    data = np.asarray(pd.ExcelFile(file_path).parse('Sheet1'), dtype=str)
    # handle name changes
    for name, map_name in TEAM_NAME_MAPPING.items():
        data[data == name] = map_name

    # date formate
    formated_dates = map(date_format, data[:, 0], [season] * len(data))
    data[:, 0] = np.asarray(formated_dates)

    odds_data[season] = data[:, [0, 3, 11]]

game_odds = []
for season, data in odds_data.items():
    for i in range(0, len(data), 2):
        date = data[i][0]
        ht, rt = data[i+1][1], data[i][1]
        ht_odds, rt_odds = data[i+1][2], data[i][2]
        game_odds.append([date, ht, rt, ht_odds, rt_odds])

game_odds = np.asarray(game_odds)

# dump to file
game_odds_path = DATASET_BASE_PATH + 'game_odds.p'
dump_to_file(game_odds, game_odds_path)
