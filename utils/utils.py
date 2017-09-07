# author: wangguibo <borgwang@126.com>
# date: 2017-8-23
# Copyright (c) 2017 by wangguibo
#
# file: utils.py
# desc: useful utils and global variables
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from bs4 import BeautifulSoup, Comment

# global variables
MONTH_MAPPING = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

TEAM_NAMES = ['TOR', 'BOS', 'NYK', 'BRK', 'NJN', 'PHI',
              'CLE', 'IND', 'DET', 'CHI', 'MIL',
              'MIA', 'ATL', 'CHO', 'CHA', 'WAS', 'ORL',
              'OKC', 'SEA', 'POR', 'UTA', 'DEN', 'MIN',
              'GSW', 'LAC', 'SAC', 'PHO', 'LAL',
              'SAS', 'DAL', 'MEM', 'HOU', 'NOP', 'NOH']

TEAM_NAME_MAPPING = {'Dallas': 'DAL', 'NewYork': 'NYK', 'Minnesota': 'MIN',
                     'Denver': 'DEN', 'Indiana': 'IND', 'Golden State': 'GSW',
                     'Phoenix': 'PHO', 'OklahomaCity': 'OKC',
                     'GoldenState': 'GSW', 'Sacramento': 'SAC',
                     'Chicago': 'CHI', 'Utah': 'UTA', 'SanAntonio': 'SAS',
                     'Brooklyn': 'BRK', 'Atlanta': 'ATL', 'Memphis': 'MEM',
                     'Toronto': 'TOR', 'NewOrleans': 'NOP', 'Houston': 'HOU',
                     'Miami': 'MIA', 'Cleveland': 'CLE', 'Orlando': 'ORL',
                     'Detroit': 'DET', 'Philadelphia': 'PHI',
                     'Charlotte': 'CHO', 'NewJersey': 'BRK', 'Boston': 'BOS',
                     'Washington': 'WAS', 'LAClippers': 'LAC',
                     'LALakers': 'LAL', 'Milwaukee': 'MIL', 'Portland': 'POR',
                     'Seattle': 'OKC'}

NAME_CHANGES = {'SEA': 'OKC', 'NJN': 'BRK', 'NOH': 'NOP', 'CHA': 'CHO'}

MONTHS = ['october', 'november', 'december', 'january', 'february', 'march',
          'april', 'may', 'june']

LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

DATASET_BASE_PATH = '../data/'

BASE_URL = 'https://www.basketball-reference.com'


def dump_to_file(contents, path):
    with open(path, 'wb') as f:
        pickle.dump(contents, f, -1)
    print('dump successfully. %d in total.\n' % len(contents))


def load_from_file(path, name='data', dtype='dict'):
    print('loading %s...' % name)
    try:
        data = pickle.load(open(path, 'rb'))
        print('suceessful loading %s. \n%d data in database...\n' %
              (name, len(data)))
    except Exception as e:
        print(e)
        print('load %s error. Starting from an empty dataset...\n' % name)
        data = {} if dtype == 'dict' else []

    return data


def read_table(table):
    if not table:
        return None
    stats_all_years = []
    for row in table.tbody.children:
        if row != '\n' and row.text != '':
            stats_ = []
            for col in row.children:
                if col != '\n':
                    if col.text != '':
                        stats_.append(text_format(col.text))
                    else:
                        stats_.append('NULL')
            stats_all_years.append(stats_)
    stats_career = []
    for row in table.tfoot.children:
        if row != '\n' and row.text != '':
            stats_ = []
            for col in row.children:
                if col != '\n':
                    if col.text != '':
                        stats_.append(text_format(col.text))
                    else:
                        stats_.append('NULL')
            stats_career.append(stats_)

    return stats_all_years, stats_career


def date_format(month, day, year):
    return year + MONTH_MAPPING[month] + day


def text_format(text):
    return str(text.encode('utf-8'))


def get_comment_soup(main_soup):
    comment_str = ''
    all_comments = main_soup.find_all(
        string=lambda text: isinstance(text, Comment))
    for c in all_comments:
        comment_str += text_format(c.extract())

    return BeautifulSoup(comment_str, 'lxml')


def get_season_by_date(date):
    assert len(date) == 8, 'invalid date format!'
    date = str(date)
    year, month = date[:4], date[4:6]
    if month < '07':
        return str(int(year) - 1) + '-' + year
    else:
        return year + '-' + str(int(year) + 1)
