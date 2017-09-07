# author: wangguibo <borgwang@126.com>
# date: 2017-8-23
# Copyright (c) 2017 by wangguibo
#
# file: fetch_game_info.py
# desc:
#     Fetch basic game info. Data save to './data/game_info.p'
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import urllib
import numpy as np
from bs4 import BeautifulSoup
import runtime_path
from utils.utils import *

# CONFIG
START_YEAR = 2008
END_YEAR = 2018


def all_with_attr(tag_name, attr, value):
    def filter_(tag):
        return not tag.has_attr('aria-label') and tag.has_attr(attr) \
            and tag[attr] == value
    return tag_name.find_all(filter_)


def main():
    # load database
    game_info_path = DATASET_BASE_PATH + 'game_info.p'
    game_info = load_from_file(path=game_info_path, name='game_info')
    base_url = BASE_URL + '/leagues/'
    years = [str(y) for y in range(START_YEAR, END_YEAR)]
    total_count = 0

    for year in years:
        if year in game_info:
            print('%s already exist.' % year)
            continue
        season = {}
        playoffs_date = None
        for month in MONTHS:
            # www.basketball-reference.com/leagues/NBA_2017_games-may.html
            url = base_url + 'NBA_' + year + '_games-' + month + '.html'
            html = urllib.urlopen(url).read()
            soup = BeautifulSoup(html, 'lxml')
            table = soup.find(id='schedule')
            if table is None:
                print('skip %s %s. No games. ' % (year, month))
                continue

            # [date, game_type, home, visit, home_score, visit_score, links]
            l_date = all_with_attr(table, 'data-stat', 'date_game')
            date = np.asarray([text_format(d['csk'][:-4]) for d in l_date])
            l_home = all_with_attr(table, 'data-stat', 'home_team_name')
            home = np.asarray([text_format(v['csk'][:3]) for v in l_home])
            l_visit = all_with_attr(table, 'data-stat', 'visitor_team_name')
            visit = np.asarray([text_format(v['csk'][:3]) for v in l_visit])

            l_home_pts = all_with_attr(table, 'data-stat', 'home_pts')
            home_pts = np.asarray([text_format(hp.text) for hp in l_home_pts])
            l_visit_pts = all_with_attr(table, 'data-stat', 'visitor_pts')
            visit_pts = np.asarray([text_format(vp.text) for vp in l_visit_pts])
            l_links = all_with_attr(table, 'data-stat', 'box_score_text')
            links = np.asarray([l.a.get('href') for l in l_links])

            # game type (regular, playoffs, finals)
            playoffs_sign = table.tbody.find(class_='thead')
            if playoffs_sign:
                playoffs_sign = playoffs_sign.previous_sibling.previous_sibling
                playoffs_date = playoffs_sign.th['csk'][:-4]
            game_type = np.array(['regular'] * len(date))
            if month == 'june':
                game_type = np.array(['finals'] * len(date))
            elif playoffs_date:
                game_type[np.where(date > playoffs_date)[0]] = 'playoffs'
            season[month] = np.vstack(
                (date, game_type, home, visit, home_pts, visit_pts, links)).T
            print('%s %s: %d games added' %
                  (year, month, len(season[month])))
            total_count += len(season[month])

        game_info[year] = season
        print('games of year %s added.' % year)
        dump_to_file(game_info, game_info_path)

    print('Loop ended.')
    dump_to_file(game_info, game_info_path)


if __name__ == '__main__':
    main()
