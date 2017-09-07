# author: wangguibo <borgwang@126.com>
# date: 2017-8-28
# Copyright (c) 2017 by wangguibo
#
# file: fetch_team_schedules.py
# desc:
#     Fetch game schedules and results of all 30 teams.
#     Data save to './data/team_schedules.p'
#     Data format: Dict
#     {team: dict {year: game_info}}
#     game_info format: List [date, opp, loc, res, pts, opp_pts, rec]
#     e.g. {'GSW':
#           {'2017':
#             [['20170412', 'BOS', 'visit', 'L', '94', '112', '42-40']]}}
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

# team name changes
# SEA -> OKC in 2008-2009
# NJN -> BRK in 2012-2013
# NOH -> NOP in 2013-2014
# CHA -> CHO in 2014-2015


def all_with_attr(tag_name, attr, value):
    def filter_(tag):
        return not tag.has_attr('aria-label') and tag.has_attr(attr) \
            and tag[attr] == value
    return tag_name.find_all(filter_)


def main():
    # load database
    team_schedules_path = DATASET_BASE_PATH + 'team_schedules.p'
    team_schedules = load_from_file(
        path=team_schedules_path, name='team_schedules')

    years = [str(y) for y in range(START_YEAR, END_YEAR)]
    base_url = BASE_URL + '/teams/'
    for i, team in enumerate(TEAM_NAMES):
        if team in team_schedules:
            seasons = team_schedules[team]
        else:
            seasons = {}
        for year in years:
            if year in seasons:
                print('%d %s %d-%d exist' % (i, team, int(year)-1, int(year)))
                continue
            url = base_url + team + '/' + year + '_games.html'
            html = urllib.urlopen(url).read()
            soup = BeautifulSoup(html, 'lxml')
            table = soup.find(id='games')
            if not table:
                print('%s %s error.' % (year, team))
                continue
            # game date
            td_date = all_with_attr(table, 'data-stat', 'date_game')
            date = np.asarray([d['csk'].replace('-', '') for d in td_date])
            # opponent team
            td_opp = all_with_attr(table, 'data-stat', 'opp_name')
            opp = np.asarray([o['csk'][:3] for o in td_opp])
            # game_location [home | visit]
            td_loc = all_with_attr(table, 'data-stat', 'game_location')
            loc = np.asarray(['visit' if l.text else 'home' for l in td_loc])
            # result [W | L]
            td_res = all_with_attr(table, 'data-stat', 'game_result')
            res = np.asarray([text_format(r.text) for r in td_res])
            # points and opponent points
            td_pts = all_with_attr(table, 'data-stat', 'pts')
            pts = np.asarray([int(p.text) for p in td_pts])
            td_opp_pts = all_with_attr(table, 'data-stat', 'opp_pts')
            opp_pts = np.asarray([int(p.text) for p in td_opp_pts])
            # record [w-l]
            td_rec_w = all_with_attr(table, 'data-stat', 'wins')
            td_rec_l = all_with_attr(table, 'data-stat', 'losses')
            rec = np.asarray([text_format(w.text + '-' + l.text)
                             for w, l in zip(td_rec_w, td_rec_l)])
            season = np.vstack((date, opp, loc, res, pts, opp_pts, rec)).T
            print('%d %s %d-%d added' % (i, team, int(year)-1, int(year)))
            seasons[year] = season

        team_schedules[team] = seasons
        dump_to_file(team_schedules, team_schedules_path)

    print('Loop ended.')
    dump_to_file(team_schedules, team_schedules_path)


if __name__ == '__main__':
    main()
