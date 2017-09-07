# author: wangguibo <borgwang@126.com>
# date: 2017-8-25
# Copyright (c) 2017 by wangguibo
#
# file: fetch_game_pages.py
# desc:
#     Fetch game html pagesusing links from './data/page_links.p'.
#     Data save to './data/game_pages.p'
#     Data format: Dict
#     {game_id: html page}   e.g. {'20161111WAS': html page}
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import urllib
from bs4 import BeautifulSoup
import runtime_path
from utils.utils import *


def main():
    # load pages database
    team_performance_path = \
        DATASET_BASE_PATH + 'team_performance/team_performance.p'
    team_performance = load_from_file(
        team_performance_path, 'team_performance')

    # load game links
    game_info_path = DATASET_BASE_PATH + 'game_info.p'
    game_info = load_from_file(path=game_info_path, name='game_info')
    num_skipped = 0
    global_count = 0
    for year, season in game_info.items():
        print('fetching year %s' % year)
        year_count = 0
        for month, games in season.items():
            month_count = 0
            for game in games:
                game_id = game[0] + '0' + game[2]
                if game_id in team_performance:
                    print('%s exist. skipping...' % game_id)
                    num_skipped += 1
                    continue
                url = BASE_URL + game[-1]
                html = urllib.urlopen(url).read()
                soup = BeautifulSoup(html, 'lxml')
                team_stats = soup.find_all('tfoot')
                h_stats, r_stats = [], []
                for stat in team_stats[:2]:
                    r_stats += [text_format(td.text)
                                for td in stat.find_all('td')
                                if td.text != '' and td != '\n']
                for stat in team_stats[2:]:
                    h_stats += [text_format(td.text)
                                for td in stat.find_all('td')
                                if td.text != '' and td != '\n']
                if len(h_stats) != 34 or len(r_stats) != 34:
                    print('%s data len exception!!! skipping...' % game_id)
                    continue
                # {game_id: [date, game_type, home, visit, h_stats, r_stats]}
                team_performance[game_id] = \
                    [game[0], game[1], game[2], game[3], r_stats, h_stats]
                month_count += 1
                print('game %s added' % game_id)

            year_count += month_count
            print('%s %d added' % (month, month_count))
            dump_to_file(team_performance, team_performance_path)

        global_count += year_count
        print('--- %s %d added' % (year, year_count))

    print('Loop ended. %d added. %d skipped' % (global_count, num_skipped))
    dump_to_file(team_performance, team_performance_path)


if __name__ == '__main__':
    main()
