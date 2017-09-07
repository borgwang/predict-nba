# author: wangguibo <borgwang@126.com>
# date: 2017-8-25
# Copyright (c) 2017 by wangguibo
#
# file: parse_tables.py
# desc:
#     Parse tables fetched from 'fetch_game_pages.py'
#     Data save to './data/player_pages.p'
#     Data format: Dict
#     {name: dict {'basic_avg_stats': list,
#                  'basic_total_stats': list,
#                  'per_minute_stats': list,
#                  'advanced_stats': list}}
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from bs4 import BeautifulSoup
import runtime_path
from utils.utils import *


def main():
    # load saved tables
    player_tables_path = DATASET_BASE_PATH + 'player_tables.p'
    player_tables = load_from_file(player_tables_path, name='player_tables')

    # load pages
    player_pages_path = DATASET_BASE_PATH + 'player_pages.p'
    player_pages = load_from_file(player_pages_path, name='player_pages')

    num_added, num_skipped = 0, 0
    for name, html in sorted(player_pages.items()):
        if name in player_tables:
            num_skipped += 1
            continue
        main_soup = BeautifulSoup(html, 'lxml')
        comment_soup = get_comment_soup(main_soup)

        # player basic stats per game
        table = main_soup.find(id='per_game')
        basic_avg_stats = read_table(table)
        # player basic stats in total
        table = comment_soup.find(id='totals')
        basic_total_stats = read_table(table)
        # per_minute
        table = comment_soup.find(id='per_minute')
        per_minute_stats = read_table(table)
        # advanced
        table = comment_soup.find(id='advanced')
        advanced_stats = read_table(table)

        player_tables[name] = {
            'basic_avg_stats': basic_avg_stats,
            'basic_total_stats': basic_total_stats,
            'per_minute_stats': per_minute_stats,
            'advanced_stats': advanced_stats}

        num_added += 1
        print('%s readed. ' % name)
        if num_added > 1 and num_added % 100 == 0:
            # dump to file player_pages.p
            print('Add %d. Skipped %d.' % (num_added, num_skipped))
            dump_to_file(player_tables, player_tables_path)

    print('Loop ended.')
    dump_to_file(player_tables, player_tables_path)


if __name__ == '__main__':
    main()
