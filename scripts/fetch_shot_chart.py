# author: wangguibo <borgwang@126.com>
# date: 2017-8-27
# Copyright (c) 2017 by wangguibo
#
# file: fetch_shot_chart.py
# desc:
#     Fetch shooting charts all of actived players.
#     Data save to './data/active_player_shooting.p'
#     Data format: Dict
#     {name: list of [game_id, opppnent, shot_type, pos_top,
#                     pos_left, distance, result, quarter]]}
#     e.g. {'Stephen Curry':
#           [['20091028GSW', 'HOU', '3', '299', '339', '27', 'Missed', '1']]
#          }
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import urllib
import numpy as np
from bs4 import BeautifulSoup
import runtime_path
from utils.utils import *


def is_shooting_urls(tag):
    return tag.has_attr('href') and '/shooting/' in tag.get('href') \
        and '/players/' in tag.get('href')


def get_active_player():
    # get active players list
    basic_info_path = './data/player_basic_info.p'
    basic_info = load_from_file(basic_info_path, name='basic_info')
    basic_info = np.asarray(basic_info)
    filter_ = np.logical_and(
        basic_info[:, 0] == 'active', basic_info[:, 3] == '2017')
    idx = np.where(filter_)[0]
    active_players = basic_info[idx][:, 1]
    return active_players


def main():
    active_players = get_active_player()

    # load links
    links_path = DATASET_BASE_PATH + 'player_links.p'
    player_links = load_from_file(links_path, name='player_links')

    # load dataset
    shooting_data_path = DATASET_BASE_PATH + 'active_player_shooting.p'
    shooting_data = load_from_file(shooting_data_path, name='shooting_data')

    num_added, num_skipped = 0, 0
    for name in active_players:
        if name in shooting_data:
            num_skipped += 1
            continue

        data = []
        # fetch all shooting urls of a single player
        home_url = BASE_URL + player_links[name]
        html = urllib.urlopen(home_url).read()
        main_soup = BeautifulSoup(html, 'lxml')
        shooting_urls = {}
        for a in main_soup.find_all(is_shooting_urls):
            year = text_format(a.contents[0])
            if year not in shooting_urls:
                shooting_urls[year] = a.get('href')

        print(name, shooting_urls.keys())

        # fetch shooting stats
        for year, shooting_url in sorted(shooting_urls.items()):
            url = BASE_URL + shooting_url
            html = urllib.urlopen(url).read()
            main_soup = BeautifulSoup(html, 'lxml')
            comment_soup = get_comment_soup(main_soup)

            all_shots = comment_soup.find_all(class_='tooltip')
            print('%d shots in %s' % (len(all_shots), year))

            for s in all_shots:
                style = re.split(r'[:,p]', s.get('style'))
                pos_top, pos_left = style[2], style[4]
                info = s.get('tip').split('<br>')
                game_info = re.split(r'[\,,\ ]', info[0])
                date = date_format(game_info[0], game_info[1], game_info[3])
                if game_info[-2] == 'vs':
                    game_id = date + game_info[-3]
                else:
                    game_id = date + game_info[-1]
                opppnent = game_info[-1]
                quarter = info[1].split(' ')[0][0]
                shot_info = info[2].split(' ')
                result = shot_info[0]
                shot_type = shot_info[1][0]
                distance = shot_info[-2]
                data.append([game_id, opppnent, shot_type, pos_top,
                             pos_left, distance, result, quarter])

        shooting_data[name] = data
        num_added += 1

        if num_added > 1 and num_added % 20 == 0:
            dump_to_file(shooting_data, shooting_data_path)

    print('Loop ended.')
    dump_to_file(shooting_data, shooting_data_path)


if __name__ == '__main__':
    main()
