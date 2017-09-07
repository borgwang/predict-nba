# author: wangguibo <borgwang@126.com>
# date: 2017-8-23
# Copyright (c) 2017 by wangguibo
#
# file: fetch_palyers_info.py
# desc:
#     Fetch basic info of all players. Data save to './data/player_info.p'
#     Data format: List
#     ['Status', 'Player', 'First year', 'Last year', 'Position',
#     'Height', 'Weight', 'Birth Date', 'College']
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
    base_url = BASE_URL + '/players/'
    player_info = []
    num_players = 0
    for letter in LETTERS:
        url = base_url + letter + '/'
        page = urllib.urlopen(url)
        html = page.read()
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find(id='players')

        _map = {'year_min': 2, 'year_max': 3, 'pos': 4, 'height': 5,
                'weight': 6, 'birth_date': 7, 'college_name': 8}
        count = 0
        for item in table.tbody.children:
            if item != '\n' and item.get_text() != '':
                player = [None] * 9
                for col in item.children:
                    if col.name == 'th':
                        name = str(col.text)
                        if col.strong is not None:
                            status = 'active'
                        elif '*' in name:
                            status = 'hall of fame'
                            name = name[:-1]
                        else:
                            status = 'retired'
                        player[0] = status
                        player[1] = name
                    if col.name == 'td':
                        player[_map.get(col.get('data-stat'))] = str(col.text)

                player_info.append(player)
                count += 1
        print('%s for %d players' % (letter.upper(), count))
        num_players += count
    print('%d players in total' % num_players)

    # dump to file
    print('Loop ended')
    player_info_path = './data/player_info.p'
    dump_to_file(player_info, player_info_path)


if __name__ == '__main__':
    main()
