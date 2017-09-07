# author: wangguibo <borgwang@126.com>
# date: 2017-8-25
# Copyright (c) 2017 by wangguibo
#
# file: fetch_palyer_links.py
# desc:
#     Fetch links of player home pages.
#     Data save to './data/player_links.p'
#     Data format:
#     Dict {name: page_link}  e.g. {'Ray Allen': '/players/a/allenra02.html'}
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from bs4 import BeautifulSoup
import urllib
import runtime_path
from utils.utils import *


def main():
    base_url = BASE_URL + '/players/'
    player_links = {}

    for letter in LETTERS:
        url = base_url + letter + '/'
        html = urllib.urlopen(url).read()
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find(id='players')
        count = 0
        for th in table.tbody.find_all(name='th'):
            name = str(th.text)
            if '*' in name:
                name = name[:-1]
            player_links[name] = th.a.get('href')
            count += 1
        print('%d player links added.' % count)

    print('Loop ended.')
    player_links_path = DATASET_BASE_PATH + 'player_links.p'
    dump_to_file(player_links, player_links_path)


if __name__ == '__main__':
    main()
