# author: wangguibo <borgwang@126.com>
# date: 2017-8-26
# Copyright (c) 2017 by wangguibo
#
# file: fetch_player_pages.py
# desc:
#     Fetch game html pagesusing links from './data/player_links.p'.
#     Data save to './data/player_pages.p'
#     Data format: Dict
#     {name: player_page}
#
# ----------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import urllib
import runtime_path
from utils.utils import *


def main():
    # load pages database
    pages_path = DATASET_BASE_PATH + 'player_pages.p'
    player_pages = load_from_file(pages_path, name='player_pages')

    links_path = DATASET_BASE_PATH + 'player_links.p'
    player_links = load_from_file(links_path, name='player_links')

    num_added, num_skipped = 0, 0

    for name, link in sorted(player_links.items()):
        if name in player_pages:
            num_skipped += 1
            continue

        url = BASE_URL + link
        html = urllib.urlopen(url).read()
        player_pages[name] = html
        num_added += 1
        print('%s loaded' % name)

        if num_added > 1 and num_added % 20 == 0:
            print('Add %d. Skipped %d.' % (num_added, num_skipped))
            dump_to_file(player_pages, pages_path)

    print('Loop ended')
    dump_to_file(player_pages, pages_path)


if __name__ == '__main__':
    main()
