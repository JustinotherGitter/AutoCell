#!usr/bin/env/python
"""
AutoCell creates a cell object, for 1 or 2 dimensions, that follows cellular automata rules.

@notes =    [
            \u2588 == █
            ]
"""

__author__ = "Justin Cooper"
__version__ = "06.04.2020"
__email__ = "justin.jb78@gmail.com"

import datetime
from typing import Union


class Timer(object):
    def __init__(self,
                 total: Union[int, float] = None
                 ) -> None:
        """
        Initializes the Timer object

        :param  total       : The amount you plan to iterate over
        :type   total       : int
        """
        self.start = datetime.datetime.now()
        self.total = total

    def remains(self,
                done: Union[int, float]
                ):
        """
        Calculates the remaining time for the given amount of iterations

        :param  done        : The amount of iterations have been completed
        :return:time left   : The time predicted to still remain
        :rtype  str
        """
        now = datetime.datetime.now()
        left = (self.total - done) * (now - self.start) / done
        sec = int(left.total_seconds())
        if sec < 60:
            return "{} seconds".format(sec)
        else:
            return "{} minutes".format(int(sec / 60))

    def elapsed(self):
        """
        Calculates the time passed for the amount of completed iterations

        :return:time passed : The time calculated to have passed
        :rtype  timedelta object
        """
        return datetime.datetime.now() - self.start


def print_progress_bar(
                       iteration: int,
                       total: int,
                       prefix: str = '',
                       suffix: str = '',
                       decimals: int = 1,
                       length: int = 100,
                       fill: str = '█',
                       print_end: str = "\r"
                       ):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
