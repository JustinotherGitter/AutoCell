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
from typing import Tuple, List, Optional, Union, Callable
from random import randint as randint

from Basics import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# TODO: Cell1D show revamp
# TODO: Checkbox for black/white in show 2D
# TODO: Cell2D.animate() not called properly when run in Terminal
# TODO: Cell2D resize not working
# TODO: Have way to show age, I.E. Pixel gets darker the older it is


class Cell(object):
    """
    Parent class for Cells to inherit from
    """

    def __init__(self,
                 shape: Union[Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]] = None,
                 is_stable: bool = None,
                 head: Union[np.ndarray, List, Tuple] = None
                 ) -> None:
        """
        Parent Class Cell initialization

        :param  shape       : The 'x'-, 'y'-, (& 'z'-) sizes of the Cells history
        :type   shape       : Union[Tuple[int, int, Optional[int]], List[int, int, Optional[int]]]
        :param  is_stable   : Determines if you want to limit the size of the Cell history
        :type   is_stable   : bool
        :param  head        : An array to be used as the starting condition
        :type   head        : Binary integer numpy.ndarray, list, or tuple
        """
        if shape is None:
            self._shape = (64, 64, 64)
        elif type(shape) in [tuple, list]:
            if len(shape) is 2:
                self._shape = (shape, 0)
            elif len(shape) is 3:
                self._shape = tuple(shape)
            else:
                raise ValueError("Shape should be of length 2 or 3")
        else:
            raise TypeError("Shape should be None or of type List or Tuple")

        self._size_x, self._size_y, self._depth = shape

        self._is_stable = bool(is_stable)

        if head is None:
            self._head = None
        elif type(head) in [list, tuple, np.ndarray]:
            self._head = np.array(head)
        else:
            raise TypeError("Head should be None or of type List, Tuple, or np.ndarray")

        self._history = None
        self._time = None
        self._processor = None

    def resize(self,
               shape: Union[int, Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]] = None,
               head: Union[np.ndarray, List, Tuple] = None
               ) -> None:
        """
        Sets the shape of the cell and re-processes

        :param  shape       : The new 'x'-, 'y'-, (& 'z'-) sizes of the Cells history
        :type   shape       : Union[int, Tuple[int, int, Optional[int]], List[int, int, Optional[int]]]
        :param  head        : A new array to be used as the starting condition
        :type   head        : Binary integer numpy.ndarray, list, or tuple
        """
        shape_change, head_change = False, False

        if shape is None:
            pass
        elif type(shape) is int:
            shape_change = True
            self._size_x = shape
            self._shape = (shape, self._size_y, self._depth)
        elif type(shape) in [tuple, list]:
            if len(shape) is 2:
                shape_change = True
                self._size_x, self._size_y = shape
                self._shape = (shape, self._depth)
            elif len(shape) is 3:
                shape_change = True
                self._size_x, self._size_y, self._depth = shape
                self._shape = shape
            else:
                raise ValueError("Shape received too many arguments to unpack")
        else:
            raise TypeError("Shape should be None or of type int, Tuple, or List")

        if head is None:
            pass
        elif type(head) in [list, tuple, np.ndarray]:
            head = np.array(head)
            if len(head.shape) is 1 and head.shape >= (self._size_x,):
                head_change = True
                self._head = head[:self._size_x]
            elif len(head.shape) is 2 and head.shape >= (self._size_y, self._size_x):
                head_change = True
                self._head = head[:self._size_y, :self._size_x]
            else:
                raise ValueError("Head passed too many dimensions to unpack")
        else:
            raise TypeError("Head should be None or of type List, Tuple, or np.ndarray")

        if head_change or shape_change:
            self.call_processor()

        print("Resize complete")

    def call_processor(self,
                       func: Callable = None
                       ) -> None:
        """
        Calls the processor of the Class to process the Cell

        :param  func        : The function considered to be the processor of the Class
        :type   func        : Callable
        """
        if func is None:
            func = self.processor
        self._time = func()

    @property
    def processor(self):
        print("Getting Cell processor")
        return self._processor

    @processor.setter
    def processor(self, func: Callable):
        print("Setting Cell processor")
        self._processor = func

    @property
    def shape(self):
        print("Getting Cell shape")
        return self._shape

    @shape.setter
    def shape(self, value: Union[int, Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]]):
        print("Setting Cell shape")
        self.resize(shape=value, head=None)

    @property
    def head(self):
        print("Getting Cell head")
        return self._head

    @head.setter
    def head(self, value: Union[np.ndarray, List, Tuple]):
        print("Setting Cell head")
        self.resize(shape=None, head=value)

    @property
    def is_stable(self):
        print("Getting Cell stable property")
        return self._is_stable

    @is_stable.setter
    def is_stable(self, value: bool):
        print("Setting Cell stable property")
        self._is_stable = bool(value)

    @property
    def history(self):
        print("Getting Cell history")
        return self._history

    @property
    def time(self):
        print("Getting Cell process time")
        return self._time

    @property
    def depth(self):
        print("Getting Cell depth")
        return self._depth

    @depth.setter
    def depth(self, value: int):
        print("Setting Cell depth")
        self.resize(shape=(int(self._shape[0]), int(self._shape[1]), int(value)), head=None)


class Cell1D(Cell):

    def __init__(self,
                 shape: Union[Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]] = None,
                 rule_no: int = None,
                 is_stable: bool = None,
                 head: Union[np.ndarray, List, Tuple] = None
                 ) -> None:
        """
        Initializes the 1D Cell object

        :param  rule_no     : The decimal equivalent of the binary rule set
        :type   rule_no     : int [0, 256)
        """
        super().__init__(shape, is_stable, head)

        if rule_no is None:
            self._rule_no = randint(0, 255)
        elif type(rule_no) is int or float:
            self._rule_no = int(abs(rule_no) % 256)
        else:
            raise TypeError("Rule number should be of type None, int, or float")

        self._rule = self.rule_calc(self._rule_no)

        if self._head is None:
            self._head = np.random.randint(2, size=self._size_x, dtype=int)
        elif len(self._head.shape) is 1 and self._head.size >= self._size_x:
            self._head = self._head[:self._size_x]
        else:
            raise ValueError("Head should be None or a 1 dimensional array of type np.ndarray")

        self._processor = self.process
        self.call_processor()

    def __str__(self):
        return f"1 Dimensional Cell with shape {self._shape} & rule {self._rule_no}, processed in {self._time}"

    def show(self,
             start: int = 0,
             end: int = None
             ):
        """
        Plots the selected rows of the Cell

        :param  start       : From which row in history to start
        :type   start       : int       (Defaults to starting row)
        :param  end         : At which row in history to end
        :type   end         : int       (Defaults to last row)
        """
        end = end if end else self._history.shape[0]

        plt.imshow(self._history[int(start):int(end)], aspect='equal')
        plt.xticks([])  # remove the tick marks by setting to an empty list
        plt.yticks([])  # remove the tick marks by setting to an empty list
        plt.show()

    def inline_print(self,
                     start: int = 0,
                     end: int = None
                     ):
        """
        Prints the selected rows of the Cell

        :param  start       : From which row in history to start
        :type   start       : int       (Defaults to starting row)
        :param  end         : At which row in history to end
        :type   end         : int       (Defaults to last row)
        """
        print("Printing cell history\n=====================\nValue 1 = █\nValue 0 = _")

        end = self._history.shape[0] if not end else end

        for i in range(int(start), int(end)):
            st = ''
            for j in self._history[i]:
                if bool(j):
                    st += '█'
                else:
                    st += '_'
            print(st)
        print()

    def process(self):
        """
        Processes the cell parameters to calculate the cell history

        :return:time        : Time that the process took to complete calculations
        :rtype: timedelta object
        """
        print("Processing")
        hist = np.zeros((self._size_y, self._size_x), dtype=int)
        hist[0] = self._head

        t_object = Timer(self._size_y)
        print_progress_bar(0, self._size_y, prefix='Progress:', suffix='Complete\tETA: N\\A', length=50)

        for i in range(1, self._size_y):
            for j in range(self._size_x):
                hist[i, j] = self._rule[str(int(hist[i - 1, (j - 1) % self._size_x]))
                                        + str(int(hist[i - 1, j % self._size_x]))
                                        + str(int(hist[i - 1, (j + 1) % self._size_x]))]

            # For Stable Solution
            if np.array_equal(hist[i], hist[i - 1]):
                print("Stable solution found after {} iterations".format(i))

                if self._is_stable:
                    hist.resize((i + 1, self._size_x))
                    break
                else:
                    hist[i+1:] = hist[i]
                    break

            # For Alternating Solution
            if i >= 2:
                if np.array_equal(hist[i], hist[i - 2]):
                    print("Oscillating solution found after {} iterations".format(i))

                    if self._is_stable:
                        hist.resize((i + 1, self._size_x))
                        break
                    else:
                        hist[i+1::2] = hist[i - 1]
                        hist[i+2::2] = hist[i]
                        break

            if i is self._size_y - 1:
                print_progress_bar(i + 1, self._size_y,
                                   prefix='Progress:', suffix='Complete\tETA: {}'.format("Done"),
                                   length=50)

            else:
                print_progress_bar(i + 1, self._size_y,
                                   prefix='Progress:', suffix='Complete\tETA: {}'.format(t_object.remains(i)),
                                   length=50)

        self._history = hist
        print("Finished in {}".format(t_object.elapsed()))
        return t_object.elapsed()

    @property
    def rule_no(self):
        print("Getting the rule number")
        return self._rule_no

    @rule_no.setter
    def rule_no(self,
                value: int
                ):
        print("Setting the rule number")
        self._rule_no = int(value)
        self._rule = self.rule_calc(self._rule_no)
        self.call_processor()

    @property
    def rule(self):
        """
        Gets the rule dictionary

        :return:rule        : A print out of the rule list as well as the rule dictionary
        :rtype: dict
        """
        print("Getting rule")
        print(" " * 20, "Rule %d" % self._rule_no)
        print((" # {} |" * 8).format(*np.arange(1, 9)))
        print((" {} |" * 8).format(*[i.replace('0', '_').replace('1', '█') for i in self._rule]))
        print(("  {}  |" * 8).format(*[self._rule[i].replace('0', '_').replace('1', '█') for i in self._rule]))
        print()
        return self._rule

    @rule.setter
    def rule(self,
             value: int
             ):
        print("Setting rule")
        self._rule = self.rule_calc(int(abs(value) % 256))

    @staticmethod
    def rule_calc(
                  rule_no: int
                  ):
        """
        Calculates the rule dictionary based on the given rule number

        :param  rule_no     : A positive integer in the range (0, 256)
        :type   rule_no     : int

        :return:dictionary  : A dictionary where each key: value pair is a unique cell state rule
        :rtype: dict
        """
        # Check to see if valid rule & right size_y
        bin_rule = bin(abs(rule_no) % 256)[:1:-1]
        if len(bin_rule) < 8:
            bin_rule = bin_rule + '0' * (8 - len(bin_rule))

        perms = ["000", "001", "010", "011", "100", "101", "110", "111"]
        return {perms[i]: [bin_rule[i] for i in range(8)][i] for i in range(8)}


class Cell2D(Cell):

    def __init__(self,
                 shape: Union[Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]] = None,
                 is_stable: bool = None,
                 head: Union[np.ndarray, List, Tuple] = None,
                 birth: Union[List, Tuple] = None,
                 death: Union[List, Tuple] = None
                 ) -> None:
        """
        Initialises the Cell object

        :param  birth       : Neighbours value(s) for which a dead cell will be born   (0 -> 1)
        :type   birth       : Union[int, List, Tuple]
        :param  death       : Neighbours value(s) at which a live cell will die        (1 -> 0)
        :type   death       : Union[int, List, Tuple]
        """
        super().__init__(shape, is_stable, head)

        # Default is Conway's Game of Life Rules
        if birth is None:
            self._birth = (3,)
        else:
            self._birth = tuple(birth)

        if death is None:
            self._death = (0, 1, 4, 5, 6, 7, 8)
        else:
            self._death = tuple(death)

        if self._head is None:
            self._head = np.random.randint(2, size=(self._size_y, self._size_x), dtype=int)
        elif len(self._head.shape) is 2 and self._head.shape >= (self._size_y, self._size_x):
            self._head = self._head[:self._size_y, :self._size_x]
        else:
            raise ValueError("Head should be None or a 2 dimensional array of type np.ndarray")

        self._processor = self.process
        self.call_processor()

    def __str__(self):
        return f"2 Dimensional Cell with shape {self._shape}, processed in {self._time}\n" \
               f"Cell dies  when number of neighbours is in {self._death}\n" \
               f"Cell lives when number of neighbours is in {self._birth}"

    def show(self,
             start: int = 0
             ):
        """
        Plots the selected layer of the Cell

        :param  start       : From which depth in history to start (Defaults to first row)
        :type   start       : int
        """
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0, bottom=0.2, right=1, top=0.95)
        ax.set_title("Shape:   {}, Frame   {}".format(self._shape, start))
        cell_frame = plt.imshow(self._history[start], cmap='binary', interpolation='None')
        ax.margins(x=0)

        # Add axes using x_pos, y_pos, width, & height starting from bottom-left
        axis_frame = fig.add_axes((0.2, 0.05, 0.6, 0.05), facecolor='lightgoldenrodyellow')
        slider_frame = Slider(axis_frame, 'Frame', 0, self._depth - 1, valfmt='%.0f', valinit=start, valstep=1)

        def update(val):
            ax.set_title("Shape:   {}, Frame   {}".format(self._shape, int(val)))
            cell_frame.set_data(self._history[int(slider_frame.val)])
            fig.canvas.draw_idle()

        slider_frame.on_changed(update)

    def animate(self,
                value: int = 50
                ):
        """
        Animates the cell history

        :param  value       : The time delay between each frame     (in milliseconds)
        :type   value       : int
        """
        fig = plt.figure()
        ax = plt.axes(aspect="equal", animated=True, xticks=[], yticks=[])
        im = plt.imshow(self._history[0], interpolation='none', )

        def init():
            im.set_data(self._history[0])
            return im,

        def animate(i):
            im.set_data(self._history[i])
            return im,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=self._depth, interval=int(value), blit=True)

    def process(self):
        """
        Processes the cell parameters to calculate the cell history

        :return:time        : Time that the process took to complete calculations
        :rtype: timedelta object
        """

        print("Processing")

        hist = np.zeros(self._shape, dtype=int)
        hist[0] = self._head

        t_object = Timer(self._depth)
        print_progress_bar(0, self._depth, prefix='Progress:', suffix='Complete\tETA: N\\A', length=50)

        for i in range(1, self._depth):
            for j in range(self._size_y):
                for k in range(self._size_x):

                    neighbours = self.check_neighbours(hist[i - 1], j, k)

                    # Check if cell dies
                    if neighbours in self._death:
                        hist[i, j, k] = 0
                    # Check if cell is born
                    elif hist[i - 1, j, k] == 0 and neighbours in self._birth:
                        hist[i, j, k] = 1
                    # Cell remains unchanged
                    else:
                        hist[i, j, k] = hist[i - 1, j, k]

            # For Stable Solution
            if np.array_equal(hist[i], hist[i - 1]):
                print("Stable solution found after {} iterations".format(i))

                if self._is_stable:
                    hist.resize((i + 1, self._size_y, self._size_x))
                    break
                else:
                    hist[i + 1:] = hist[i]
                    break

            # For Alternating Solution
            if i >= 2:
                if np.array_equal(hist[i], hist[i - 2]):
                    print("Oscillating solution found after {} iterations".format(i))

                    if self._is_stable:
                        hist.resize((i + 1, self._size_y, self._size_x))
                        break
                    else:
                        hist[i + 1::2] = hist[i - 1]
                        hist[i + 2::2] = hist[i]
                        break

            if i is self._depth - 1:
                print_progress_bar(i + 1, self._depth,
                                   prefix='Progress:', suffix='Complete\tETA: {}'.format("Done"),
                                   length=50)

            else:
                print_progress_bar(i + 1, self._depth,
                                   prefix='Progress:', suffix='Complete\tETA: {}'.format(t_object.remains(i)),
                                   length=50)

        self._history = hist
        print("Finished in {}".format(t_object.elapsed()))
        return t_object.elapsed()

    def check_neighbours(self,
                         hist: np.ndarray,
                         y: int,
                         x: int
                         ):
        """
        Calculates the sum of the neighbouring cells

        :param  hist        : An array to calculate the neighbours from
        :type   hist        : np.ndarray
        :param  y           : The current 'y' index of the hist array
        :type   y           : int
        :param  x           : The current 'x' index of the hist array
        :type   x           : int

        :return:Sum         : The sum of all neighbours, wrapped around all array edges
        :rtype: int
        """
        return sum(hist[(np.array([-1, -1, -1, 0, 0, 1, 1, 1]) + y) % self._size_y,
                        (np.array([-1, 0, 1, -1, 1, -1, 0, 1]) + x) % self._size_x])


def setup(def_dim=1, def_shape=(512, 512, 512), def_rule=np.random.randint(0, 256, dtype=int), def_stable=False):
    # DIMENSIONS
    def dim():
        d = input("Should the Cell be 1 or 2 dimensions?    :  ")
        try:
            if d is "":
                print("Default {} dimension chosen".format(def_dim))
                return def_dim
            elif int(d) in [1, 2]:
                return int(d)
            else:
                print("Provide either a 1 or a 2")
                dim()
        except ValueError:
            print("Provide an integer")
            dim()

    # SHAPE
    def shape(value):
        def ax(st, ind):

            x = input(st)

            try:
                if x == "":
                    print("Default size {} chosen".format(def_shape[ind]))
                    return def_shape[ind]

                elif int(float(x)) > 0:
                    return int(float(x))

                else:
                    print("Provide an integer greater than 0")
                    ax(st, ind)

            except ValueError:
                print("Provide an integer")
                ax(st, ind)

        if value is 1:
            return (ax("Provide an 'x' size for the cell:  ", 0),
                    ax("Provide a 'length' for the cell:  ", 1),
                    0)
        else:
            return (ax("Provide an 'x' size for the cell:  ", 0),
                    ax("Provide a 'y' size for the cell:  ", 1),
                    ax("Provide a 'depth' for the cell:  ", 2))

    # RULE_NO
    def rule():
        z = input("Provide an integer to be used as the rule list: ")
        try:
            if z == "":
                print("Default rule {} chosen".format(def_rule))
                return def_rule

            elif 0 <= int(float(z)) <= 255:
                return int(float(z))

            else:
                print("Provide an integer in the range [0, 255]")
                rule()

        except ValueError:
            print("Provide an integer")
            rule()

    # IS_STABLE
    def is_stable():
        w = input("Do you want to end after a stable solution is found? [y/N]:  ")
        try:
            if w == "":
                print("Default bool {} chosen".format(def_stable))
                return def_stable

            elif w in ['y', 'Y', 'yes', 'YES']:
                return True

            elif w in ['n', 'N', 'no', 'NO']:
                return False

            else:
                print("Answer with yes or no")
                is_stable()

        except ValueError:
            print("Answer with yes or no")
            is_stable()

    print("Setup initialized\nBlank answers taken as default")
    dims = dim()
    if dims is 1:
        return shape(dims), rule(), is_stable()
    else:
        return shape(dims), is_stable()


def main():
    create = setup()
    if create[0][2] is 0:
        cell1 = Cell1D(*create)
        print(cell1.rule)
        # cell1.inline_print()
        cell1.show()
    else:
        cell1 = Cell2D(*create)
        cell1.animate()


if __name__ == '__main__':
    main()
