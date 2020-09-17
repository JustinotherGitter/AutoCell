#!usr/bin/env/python
"""
Sandpiles creates a sandpile that recursively topples until no pile is taller than a certain amount

@notes =    [
            \u2588 == █
            ]
"""

__author__ = "Justin Cooper"
__version__ = "26.06.2020"
__email__ = "justin.jb78@gmail.com"

from random import randint
from typing import Tuple, List, Optional, Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from Basics import *


# TODO: Find out how large history should be (t-axis)


class Sandpile(object):
    """

    """

    def __init__(self,
                 shape: Union[int, Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]] = None,
                 height: int = None,
                 max_contents: int = None,
                 processor: str = None,
                 max_history: int = None):
        """

        :param shape:
        :param height:
        :param max_contents:
        """
        if shape is None:
            self._shape = (129, 129)
        elif type(shape) is int:
            if shape % 2 == 0:
                shape += 1
            self._shape = (shape, shape)
        elif type(shape) in [tuple, list] and len(shape) is 2:
            if shape[0] % 2 == 0:
                shape[0] += 1
            if shape[1] % 2 == 0:
                shape[1] += 1
            self._shape = shape
        else:
            raise TypeError("Shape should be None, int, or of type List or Tuple")

        if height is None:
            self._height = randint(12, 2048)
        elif type(height) is int:
            self._height = height
        else:
            raise TypeError("Height should be None or int")

        if max_contents is None:
            self._max_contents = 3
        elif type(max_contents) is int:
            self._max_contents = max_contents
        else:
            raise TypeError("max_contents should be None or int")

        self._processor = processor

        if max_history is None:
            self._max_history = 2048
        elif type(max_history) is int:
            self._max_history = max_history
        else:
            raise TypeError("max_contents should be None or int")

        self._board = np.zeros(self._shape)
        self._board[int(self._shape[0] / 2), int(self._shape[1] / 2)] = self._height
        self._history = self._board

        self.process(self._processor)

    @property
    def height(self):
        print("Getting height.")
        return self._height

    @height.setter
    def height(self, value):
        print("Setting height.")
        self._height = value

    @property
    def shape(self):
        print("Getting shape.")
        return self._shape

    @shape.setter
    def shape(self, value):
        print("Setting shape.")
        if type(value) is int:
            self._shape = (value, value)

    def show(self, frame: int = 0):
        """
        Plots the selected rows of the Cell

        :param  frame       : Frame to display
        :type   frame       : int       (Defaults to starting row)
        """
        if len(self._history.shape) is 2:
            plt.imshow(self._history, aspect='equal')
        else:
            plt.imshow(self._history[int(frame)], aspect='equal')
        plt.xticks([])  # remove the tick marks by setting to an empty list
        plt.yticks([])  # remove the tick marks by setting to an empty list
        plt.show()

    def show3D(self, frame: int = 0):

        z = self._history
        print(z.shape)
        x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

        # show height map in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        ax.set_zlim(0, 7)
        plt.title('z as 3d height map')
        plt.show()

        # show height map in 2d
        plt.figure()
        plt.title('z as 2d heat map')
        p = plt.imshow(z)
        plt.colorbar(p)
        plt.show()

    def animate(self, fps: int = 50, colmap: str = "Greys", save_name: str = None, delay: int = 0, maxheight: int = 5):

        fig = plt.figure()

        ims = []
        for im_data in self._history:
            im = plt.imshow(im_data, animated=True, vmin=0, vmax=maxheight, cmap=colmap)
            ims.append([im])
            if (im_data is self._history[-1]) & (delay != 0):
                for i in range(delay):
                    im = plt.imshow(im_data, animated=True, vmin=0, vmax=maxheight, cmap=colmap)
                    ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)

        if save_name is not None:
            ani.save(save_name)  # 'dynamic_images.mp4')

        plt.show()

        print('Animation complete!')

    def process(self, processor: Callable = None):
        if processor is None:
            self.optimized_processor()
        elif processor is "animation":
            self.animation_processor()
        elif processor is "optimized":
            self.optimized_processor()
        else:
            print("Unrecognized processor.")

    def optimized_processor(self):

        trash = 0
        counter = 0

        t_object = Timer()

        print(f"Starting processing with {self._board.max()} grains" + "\n" +
              "Upper limit on time not calculable due to history deletion")

        while self._board.max() > self._max_contents:
            for x in range(self._shape[0]):
                for y in range(self._shape[1]):

                    # Optimized Sandpile rule
                    if self._history[x, y] > self._max_contents:
                        multiplier = int(self._history[x, y] / 4)
                        self._board[x, y] -= 4 * multiplier
                        try:
                            self._board[x - 1, y] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x + 1, y] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x, y - 1] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x, y + 1] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier

            counter += 1
            self._history = self._board

        print(f"Counter ended at {counter + 1} frames created")
        print(f"{trash} grains fell off the edge")
        print(f"Processing took {t_object.elapsed()}")

    def animation_processor(self):
        self._history = np.zeros((self._max_history, self._shape[0], self._shape[1]))
        self._history[0] = self._board

        trash = 0
        counter = 0

        print(f"Starting processing with {self._board.max()} grains")

        t_object = Timer(self._history.shape[0])
        print_progress_bar(0, self._history.shape[0], prefix='Progress:', suffix='Complete\tETA: N\\A', length=50)

        while self._board.max() > self._max_contents and counter - 1 < self._max_history:
            for x in range(self._shape[0]):
                for y in range(self._shape[1]):

                    # Optimized Sandpile rule
                    # Set multiplier to 1 for unoptimized animation (~ 4 times slower)
                    if self._history[counter, x, y] > self._max_contents:
                        # multiplier = int(self._history[counter, x, y] / 4)
                        multiplier = 1
                        self._board[x, y] -= 4 * multiplier
                        try:
                            self._board[x - 1, y] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x + 1, y] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x, y - 1] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier
                        try:
                            self._board[x, y + 1] += 1 * multiplier
                        except IndexError:
                            trash += 1 * multiplier

            counter += 1
            try:
                self._history[counter] = self._board
            except IndexError:
                self._history.resize((self._history.shape[0] + 100, self._history.shape[1], self._history.shape[2]))
                self._history[counter] = self._board

            print_progress_bar(counter + 1, self._history.shape[0],
                               prefix='Progress:', suffix='Complete\tETA: {}'.format(t_object.remains(counter)),
                               length=50)

        self._history = np.delete(self._history, [range(counter + 1, self._history.shape[0])], 0)
        print(f"Counter ended at {counter + 1} frames created")
        print(f"{trash} grains fell off the edge")
