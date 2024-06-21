import numpy as np
import matplotlib.pylab as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import pickle
from visualizers.base_renderer import MatplotlibRenderer
import math


class GridRoomRenderer(MatplotlibRenderer):
    def __init__(self, size: int, save_gif=False, save_dir=None, show_axis=True):
        self.n_rows = 1
        self.n_cols = size
        super().__init__(xaxis_range=[0, self.n_cols], yaxis_range=[0, self.n_rows], auto_range=False,
                         figure_size=[4, 4], figure_dpi=240, show_axis=show_axis,
                         save_gif=save_gif, save_dir=save_dir)

    def render_render_grid(self, add_line=True):
        # plot grid
        if add_line:
            for i in range(self.n_rows + 1):
                self._axis.add_artist(lines.Line2D([0, self.n_cols], [i, i], color="k", linestyle=":", linewidth=0.5))
            for j in range(self.n_cols + 1):
                self._axis.add_artist(lines.Line2D([j, j], [self.n_rows, 0], color="k", linestyle=":", linewidth=0.5))

    def render_agents(self, agent_pos_list, colors):
        i = 0
        for pos, color in zip(agent_pos_list,  colors):
            row, col = 0, pos
            n_agents_per_row = math.ceil(math.sqrt(len(agent_pos_list)))
            
            width = 1.0 / n_agents_per_row
            agent_row = row + i // n_agents_per_row * width 
            agent_col = col + i % n_agents_per_row * width
            self._axis.add_patch(patches.Rectangle((agent_col, agent_row), 0.2, 0.2,
                                                    facecolor=color, alpha=1))
            i = i + 1


    def mark_target(self, pos, color='g'):
        row, col = 0, pos
        self._axis.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=color, alpha=0.3))