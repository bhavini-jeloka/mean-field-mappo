import numpy as np
import matplotlib.pylab as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import pickle
from .base_renderer import MatplotlibRenderer
import math


class GridRoomRenderer(MatplotlibRenderer):
    def __init__(self, grid_size: int, save_gif=False, save_dir=None, show_axis=True):
        self.n_rows, self.n_cols = grid_size, grid_size
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

    def render_agents(self, agent_pos_list, status_list, colors):
        plot_pos_list, plot_status_list, plot_color_list = [], [], []
        for pos, status, color in zip(agent_pos_list, status_list, colors):
            if pos not in plot_pos_list:
                plot_pos_list.append(pos)
                plot_status_list.append([status])
                plot_color_list.append([color])
            else:
                index = plot_pos_list.index(pos)
                plot_status_list[index].append(status)
                plot_color_list[index].append(color)

        for pos, status_list, color_list in zip(plot_pos_list, plot_status_list, plot_color_list):
            row, col = pos[0], pos[1]
            n_agents_per_row = math.ceil(math.sqrt(len(status_list)))
            width = 1.0 / n_agents_per_row
            for i, status in enumerate(status_list):
                agent_row = row + i // n_agents_per_row * width
                agent_col = col + i % n_agents_per_row * width
                self._axis.add_patch(patches.Rectangle((agent_col, agent_row), 0.2, 0.2,
                                                       facecolor=color_list[i], alpha=status_list[i] * 0.8 + 0.2))

    def mark_cell(self, pos, color='g'):
        row, col = pos[0], pos[1]
        self._axis.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=color, alpha=0.3))


if __name__ == "__main__":
    color_list = ['r', 'r', 'b', 'b', 'b', 'b']

    pos_list_0 = [(0, 0), (0, 0), (3, 3), (3, 3), (3, 3), (3, 3)]
    status_list_0 = [1, 1, 1, 1, 1, 1]

    pos_list_1 = [(1, 0), (0, 1), (3, 2), (3, 2), (2, 3), (2, 3)]
    status_list_1 = [1, 1, 1, 1, 1, 1]

    pos_list_2 = [(1, 1), (0, 3), (3, 1), (2, 2), (1, 3), (2, 2)]
    status_list_2 = [1, 1, 1, 1, 1, 1]

    pos_list_3 = [(1, 1), (0, 3), (2, 1), (1, 2), (1, 3), (2, 1)]
    status_list_3 = [0, 0, 1, 1, 1, 1]

    renderer = GridRoomRenderer(grid_size=4, save_gif=False, save_dir="results")
    renderer.create_figure()
    renderer.reset()

    renderer.render_render_grid()
    renderer.render_agents(pos_list_0, status_list_0, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(0.1)
    renderer.clear()

    renderer.render_render_grid()
    renderer.render_agents(pos_list_0, status_list_0, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(2)
    renderer.clear()

    renderer.render_render_grid()
    renderer.render_agents(pos_list_1, status_list_1, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(1)
    renderer.clear()

    renderer.render_render_grid()
    renderer.render_agents(pos_list_2, status_list_2, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(1)
    renderer.clear()

    renderer.render_render_grid()
    renderer.render_agents(pos_list_3, status_list_3, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(1)

    renderer.render_render_grid()
    renderer.render_agents(pos_list_3, status_list_3, color_list)
    renderer.mark_cell((2, 2))
    renderer.show()
    renderer.hold(3)
