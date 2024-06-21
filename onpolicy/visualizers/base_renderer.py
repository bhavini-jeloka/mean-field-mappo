import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import imageio
import moviepy.video.io.ImageSequenceClip
import os
from pathlib import Path


class Renderer():
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def create_figure(self):
        pass

    def render_all(self):
        pass


    def render_obstacles(self):
        pass

    def render_trajectories(self):
        pass


class MatplotlibRenderer(Renderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None,
                 axis_equal=True, show_axis=True, save_gif=False, save_dir=None, hold_time=0.3):
        Renderer.__init__(self)
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.auto_range = auto_range
        self.figure_size = figure_size
        self.figure_dpi = figure_dpi
        self.axis_equal = axis_equal
        self.show_axis = show_axis
        self._figure = None
        self._axis = None
        self._hold_time = hold_time

        # for saving animation
        self.save_gif = save_gif
        self.save_dir = save_dir
        self.frame = int(0)
        self.episode = int(0)

    def create_figure(self):
        self._figure = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
        self._axis = self._figure.add_subplot(1, 1, 1)
        self._axis.grid(False)
        if self.axis_equal:
            self._axis.set_aspect('equal', adjustable='box')
        if not self.show_axis:
            self._axis.axis('off')
        plt.figure(self._figure.number)

    def set_range(self):
        if not self.auto_range:
            self._axis.axis([self.xaxis_range[0], self.xaxis_range[1], self.yaxis_range[0], self.yaxis_range[1]])
        plt.grid(False)

    def show(self):
        if not self.show_axis:
            self._axis.axis('off')
        self.set_range()
        plt.pause(0.01)
        if self.save_gif:
            self.save()
        self.frame += 1

    def clear(self):
        plt.cla()

    def reset(self):
        self.episode += 1
        self.frame = 0

    def save(self, save_path_name=None):
        if self.save_dir is not None and not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.set_range()
        if save_path_name is None:
            assert self.save_dir is not None
            save_dir_path = Path(self.save_dir)
            save_path_name = save_dir_path.joinpath('screenshot_{}.png'.format(self.frame))

        plt.savefig(save_path_name)

    def hold(self, t=None):
        if t is not None:
            time.sleep(t)
        else:
            time.sleep(self._hold_time)

    def render_gif(self, episode=None, duration=0.3):
        if episode is None:
            episode = 0

        # get n_frames
        n_frames = 0
        for x in os.listdir(self.save_dir):
            if x.startswith("ep{}".format(episode)) and x.endswith(".png"):
                file_name = x.split(".")[0]
                frame = int(file_name.split("-")[-1])
                if frame > n_frames:
                    n_frames = frame
        frames = [0 for _ in range(3)]
        for frame in range(n_frames + 1):
            frames.append(frame)
        frames += [n_frames for _ in range(3)]

        images = []
        for frame in frames:
            file_name = self.save_dir / 'ep{}-frame-{}.png'.format(episode, frame)
            images.append(imageio.imread(file_name))

        gif_dir = self.save_dir / 'ep{}-movie.gif'.format(episode)
        imageio.mimsave(gif_dir, images, duration=duration)

    def render_mp4(self, episode=None, duration=0.3):
        if episode is None:
            episode = 0

        # get n_frames
        n_frames = 0
        for x in os.listdir(self.save_dir):
            if x.startswith("ep{}".format(episode)) and x.endswith(".png"):
                file_name = x.split(".")[0]
                frame = int(file_name.split("-")[-1])
                if frame > n_frames:
                    n_frames = frame

        image_folder = self.save_dir
        fps = int(1 / duration)

        frames = [0 for _ in range(3)]
        for frame in range(n_frames + 1):
            frames.append(frame)
        frames += [n_frames for _ in range(3)]

        image_files = [os.path.join(image_folder, 'screenshot_{}.png'.format(frame))
                       for frame in frames]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(str(self.save_dir / 'ep{}-movie.mp4'.format(episode)))


if __name__ == "__main__":
    # test render mp4
    from pathlib import Path

    renderer = MatplotlibRenderer(save_dir=Path('../experiment_data/6X6_ROOM/gif'))
    renderer.frame = 42
    renderer.render_mp4()