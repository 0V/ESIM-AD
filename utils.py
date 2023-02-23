import struct
import platform
import contextlib
from typing import Any, List, Tuple, Union, TypeVar, Optional

import cv2
import Imath
import numpy as np
import joblib
import OpenEXR
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ipywidgets import HBox, Play, IntSlider, jslink, interactive_output
from numpy.typing import NDArray
from IPython.display import display

Number = TypeVar('Number', int, float)
Event = Tuple[int, int, float, float]


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):

    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()


def gamma_tmo(x: NDArray[Any], gam: float = 2.2) -> NDArray[np.double]:
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / gam)


def plotPlayer(plot, start, end, step=1):
    interval = (end - start + 1) // 2
    slider = IntSlider(min=start, max=end, step=1, continuous_update=True)
    play = Play(min=start, max=end, step=step, interval=interval, description="Movie")
    jslink((play, 'value'), (slider, 'value'))
    controller = HBox([play, slider])
    output = interactive_output(plot, {"t": slider})
    return display(controller, output)


def plotFrame(frames, t):
    plt.figure(dpi=100)
    plt.imshow(frames[t], cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def evs_to_video(evs: List[Tuple[Number, ...]],
                 video_shape: Union[List[Any], Tuple[Any, ...]],
                 fps: float = 60.0) -> NDArray[np.uint8]:
    """
    video_shape: a tuple like (num_frames, height, width)
    """
    video = np.zeros(video_shape, dtype='float32')
    for x, y, t, p in evs:
        ti = round(t * fps)
        video[ti, y, x] = p

    video = ((video * 0.5 + 0.5) * 255.0).astype('uint8')

    return video


def video_save(filename: str, video: NDArray[np.uint8], fps: float = 60) -> None:
    width = video.shape[2]
    height = video.shape[1]
    assert video.dtype == np.uint8

    if platform.system() == "Windows":
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif platform.system() == "Linux":
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        codec = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(filename, codec, fps, (width, height))

    if video.ndim == 3:
        video = np.expand_dims(video, axis=3)
        video = np.tile(video, (1, 1, 1, 3))

    for frame in video:
        writer.write(frame)

    writer.release()


def video_load(filename: str) -> NDArray[np.uint8]:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError('Failed to open file: %s' % (filename))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mov = np.zeros((frames, height, width, 3), dtype='uint8')
    i = 0
    while True:
        ret, img = cap.read()

        if not ret:
            break

        mov[i] = img
        i += 1

    cap.release()

    return mov


def evs_save(filename: str, evs: List[Event]) -> None:
    n = len(evs)
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', n))
        np.array(evs, dtype='float32').tofile(f)


def evs_load(filename: str) -> List[Event]:
    with open(filename, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        evs_npy = np.fromfile(f, dtype='float32').reshape((n, 4))

    evs = [(int(e[0]), int(e[1]), float(e[2]), float(e[3])) for e in evs_npy]
    return evs


def relative_variance(var_img, img):
    non_zeros_idx = img > 0
    relvar_img = np.zeros_like(var_img)
    relvar_img[non_zeros_idx] = var_img[non_zeros_idx] / (img[non_zeros_idx])
    return relvar_img


def extract_gbuffers(file_path: str) -> Tuple[NDArray[np.double], ...]:
    # open *.exr file
    exr = OpenEXRLoader(file_path)
    img_width, img_height = exr.size[:2]

    # make stacks of images
    color_img = exr.get_image('R', 'G', 'B')
    albedo_img = exr.get_image('Albedo.R', 'Albedo.G', 'Albedo.B')
    normal_img = exr.get_image('Nx', 'Ny', 'Nz')
    depth_img = exr.get_image('Depth')[..., 0]

    var_color_img = exr.get_image('Variance.R', 'Variance.G', 'Variance.B')
    var_albedo_img = exr.get_image('Variance.Albedo.R', 'Variance.Albedo.G', 'Variance.Albedo.B')
    var_normal_img = exr.get_image('Variance.Nx', 'Variance.Ny', 'Variance.Nz')
    var_depth_img = exr.get_image('Variance.Depth')[..., 0]

    xs_img = np.arange(img_width)
    ys_img = np.arange(img_height)
    xs_img, ys_img = np.meshgrid(xs_img, ys_img)

    xyz_img = np.stack([xs_img, ys_img, depth_img], axis=-1)
    var_xyz_img = np.stack([np.zeros_like(xs_img), np.zeros_like(ys_img), var_depth_img], axis=-1)

    return color_img, albedo_img, normal_img, xyz_img, \
           var_color_img, var_albedo_img, var_normal_img, var_xyz_img


class OpenEXRLoader(object):
    def __init__(self, filepath):
        self.pt = Imath.PixelType(Imath.PixelType.FLOAT)
        self.img_exr = OpenEXR.InputFile(filepath)
        dw = self.img_exr.header()['dataWindow']
        self.size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def get_header(self):
        return self.img_exr.header()

    def get_channels(self, *names):
        channels = [np.frombuffer(self.img_exr.channel(name, self.pt), dtype='float32') for name in names]
        imgs = [c_img.reshape(self.size[1], self.size[0], 1) for c_img in channels]
        return imgs

    def get_image(self, *names):
        channels = [np.frombuffer(self.img_exr.channel(name, self.pt), dtype='float32') for name in names]
        imgs = [c_img.reshape(self.size[1], self.size[0]) for c_img in channels]
        return np.stack(imgs, axis=2).astype('double')

    def show_channels(self, *names):
        channels = []
        labels = []

        for name in names:
            c_str = self.img_exr.channel(name, self.pt)
            channels.append(np.frombuffer(c_str, dtype='float32'))
            labels.append(name)

        imgs = []
        for c_img in channels:
            imgs.append(c_img.reshape(self.size[1], self.size[0], 1))

        fig = plt.figure(figsize=(4 * (len(names) + 1), 4))
        tile_length = len(names) + 1
        plot_tile = [100 + 10 * tile_length + (idx + 1) for idx in range(tile_length)]
        axs = [fig.add_subplot(t) for t in plot_tile]

        if len(names) > 1:
            labels.append('all')
            imgs.append(np.array(channels).T.reshape(self.size[1], self.size[0], len(names)))

        for idx in range(len(imgs)):
            img = np.copy(imgs[idx])
            label = labels[idx]
            ax = axs[idx]
            img[img > 1] = 1
            img[img < 0] = 0
            ax.set_title(label)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def show_channels_each(self, *names):
        channels = []
        labels = []

        for name in names:
            c_str = self.img_exr.channel(name, self.pt)
            channels.append(np.frombuffer(c_str, dtype='float32'))
            labels.append(name)

        imgs = []
        for c_img in channels:
            imgs.append(c_img.reshape(self.size[1], self.size[0], 1))

        fig = plt.figure(figsize=(4 * len(names), 4))
        tile_length = len(names)
        plot_tile = [100 + 10 * tile_length + (idx + 1) for idx in range(tile_length)]
        axs = [fig.add_subplot(t) for t in plot_tile]

        for idx in range(len(imgs)):
            img = np.copy(imgs[idx])
            label = labels[idx]
            ax = axs[idx]
            img[img > 1] = 1
            img[img < 0] = 0
            ax.set_title(label)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
