import os
import argparse
from itertools import chain, product

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import joblib
from tqdm.auto import tqdm

from esimad import lin_log, esim_ours, esim_simple, var_lin_log, wlr_denoise
from esimad.utils import evs_save, gamma_tmo, video_save, tqdm_joblib, evs_to_video, extract_gbuffers


def main(args):
    # Make output folder
    data_name = os.path.basename(os.path.normpath(args.data_root))
    out_dir = os.path.join(args.output, "{:s}_{:d}spp_tau{:.2f}".format(data_name.lower(), args.spp, args.thres))
    os.makedirs(out_dir, exist_ok=True)

    # Debug option
    limit_file_length = 30

    # List names of files
    target_dir = os.path.join(args.data_root, f"{args.spp:d}spp")
    exr_files = os.listdir(target_dir)
    exr_files = [os.path.join(target_dir, f) for f in exr_files if f.endswith(".exr")]
    exr_files = sorted(exr_files)

    if args.frame_limit:
        exr_files = exr_files[:limit_file_length]
    print(f"{len(exr_files)} files detected.")

    # Time information
    num_frames = len(exr_files)
    times = np.arange(num_frames) / args.fps

    # Load HDR images (with G-buffers)
    color_imgs = []
    albedo_imgs = []
    normal_imgs = []
    position_imgs = []
    var_color_imgs = []
    var_albedo_imgs = []
    var_normal_imgs = []
    var_position_imgs = []
    for exr_file in tqdm(exr_files):
        (
            color_img,
            albedo_img,
            normal_img,
            position_img,
            var_color_img,
            var_albedo_img,
            var_normal_img,
            var_position_img,
        ) = extract_gbuffers(exr_file)
        color_imgs.append(color_img)
        albedo_imgs.append(albedo_img)
        normal_imgs.append(normal_img)
        position_imgs.append(position_img)
        var_color_imgs.append(var_color_img)
        var_albedo_imgs.append(var_albedo_img)
        var_normal_imgs.append(var_normal_img)
        var_position_imgs.append(var_position_img)

    color_imgs = np.stack(color_imgs, axis=0)
    albedo_imgs = np.stack(albedo_imgs, axis=0)
    normal_imgs = np.stack(normal_imgs, axis=0)
    position_imgs = np.stack(position_imgs, axis=0)
    var_color_imgs = np.stack(var_color_imgs, axis=0)
    var_albedo_imgs = np.stack(var_albedo_imgs, axis=0)
    var_normal_imgs = np.stack(var_normal_imgs, axis=0)
    var_position_imgs = np.stack(var_position_imgs, axis=0)

    gray_imgs = color_imgs[..., 0] * 0.299 + color_imgs[..., 1] * 0.587 + color_imgs[..., 2] * 0.114
    var_gray_imgs = (
        var_color_imgs[..., 0] * (0.299**2)
        + var_color_imgs[..., 1] * (0.587**2)
        + var_color_imgs[..., 2] * (0.114**2)
    )
    gbuf_imgs = np.concatenate([albedo_imgs, normal_imgs, position_imgs], axis=3)
    var_gbuf_imgs = np.concatenate([var_albedo_imgs, var_normal_imgs, var_position_imgs], axis=3)

    video_save(os.path.join(out_dir, "noisy_gray.mp4"), (gamma_tmo(gray_imgs) * 255).astype("uint8"), fps=args.fps)

    # Reference
    gt_dir = os.path.join(args.data_root, "4096spp")
    gt_files = os.listdir(gt_dir)
    gt_files = [os.path.join(gt_dir, f) for f in gt_files if f.endswith(".exr")]
    gt_files = sorted(gt_files)

    if args.frame_limit:
        gt_files = gt_files[:limit_file_length]
    print(f"{len(exr_files)} files detected.")

    gt_imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in gt_files]
    gt_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gt_imgs]
    gt_imgs = np.stack(gt_imgs, axis=0).astype("double")

    gt_gray_imgs = gt_imgs[..., 0] * 0.299 + gt_imgs[..., 1] * 0.587 + gt_imgs[..., 2] * 0.114
    video_save(os.path.join(out_dir, "refs_gray.mp4"), (gamma_tmo(gt_gray_imgs) * 255).astype("uint8"), fps=args.fps)

    img_width = gt_imgs.shape[2]
    img_height = gt_imgs.shape[1]
    with tqdm_joblib(img_width * img_height, desc="Refs - ESIM"):
        res_refs = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(esim_simple)(gt_gray_imgs[:, y, x], times, (x, y), args.thres)
            for y, x in product(range(img_height), range(img_width))
        )

    refs_evs = list(chain.from_iterable(res_refs))
    print("refs: %d events detected." % (len(refs_evs)))

    refs_video = evs_to_video(refs_evs, video_shape=(num_frames, img_height, img_width))
    video_save(os.path.join(out_dir, "refs_evs.mp4"), refs_video, fps=args.fps)
    np.save(os.path.join(out_dir, "refs_evs.npy"), refs_video)
    evs_save(os.path.join(out_dir, "refs_evs.raw"), refs_evs)

    # Simple ESIM (esim)
    print("*** ESIM (esim) ***")
    img_width = gray_imgs.shape[2]
    img_height = gray_imgs.shape[1]
    with tqdm_joblib(img_width * img_height, desc="ESIM"):
        res_esim = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(esim_simple)(gray_imgs[:, y, x], times, (x, y), args.thres)
            for y, x in product(range(img_height), range(img_width))
        )

    esim_evs = list(chain.from_iterable(res_esim))
    print("esim: %d events detected." % (len(esim_evs)))

    esim_video = evs_to_video(esim_evs, video_shape=(num_frames, img_height, img_width))
    video_save(os.path.join(out_dir, "esim_evs.mp4"), esim_video, fps=args.fps)
    np.save(os.path.join(out_dir, "esim_evs.npy"), esim_video)
    evs_save(os.path.join(out_dir, "esim_evs.raw"), esim_evs)

    # WLR-ESIM (alph)
    print("*** WLR-ESIM (alph) ***")
    img_width = gray_imgs.shape[2]
    img_height = gray_imgs.shape[1]
    with tqdm_joblib(img_width * img_height, desc="WLR denoise"):
        alph_imgs = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(wlr_denoise)(
                gray_imgs,
                gbuf_imgs,
                var_gray_imgs,
                var_gbuf_imgs,
                (x, y),
                ksize=args.ksize,
                spp=args.spp,
                method=args.wlr,
            )
            for y, x in product(range(img_height), range(img_width))
        )

    alph_imgs = np.stack(alph_imgs, axis=1).reshape(-1, img_height, img_width)
    video_save(os.path.join(out_dir, "alph_gray.mp4"), (gamma_tmo(alph_imgs) * 255).astype("uint8"))

    with tqdm_joblib(img_width * img_height, desc="WLR to ESIM"):
        res_wlr_esim = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(esim_simple)(alph_imgs[:, y, x], times, (x, y), args.thres)
            for y, x in product(range(img_height), range(img_width))
        )

    wlr_esim_evs = list(chain.from_iterable(res_wlr_esim))
    print("alph: %d events detected." % (len(wlr_esim_evs)))

    wlr_esim_video = evs_to_video(wlr_esim_evs, video_shape=(num_frames, img_height, img_width))
    video_save(os.path.join(out_dir, "alph_evs.mp4"), wlr_esim_video, fps=args.fps)
    np.save(os.path.join(out_dir, "alph_evs.npy"), wlr_esim_video)
    evs_save(os.path.join(out_dir, "alph_evs.raw"), wlr_esim_evs)

    # ESIM-AD (ours)
    print("*** ESIM-AD (ours) ***")
    img_width = gray_imgs.shape[2]
    img_height = gray_imgs.shape[1]
    log_gray_imgs = lin_log(gray_imgs)
    var_log_gray_imgs = var_lin_log(gray_imgs, var_gray_imgs)
    with tqdm_joblib(img_width * img_height, desc="ESIM-AD"):
        res_ours = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(esim_ours)(
                log_gray_imgs,
                gbuf_imgs,
                var_log_gray_imgs,
                var_gbuf_imgs,
                times,
                (x, y),
                ksize=args.ksize,
                spp=args.spp,
                threshold=args.thres,
                method=args.wlr,
            )
            for y, x in product(range(img_height), range(img_width))
        )

    ours_evs = list(chain.from_iterable((t[0] for t in res_ours)))
    wlr_count = sum(t[1] for t in res_ours)
    print(
        "ours: %d events detected. %d WLR fits (%.2f %%)."
        % (len(ours_evs), wlr_count, 100.0 * wlr_count / (img_width * img_height * num_frames))
    )

    ours_video = evs_to_video(ours_evs, video_shape=(num_frames, img_height, img_width))
    video_save(os.path.join(out_dir, "ours_evs.mp4"), ours_video, fps=args.fps)
    np.save(os.path.join(out_dir, "ours_evs.npy"), ours_video)
    evs_save(os.path.join(out_dir, "ours_evs.raw"), ours_evs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="denoise-esim")

    # yapf: disable
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--spp', type=int, default=32)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--ksize', type=int, default=13)
    parser.add_argument('--thres', type=float, default=1.0)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--wlr', type=str, choices=['simple', 'tsvd', 'varopt'])
    parser.add_argument('--frame_limit', action='store_true',
                        help='Use only first 10 frames, if specified')
    # yapf: enable

    args = parser.parse_args()
    main(args)
