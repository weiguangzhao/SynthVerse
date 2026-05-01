from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig

from datasets.providers.base_provider import BaseDataProvider
from datasets.datatypes import RawSliceData
from datasets.utils.geometry import batch_distance_to_depth_np

logger = logging.getLogger(__name__)


# -----------------------------
# index cache
# -----------------------------

def _default_index_cache_path(data_root: str) -> str:
    return osp.join(data_root, "index.json")


def _save_index(cache_path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _load_index(cache_path: str) -> Dict[str, Any]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scan_frame_ids(frames_dir: str) -> List[int]:
    """
    慢：会列 frames 目录下所有文件。
    仅在 build_mode="scan_frames" 时用。
    """
    ids: List[int] = []
    with os.scandir(frames_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.endswith(".png"):
                continue
            if name.endswith("_depth.png"):
                continue
            stem = name[:-4]
            if not stem.isdigit():
                continue
            ids.append(int(stem))
    ids.sort()
    return ids


def _build_index(
    data_root: str,
    frame_digits: int,
    build_mode: str = "from_anno",  # "from_anno" | "scan_frames"
) -> Dict[str, Any]:
    """
    build_mode="from_anno":
      - 只读每个 seq 的 npy（通常比列 frames 快很多）
      - 默认帧编号是 0..T-1
    build_mode="scan_frames":
      - 列 frames 目录，拿真实存在的帧 id（最稳，但对象存储很慢）
    """
    sequences = sorted([d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))])

    annot_paths: Dict[str, str] = {}
    frames_dirs: Dict[str, str] = {}
    frame_ids: Dict[str, List[int]] = {}
    num_imgs: Dict[str, int] = {}

    for i, seq in enumerate(sequences):
        seq_dir = osp.join(data_root, seq)
        annot_p = osp.join(seq_dir, f"{seq}.npy")
        frames_d = osp.join(seq_dir, "frames")

        if not osp.isfile(annot_p):
            logger.warning(f"[OurDataProvider][index] missing annot: {annot_p} (skip)")
            continue

        annot_paths[seq] = annot_p
        frames_dirs[seq] = frames_d

        if build_mode == "scan_frames":
            ids = _scan_frame_ids(frames_d)
            frame_ids[seq] = ids
            num_imgs[seq] = int(len(ids))
            logger.info(f"[OurDataProvider][index] ({i+1}/{len(sequences)}) seq={seq} frames={len(ids)}")
        else:
            # from_anno (fast for object storage)
            annot = np.load(annot_p, allow_pickle=True).item()
            T = int(len(annot["intrinsics"]))
            frame_ids[seq] = list(range(T))
            num_imgs[seq] = T
            logger.info(f"[OurDataProvider][index] ({i+1}/{len(sequences)}) seq={seq} frames={T} (from anno)")

    kept_sequences = sorted(list(annot_paths.keys()))

    return dict(
        version=1,
        created_time=time.time(),
        data_root=osp.abspath(data_root),
        frame_digits=int(frame_digits),
        build_mode=str(build_mode),
        sequences=kept_sequences,
        annot_paths=annot_paths,
        frames_dirs=frames_dirs,
        frame_ids=frame_ids,
        num_imgs=num_imgs,
    )


# -----------------------------
# depth helper
# -----------------------------

def _distance16_to_metric(raw_u16: np.ndarray, depth_range: Tuple[float, float]) -> np.ndarray:
    """
    16-bit depth png encode: camera distance mapped to [near, far]
    """
    raw = raw_u16.astype(np.float32)
    d0, d1 = float(depth_range[0]), float(depth_range[1])
    return d0 + raw * (d1 - d0) / 65535.0


def _dynamic_filter_mask(
    traj_3d_TN3: np.ndarray,
    threshold: float,
    keep_static_prob: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    traj_3d_TN3: [T, N, 3]
    return: mask [N] or None
    """
    thr = float(threshold)
    ksp = float(keep_static_prob)
    if thr <= 0.0 and ksp >= 1.0:
        return None
    if traj_3d_TN3.size == 0 or traj_3d_TN3.shape[1] == 0:
        return None

    max_xyz = np.max(np.abs(traj_3d_TN3), axis=0)  # [N,3]
    min_xyz = np.min(np.abs(traj_3d_TN3), axis=0)  # [N,3]
    diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)  # [N]

    mask = diff_l2 > thr
    if ksp < 1.0:
        keep_mask = rng.random(mask.shape[0], dtype=np.float32) < ksp
        mask = mask | keep_mask
    return mask


# -----------------------------
# Provider
# -----------------------------

class OurDataProvider(BaseDataProvider):
    """
    Directory layout:
      data_root/
        seq_name/
          seq_name.npy
          frames/
            0000.png
            0000_depth.png
            ...

    annot keys (按你旧 OurDataset 推断):
      depth_range: [2]
      coords     : [T, N, 2]
      traj_3d    : [T, N, 3]
      occluded   : [T, N]
      intrinsics : [T, 3, 3]    (pixel unit)
      matrix_world: [T, 4, 4]   (c2w)
    """

    def __init__(self, cfg: DictConfig, override_anno: Optional[str] = None):
        super().__init__(cfg, override_anno)

        self.data_root: str = cfg.data_root
        self.frame_digits: int = int(cfg.get("frame_digits", 4))

        self.dynamic_filter_threshold: float = float(cfg.get("dynamic_filter_threshold", 0.0))
        self.keep_static_prob: float = float(cfg.get("keep_static_prob", 1.0))

        index_cache_path: str = str(cfg.get("index_cache_path", _default_index_cache_path(self.data_root)))
        build_mode: str = str(cfg.get("index_build_mode", "from_anno"))  # "from_anno" or "scan_frames"
        force_rebuild: bool = bool(cfg.get("force_rebuild_index", False))

        if (not force_rebuild) and osp.isfile(index_cache_path):
            cache = _load_index(index_cache_path)
            logger.info(f"[OurDataProvider] Loaded index: {index_cache_path}")
        else:
            logger.info(f"[OurDataProvider] Build index -> {index_cache_path} (mode={build_mode})")
            cache = _build_index(self.data_root, frame_digits=self.frame_digits, build_mode=build_mode)
            _save_index(index_cache_path, cache)

        self.sequences: List[str] = list(cache["sequences"])
        self.annot_paths: Dict[str, str] = {k: str(v) for k, v in cache["annot_paths"].items()}
        self.frames_dirs: Dict[str, str] = {k: str(v) for k, v in cache["frames_dirs"].items()}
        self.frame_ids: Dict[str, List[int]] = {k: [int(x) for x in v] for k, v in cache["frame_ids"].items()}
        self.num_imgs: Dict[str, int] = {k: int(v) for k, v in cache["num_imgs"].items()}

        # 和 KubricDataProvider 一样保留 video_folders（这里用 seq 的完整路径）
        self.video_folders: List[str] = [osp.join(self.data_root, s) for s in self.sequences]

        logger.info(f"Loaded Our data from {self.data_root} with {len(self.sequences)} sequences")

    def load_seq_lens(self) -> List[int]:
        # 直接用 index，避免每次读 npy
        return [int(self.num_imgs[s]) for s in self.sequences]

    def _load_slice(
        self,
        seq_id: int,
        start: int,
        length: int,
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        seq_name = self.sequences[seq_id]
        annot_file = self.annot_paths[seq_name]
        frames_dir = self.frames_dirs[seq_name]
        ids_all = self.frame_ids[seq_name]

        end = start + length * stride
        local_ids = ids_all[start:end:stride]
        if len(local_ids) != length:
            # 理论上 BaseDataProvider 会保证长度够，这里做一下兜底
            if len(local_ids) == 0:
                local_ids = [0] * length
            else:
                local_ids = local_ids + [local_ids[-1]] * (length - len(local_ids))

        annot = np.load(annot_file, allow_pickle=True).item()
        depth_range = annot["depth_range"]

        # Our: coords/traj/occluded 预期是 [T,N,...]
        traj_2d = np.asarray(annot["coords"], dtype=np.float32)[local_ids]          # [T,N,2]
        traj_3d = np.asarray(annot["traj_3d"], dtype=np.float32)[local_ids]         # [T,N,3]
        visibs = ~np.asarray(annot["occluded"], dtype=bool)[local_ids]              # [T,N]

        intrinsics = np.asarray(annot["intrinsics"], dtype=np.float32)[local_ids]   # [T,3,3] pixel
        c2w = np.asarray(annot["matrix_world"], dtype=np.float32)[local_ids]        # [T,4,4]
        extrinsics = np.linalg.inv(c2w).astype(np.float32)                          # [T,4,4] w2c

        # read rgb + distance (parallel)
        rgb_futs: List[Future[np.ndarray]] = []
        dist_futs: List[Future[np.ndarray]] = []

        for frame_id in local_ids:
            rgb_path = osp.join(frames_dir, f"{frame_id:0{self.frame_digits}d}.png")
            dep_path = osp.join(frames_dir, f"{frame_id:0{self.frame_digits}d}_depth.png")

            def _read_rgb(p: str) -> np.ndarray:
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(p)
                return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            def _read_dist(p: str, dr: Tuple[float, float]) -> np.ndarray:
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise FileNotFoundError(p)
                return _distance16_to_metric(raw, dr)

            rgb_futs.append(executor.submit(_read_rgb, rgb_path))
            dist_futs.append(executor.submit(_read_dist, dep_path, tuple(depth_range)))

        rgbs_list: List[np.ndarray] = [f.result() for f in rgb_futs]
        distances = np.stack([f.result() for f in dist_futs], axis=0).astype(np.float32)  # [T,H,W]
        rgbs = np.stack(rgbs_list, axis=0).astype(np.uint8)  # [T,H,W,3]

        H, W = rgbs.shape[1], rgbs.shape[2]

        # boundary + finite check
        finite = np.isfinite(traj_2d).all(axis=-1)  # [T,N]
        in_x = (traj_2d[:, :, 0] >= 0) & (traj_2d[:, :, 0] <= (W - 1))
        in_y = (traj_2d[:, :, 1] >= 0) & (traj_2d[:, :, 1] <= (H - 1))
        visibs = visibs & finite & in_x & in_y

        # dynamic filter (apply to traj_2d + traj_3d + visibs)
        maskN = _dynamic_filter_mask(
            traj_3d_TN3=traj_3d,
            threshold=self.dynamic_filter_threshold,
            keep_static_prob=self.keep_static_prob,
            rng=rng,
        )
        if maskN is not None:
            traj_2d = traj_2d[:, maskN]
            traj_3d = traj_3d[:, maskN]
            visibs = visibs[:, maskN]

        # Our depth png is camera distance -> convert to z-depth
        depths = batch_distance_to_depth_np(distances, intrinsics).astype(np.float32)  # [T,H,W]

        return RawSliceData.create(
            seq_name=self.video_folders[seq_id],
            seq_id=seq_id,
            gt_trajs_3d=traj_3d,
            visibs=visibs,
            valids=np.ones_like(visibs, dtype=np.bool_),
            rgbs=rgbs,
            gt_depths=depths,
            gt_query_point=None,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([H, W], dtype=np.int32),  # (h, w)
            copy_gt_to_est=True,
        )
