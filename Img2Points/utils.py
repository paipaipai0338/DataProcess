import copy
from dataclasses import dataclass
import cv2
import scipy.io as sio
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

@dataclass
class Ray:
    """
    射线模型： 一个 cam 的一个像素点对应一个 Ray
    """
    '''Ray的时空属性'''
    # Ray 对应的camera
    camera_id: str  # 相机 ID
    # Ray所处的相机坐标系原点 世界坐标系前用np.array([0, 0, 0], dtype=np.float64)初始化
    origin: np.ndarray      # [3,]
    # Ray 的射线在相机坐标系下的朝向
    direction: np.ndarray   # [3,]
    # 帧 ID
    frame_id: int
    # Ray对应的像素坐标 [u, v]
    pixel: np.ndarray       # [2,]
    # cam 下像素点索引 chessboard 对应焦点顺序，2D keypoint 对应关键点索引
    pixel_id: int

    '''Ray的估计结果属性'''
    # 多相机联立求解深度
    depth: float
    # 2D 估计结果中的置信度
    score: float
    # 人 ID 这里由于 2D 结果具有整体性，可作为后续关键点匹配判据
    person_id: int
    # Ray 的有效性，由于会出现遮挡等情况，保证关键点的集合空间不变性
    valid: bool

def load_extrinsic_data(file_full_path):
    T = np.load(file_full_path)
    R = T[:3, :3]
    t = T[:3, 3]
    return T, R, t

def load_intrinsic_data(file_full_path):
    file_type = Path(file_full_path).suffix.lower()
    if file_type == ".mat":
        data = sio.loadmat(file_full_path)
        K = data["K"]
        dist = data["dist"]
    elif file_type == ".npz":
        data = np.load(file_full_path, allow_pickle=True)
        K = data["K"]
        dist = data["dist"]
    else:
        K = None
        dist = None
    return K, dist

def build_ray_from_pixel(u, v, K, dist, camera_id, pixel_id, score, frame_id, person_id, valid):
    """
    从像素中构建 Ray
    :param u: 像素坐标 (u,v)
    :param v: 像素坐标 (u,v)
    :param K: 相机内参
    :param dist: 相机内参
    :param camera_id: Ray 归属
    :param pixel_id: Ray 归属
    :param score: 估计结果的可靠性，来源于视觉模型
    :param frame_id: Ray 归属
    :param person_id: Ray 归属
    :param valid: Ray 的可靠性
    :return: Ray dataclass
    """
    # 像素点 OpenCV 需要 Nx1x2 格式
    pts_obs = np.array([[[u, v]]], dtype=np.float32)  # shape (1,1,2)

    # 去畸变 (Undistort Points)
    pts_corrected = cv2.undistortPoints(pts_obs, K, dist, P=None)
    x_norm = pts_corrected[0, 0, 0]
    y_norm = pts_corrected[0, 0, 1]

    # 构建相机系下的方向向量 (未单位化，Z=1)
    dir_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    # 单位化方向向量
    dir_cam_unit = dir_cam / np.linalg.norm(dir_cam)

    return Ray(
        origin=np.array([0, 0, 0], dtype=np.float64),
        direction=dir_cam_unit,
        pixel=np.array([u, v], dtype=np.float64),
        camera_id=camera_id,
        pixel_id=pixel_id,
        depth=0.0,
        score=score,
        frame_id=frame_id,
        person_id=person_id,
        valid=valid,
    )

def calculate_chessboard_3D_coordinate(pixel_id, frame_id, rays, person_id=-1, reference_cam="A", calib_path=None):
    """
    求解棋盘格数据的 3D 坐标，确保相机的像素 id 在多个 cam 下是对应关系，否则求解失败
    :param pixel_id: 相机的像素id
    :param frame_id: 对应帧数
    :param person_id: 棋盘格数据无此信息，但为保证集合不变性传入-1
    :param rays: 所有 ray 集合 List[Ray]
    :param reference_cam: cam id str
    :return:
        coordinate: 求解结果
        rays_to_cam_ref: 将所有 Ray 的空间信息转换至 reference_cam坐标系下
    """
    rays_seleted = [r for r in rays if r.pixel_id == pixel_id]
    rays_seleted = [r for r in rays_seleted if r.person_id == person_id]
    rays_seleted = [r for r in rays_seleted if r.frame_id == frame_id]
    if calib_path is None:
        raise ValueError("calib_path is required")
    rays_to_cam_ref = [
        transform_ray_to_reference(r, calib_path=calib_path, reference_cam=reference_cam)
        for r in rays_seleted
    ]

    I = np.eye(3)
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for p in rays_to_cam_ref:
        o = p.origin
        d = p.direction
        P = I - np.outer(d, d)
        wi = p.score if p.score > 0.5 else 0
        A += wi * P
        b += wi * (P @ o)

    coordinate, *_ = np.linalg.lstsq(A, b, rcond=None)
    for p in rays_to_cam_ref:
        p.depth = float(p.direction @ (coordinate - p.origin))
    return coordinate, rays_to_cam_ref

def transform_ray_to_reference(ray, calib_path, reference_cam="A"):
    """
    将传入的单个 Ray 中的空间属性对齐到 reference_cam，即完成坐标系转换
    :param ray: dataclass
    :param reference_cam: cam id str
    :return: ray dataclass
    """
    calib_path = Path(calib_path)
    if ray.camera_id == reference_cam:
        R = np.eye(3)
        t = np.zeros(3, dtype=float)
    else:
        _, R, t = load_extrinsic_data(
            calib_path / f"extrinsic_T_cam_{ray.camera_id}_to_cam_{reference_cam}.npy"
        )
    r = copy.copy(ray)
    r.origin = ray.origin + t
    r.direction = R @ ray.direction
    return r

class ImagePixelPicker:
    MISSING_PIXEL = (np.inf, np.inf, np.inf, np.inf, np.inf)

    def __init__(self, image_path, results_dict, image_key, expected_points=5):
        """
        Initialize image pixel picker

        Args:
            image_path: Path to the image file
            results_dict: Dictionary to store results {image_path: pixels}
            image_key: Key to use in results_dict (usually the image full path)
            expected_points: Number of fixed-order corner reflector IDs to record
        """
        # Read image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Store results dictionary and key
        self.results_dict = results_dict
        self.image_key = image_key
        self.expected_points = int(expected_points)

        # Convert BGR to RGB for display
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Store recorded pixels
        self.pixels = []  # Store (x, y, r, g, b)

        # Create GUI
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)

        # Display image
        self.im = self.ax.imshow(self.image_rgb)
        self.ax.set_title('Click on image to record pixel coordinates and color values\nPress Save to store results',
                          fontsize=12)
        self.ax.axis('off')

        # Create buttons
        ax_undo = plt.axes([0.04, 0.05, 0.1, 0.05])
        ax_missing = plt.axes([0.20, 0.05, 0.13, 0.05])
        ax_skip = plt.axes([0.39, 0.05, 0.1, 0.05])
        ax_clear = plt.axes([0.55, 0.05, 0.1, 0.05])
        ax_save = plt.axes([0.85, 0.05, 0.1, 0.05])

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_missing = Button(ax_missing, 'Missing')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_skip = Button(ax_skip, 'Skip')

        # Bind events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.btn_undo.on_clicked(self.undo_last)
        self.btn_missing.on_clicked(self.mark_missing)
        self.btn_save.on_clicked(self.save_pixels)
        self.btn_clear.on_clicked(self.clear_all)
        self.btn_skip.on_clicked(self.skip_image)

        # Display selected points
        self.scatter = self.ax.scatter([], [], c='red', s=50, marker='o')
        self.texts = []  # Store annotation texts

        # Status display
        self.status_text = self.fig.text(0.5, 0.02, 'Ready - Click on image to record pixels',
                                         ha='center', va='bottom', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.show()

    def on_click(self, event):
        """Mouse click event handler"""
        if event.inaxes != self.ax:
            return
        if len(self.pixels) >= self.expected_points:
            self.status_text.set_text(
                f'Already recorded {self.expected_points} IDs. Use Undo/Clear or Save.')
            self.fig.canvas.draw_idle()
            return

        # Get pixel coordinates
        x, y = int(event.xdata), int(event.ydata)

        # Get color values (RGB)
        if 0 <= x < self.image_rgb.shape[1] and 0 <= y < self.image_rgb.shape[0]:
            r, g, b = [int(v) for v in self.image_rgb[y, x]]

            # Store pixel info
            self.pixels.append((x, y, r, g, b))
            point_id = len(self.pixels) - 1

            # Update display
            self.update_display()

            # Update status
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            self.status_text.set_text(
                f'Recorded ID {point_id}/{self.expected_points - 1} | Last: ({x},{y}) RGB({r},{g},{b}) {hex_color}')

            # Output to console
            print(f"Recorded ID {point_id}: Coordinates({x}, {y}) RGB({r}, {g}, {b}) HEX:{hex_color}")
            self.fig.canvas.draw_idle()

    def update_display(self):
        """Update markers on image"""
        for text in self.texts:
            text.remove()
        self.texts.clear()

        visible_points = []
        for point_id, (x, y, r, g, b) in enumerate(self.pixels):
            if np.isfinite(x) and np.isfinite(y):
                visible_points.append((x, y))
                text = self.ax.text(x + 5, y - 5, f'{point_id}',
                                    fontsize=10, color='red',
                                    fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white', alpha=0.7))
                self.texts.append(text)

        if visible_points:
            self.scatter.set_offsets(np.asarray(visible_points))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

        self.fig.canvas.draw_idle()

    def undo_last(self, event):
        """Undo last selection"""
        if self.pixels:
            removed = self.pixels.pop()
            self.update_display()
            removed_id = len(self.pixels)
            if np.isfinite(removed[0]) and np.isfinite(removed[1]):
                hex_color = f'#{int(removed[2]):02x}{int(removed[3]):02x}{int(removed[4]):02x}'
                self.status_text.set_text(
                    f'Undid ID {removed_id} ({removed[0]}, {removed[1]}), {len(self.pixels)} IDs remaining')
                print(
                    f"Undo ID {removed_id}: Coordinates({removed[0]}, {removed[1]}) {hex_color} Remaining {len(self.pixels)} IDs")
            else:
                self.status_text.set_text(
                    f'Undid missing ID {removed_id}, {len(self.pixels)} IDs remaining')
                print(f"Undo missing ID {removed_id}: Remaining {len(self.pixels)} IDs")

    def mark_missing(self, event):
        """Record current fixed-order ID as missing."""
        if len(self.pixels) >= self.expected_points:
            self.status_text.set_text(
                f'Already recorded {self.expected_points} IDs. Use Undo/Clear or Save.')
            self.fig.canvas.draw_idle()
            return

        self.pixels.append(self.MISSING_PIXEL)
        missing_id = len(self.pixels) - 1
        self.update_display()
        self.status_text.set_text(
            f'Marked ID {missing_id}/{self.expected_points - 1} as missing: (inf, inf)')
        print(f"Marked missing ID {missing_id}: {self.MISSING_PIXEL}")
        self.fig.canvas.draw_idle()

    def _fixed_length_pixels(self):
        """Return exactly expected_points records, padding missing IDs with inf."""
        pixels = list(self.pixels[:self.expected_points])
        while len(pixels) < self.expected_points:
            pixels.append(self.MISSING_PIXEL)
        return pixels

    def clear_all(self, event):
        """Clear all records"""
        self.pixels.clear()
        self.update_display()
        self.status_text.set_text(f'Cleared all {len(self.pixels)} points')
        print("Cleared all recorded points")

    def save_pixels(self, event):
        """Save pixels to results dictionary and close"""
        # Store in dictionary
        pixels = self._fixed_length_pixels()
        self.results_dict[self.image_key] = pixels
        self.pixels = pixels

        # Update status
        self.status_text.set_text(f'Saved {len(pixels)} fixed-order IDs for {self.image_key}')
        print(f"\n✓ Saved {len(self.pixels)} points for: {self.image_key}")
        print(f"  Total images processed: {len(self.results_dict)}")

        # Close the figure after a short delay
        plt.close(self.fig)

    def skip_image(self, event):
        """Skip current image without saving"""
        self.results_dict[self.image_key] = [self.MISSING_PIXEL] * self.expected_points
        print(f"\n✓ Skipped image: {self.image_key}")
        print(f"  Total images processed: {len(self.results_dict)}")
        plt.close(self.fig)


def get_corner_pixel_from_img(img_path: Path, pkl_save_path: Path, expected_points: int = 3) -> dict:
    """
    Pick corner pixels from images and save them to a pickle file.

    The saved pickle format is the same format consumed by the reconstruction
    code below:
        { "cam_xxx/frame.jpg": [(x, y, r, g, b), ...], ... }

    Missing fixed-order IDs are saved as:
        (np.inf, np.inf, np.inf, np.inf, np.inf)

    If pkl_save_path already exists, this function resumes from it and skips
    image keys that have already been saved or explicitly skipped.
    """
    img_path = Path(img_path)
    pkl_save_path = Path(pkl_save_path)
    pkl_save_path.parent.mkdir(parents=True, exist_ok=True)

    if pkl_save_path.exists():
        with open(pkl_save_path, 'rb') as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Existing pickle must contain a dict: {pkl_save_path}")
        pixel_records = loaded
        print(f"Loaded {len(pixel_records)} existing records from: {pkl_save_path}")
    else:
        pixel_records = {}
        print(f"Creating new pixel record file: {pkl_save_path}")

    def save_records() -> None:
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(pixel_records, f)

    def is_complete_record(pixels) -> bool:
        return (
            len(pixels) == expected_points
            and all(len(pixel) >= 5 for pixel in pixels)
        )

    cams = sorted(os.listdir(img_path))

    for cam in cams:
        cam_path = img_path / cam / "frames"
        if not cam_path.exists():
            print(f"Warning: Path does not exist: {cam_path}")
            continue

        img_files = [
            name for name in sorted(os.listdir(cam_path))
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ][:1]  # Sort for consistent ordering

        for img_file in img_files:
            img_full_path = str(cam_path / img_file)
            img_key = f"{cam}"
            if img_key in pixel_records and is_complete_record(pixel_records[img_key]):
                print(f"Skip existing record: {img_key} ({len(pixel_records[img_key])} points)")
                continue
            if img_key in pixel_records:
                print(
                    f"Reprocess incomplete record: {img_key} ({len(pixel_records[img_key])}/{expected_points} points)")

            print(f"\n--- Processing: {img_key} ---")

            try:
                ImagePixelPicker(img_full_path, pixel_records, img_key, expected_points=expected_points)

            except Exception as e:
                print(f"Error processing {img_key}: {e}")
                pixel_records[img_key] = [ImagePixelPicker.MISSING_PIXEL] * expected_points

            save_records()
            print(f"Saved progress to: {pkl_save_path}")

    for img_key, pixels in pixel_records.items():
        if pixels:
            print(f"  ✓ {img_key}: {len(pixels)} points")
        else:
            print(f"  - {img_key}: No points recorded")

    for img_key, pixels in list(pixel_records.items())[:3]:  # Show first 3 entries
        print(f"    '{img_key}': [")
        for pixel in pixels[:2]:  # Show first 2 pixels
            x, y, r, g, b = pixel
            print(f"        ({x}, {y}, {r}, {g}, {b}),")
        if len(pixels) > 2:
            print(f"        ... and {len(pixels) - 2} more")
        print(f"    ],")
    if len(pixel_records) > 3:
        print(f"    ... and {len(pixel_records) - 3} more")
    print("}")
    save_records()
    print(f"Final saved {len(pixel_records)} records to: {pkl_save_path}")
    return pixel_records

def _ray_point_distance(point, ray):
    vec = point - ray.origin
    return float(np.linalg.norm(np.cross(vec, ray.direction)) / np.linalg.norm(ray.direction))


def _calculate_corner_coordinate_robust(
    pixel_id,
    corner_rays,
    calib_path,
    reference_cam="A",
    min_rays=2,
    max_ray_error=0.03,
):
    if len(corner_rays) < min_rays:
        raise ValueError(f"corner {pixel_id}: only {len(corner_rays)} valid rays, need at least {min_rays}")

    active_rays = list(corner_rays)
    removed = []
    last_errors = {}

    while True:
        coordinate_3d, rays_to_ref = calculate_chessboard_3D_coordinate(
            pixel_id=pixel_id,
            rays=active_rays,
            frame_id=-1,
            person_id=-1,
            reference_cam=reference_cam,
            calib_path=calib_path,
        )

        if not np.isfinite(coordinate_3d).all():
            raise ValueError(f"corner {pixel_id}: non-finite 3D coordinate from ray solve")

        last_errors = {
            r.camera_id: _ray_point_distance(coordinate_3d, r)
            for r in rays_to_ref
        }
        if not last_errors:
            raise ValueError(f"corner {pixel_id}: no rays left after transform")

        worst_cam, worst_error = max(last_errors.items(), key=lambda item: item[1])
        if worst_error <= max_ray_error or len(active_rays) <= min_rays:
            return coordinate_3d, rays_to_ref, last_errors, removed

        removed.append((worst_cam, worst_error))
        active_rays = [r for r in active_rays if r.camera_id != worst_cam]


def get_corner_coordinate(
    pkl_save_path: Path,
    calib_path: Path,
    min_rays: int = 2,
    max_ray_error: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    with open(str(pkl_save_path), 'rb') as f:
        data = pickle.load(f)
    coordinates = {}
    for k, vs in data.items():
        cam_name = str(k).split("/")[-1]
        cam = cam_name.split("_", 1)[1] if cam_name.startswith("cam_") else cam_name
        coordinates[cam] = [v[:2] for v in vs]

    rays = []
    for k, v in coordinates.items():
        for idx in range(len(v)):
            cam_id = k.split('_')[-1]
            K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam_id}.npz")
            if np.isfinite(v[idx][0]) and np.isfinite(v[idx][1]):
                ray = build_ray_from_pixel(v[idx][0], v[idx][1], K, dist, camera_id=cam_id, pixel_id=idx, score=1 , frame_id=-1, person_id=-1, valid=True)
                rays.append(ray)

    pixel_ids = [0, 1, 2]
    coordinates = []
    error = {}
    cam_pos = {}
    for pixel_id in pixel_ids:
        corner_rays = [ray for ray in rays if ray.pixel_id == pixel_id]

        coordinate_3d, rays_to_ref, ray_errors, removed = _calculate_corner_coordinate_robust(
            pixel_id=pixel_id,
            corner_rays=corner_rays,
            calib_path=calib_path,
            reference_cam="A",
            min_rays=min_rays,
            max_ray_error=max_ray_error,
        )
        coordinates.append(coordinate_3d)
        error[f"corner{pixel_id}_used_cams"] = sorted(ray_errors.keys())
        error[f"corner{pixel_id}_removed_cams"] = [
            {"cam": cam, "error": float(err)}
            for cam, err in removed
        ]
        error[f"corner{pixel_id}_max_ray_error"] = float(max(ray_errors.values()))
        error[f"corner{pixel_id}_mean_ray_error"] = float(np.mean(list(ray_errors.values())))
        for r in rays_to_ref:
            if r.camera_id not in cam_pos.keys():
                cam_pos[r.camera_id] = r.origin
            error["corner{}_cam{}".format(pixel_id, r.camera_id)] = ray_errors[r.camera_id]

    coordinates = np.array(coordinates)
    return coordinates, error