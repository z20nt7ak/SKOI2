"""Трекинг плоского объекта: SIFT для поиска, оптический поток для сопровождения."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "task/test-videos"
RESULTS = ROOT / "results"
DEFAULT_VIDEOS = [
    "mona-lisa.mp4", "mona-lisa-blur.mp4",
    "mona-lisa-blur-extra-credit.mp4", "IMG_8805.mp4",
]
BOOK_CORNERS = [210, 286, 940, 286, 928, 1446, 173, 1428]
MIN_POINTS = 10
RATIO = 0.75
MAX_SIDE = 1280


def prepare_template(frame, corners):
    height, width = frame.shape[:2]
    scale = min(1, MAX_SIDE / max(height, width))
    size = (round(width * scale / 2) * 2, round(height * scale / 2) * 2)

    if corners is None:
        corners = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]
    corners = np.float32(corners).reshape(4, 2)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        raise ValueError("Укажите четыре угла объекта последовательно по периметру")
    for x, y in corners:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Углы объекта должны находиться внутри первого кадра")

    corners[:, 0] *= (size[0]-1) / (width-1)
    corners[:, 1] *= (size[1]-1) / (height-1)
    if cv2.contourArea(corners) < 100:
        raise ValueError("Выбранная область слишком маленькая")
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame, corners, size


def is_valid_box(box, shape):
    height, width = shape
    if not np.isfinite(box).all() or not cv2.isContourConvex(box):
        return False
    area = cv2.contourArea(box)
    if area <= 100 or area >= 4 * height * width:
        return False
    for i in range(4):
        side = box[(i + 1) % 4] - box[i]
        if np.linalg.norm(side) < 5:
            return False
    if np.max(np.abs(box)) > 4 * max(height, width):
        return False
    screen = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    visible_area, _ = cv2.intersectConvexConvex(box, screen)
    return visible_area >= 0.1 * area


class Tracker:
    def __init__(self, frame, corners):
        self.corners = np.float32(corners).reshape(4, 2)
        self.sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(self.corners), 255)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(gray, mask)
        if self.descriptors is None or len(self.keypoints) < MIN_POINTS:
            raise ValueError("На выбранном объекте слишком мало ключевых точек")

        self.previous_gray = None
        self.homography = None
        self.points = None

    def locate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix, inliers, coverage = self.detect(gray)
        mode = "detect"

        weak_detection = matrix is None or inliers < 20 or coverage < 0.1
        if weak_detection and self.points is not None:
            flow_matrix = self.follow(gray)
            if flow_matrix is not None:
                matrix = flow_matrix
                mode = "flow"

        self.previous_gray = gray
        self.homography = matrix
        self.points = None
        if matrix is None:
            return None, "lost"

        box = cv2.perspectiveTransform(self.corners.reshape(4, 1, 2), matrix)
        self.select_points(gray, box)
        return box, mode

    def detect(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0, 0

        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = []
        for pair in pairs:
            if len(pair) == 2:
                best, second = pair
                if best.distance < RATIO * second.distance:
                    good.append(best)
        good.sort(key=lambda match: match.distance)
        used_points = set()
        source, target = [], []
        for match in good:
            if match.trainIdx in used_points:
                continue
            used_points.add(match.trainIdx)
            source.append(self.keypoints[match.queryIdx].pt)
            target.append(keypoints[match.trainIdx].pt)
        return self.estimate(np.float32(source), np.float32(target), gray.shape)

    def estimate(self, source, target, shape):
        if len(source) < MIN_POINTS:
            return None, 0, 0
        matrix, mask = cv2.findHomography(
            source, target, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995
        )
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None, 0, 0
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 8 or count / len(source) < 0.4:
            return None, 0, 0

        hull = cv2.convexHull(source[inliers])
        coverage = cv2.contourArea(hull) / cv2.contourArea(self.corners)
        if coverage < 0.025:
            return None, 0, 0

        corners_3d = np.column_stack([self.corners, np.ones(4)])
        projected = corners_3d @ matrix.T
        z = projected[:, 2]
        same_sign = (z > 1e-6).all() or (z < -1e-6).all()
        if not same_sign:
            return None, 0, 0
        box = (projected[:, :2] / z[:, None]).astype(np.float32)
        if not is_valid_box(box, shape):
            return None, 0, 0
        return matrix, count, coverage

    def follow(self, gray):
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.points, None, **params
        )
        if forward is None or status is None or error is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, forward, None, **params
        )
        if backward is None or back_status is None:
            return None

        distance = np.linalg.norm(backward - self.points, axis=2).ravel()
        height, width = gray.shape
        keep = []
        for i in range(len(self.points)):
            if status[i, 0] != 1 or back_status[i, 0] != 1:
                continue
            if not (distance[i] < 1 and error[i, 0] < 20):
                continue
            x, y = forward[i, 0]
            if 0 <= x < width and 0 <= y < height:
                keep.append(i)
        if len(keep) < MIN_POINTS:
            return None
        try:
            inverse = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return None
        source = cv2.perspectiveTransform(self.points[keep], inverse).reshape(-1, 2)
        target = forward[keep].reshape(-1, 2)
        matrix, _, _ = self.estimate(source, target, gray.shape)
        if matrix is None:
            return None
        corners = self.corners.reshape(4, 1, 2)
        old_box = cv2.perspectiveTransform(corners, self.homography)
        new_box = cv2.perspectiveTransform(corners, matrix)
        area_ratio = cv2.contourArea(new_box) / cv2.contourArea(old_box)
        if not 0.5 < area_ratio < 2:
            return None
        return matrix

    def select_points(self, gray, box):
        mask = np.zeros_like(gray)
        polygon = np.int32(np.round(box.reshape(4, 2)))
        cv2.fillConvexPoly(mask, polygon, 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        self.points = cv2.goodFeaturesToTrack(gray, 200, 0.01, 3, mask=mask, blockSize=3)


def draw(frame, box, label, mode, points=None):
    if mode == "detect":
        color = (40, 230, 70)
    else:
        color = (0, 210, 255)
    if points is not None:
        for x, y in points.reshape(-1, 2):
            center = (round(float(x)), round(float(y)))
            cv2.circle(frame, center, 2, (40, 230, 70), -1, cv2.LINE_AA)
    cv2.polylines(frame, [np.int32(np.round(box))], True, color, 2, cv2.LINE_AA)
    corners = box.reshape(4, 2)
    top_corner = np.argmin(corners[:, 1])
    x, y = corners[top_corner]
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    x = int(np.clip(x, 5, max(5, frame.shape[1] - text_width - 5)))
    y = int(np.clip(y - 8, 25, frame.shape[0] - 5))
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def create_writer(path, fps, size):
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{path.stem}_tracked.mp4"
    if output.resolve() == path.resolve():
        raise ValueError("Выходной файл совпадает с входным")
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, output
        writer.release()
    raise ValueError("Не удалось создать выходной MP4")


def choose_samples(path, total):
    if path.name == "mona-lisa-blur.mp4":
        return {9, 28, 29}
    if path.name == "mona-lisa-blur-extra-credit.mp4":
        return {10, 29, 30, 390, 399, 400}
    return set(np.linspace(0, max(0, total - 12), 6).astype(int))


def make_sample(frame, index, fps, mode):
    height, width = frame.shape[:2]
    scale = min(400 / width, 320 / height)
    image = cv2.resize(frame, (round(width * scale), round(height * scale)))
    text = f"#{index} {index/fps:.2f}s {mode}"
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def save_samples(path, samples):
    if not samples:
        return
    while len(samples) % 3 != 0:
        samples.append(np.zeros_like(samples[0]))
    rows = []
    for i in range(0, len(samples), 3):
        rows.append(np.hstack(samples[i:i+3]))
    cv2.imwrite(str(RESULTS / f"{path.stem}.jpg"), np.vstack(rows))


def process(path, label, corners, show):
    capture = cv2.VideoCapture(str(path))
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Не удалось прочитать {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Некорректная частота кадров видео")
        first_frame, corners, size = prepare_template(frame, corners)
        tracker = Tracker(first_frame, corners)
        writer, output = create_writer(path, fps, size)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = choose_samples(path, total)
        counts = {"detect": 0, "flow": 0, "lost": 0}
        samples = []
        index = 0
        started = time.perf_counter()
        while ok:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            box, mode = tracker.locate(frame)
            if box is not None:
                draw(frame, box, label, mode, tracker.points)
            writer.write(frame)
            counts[mode] += 1
            if index in sample_indices:
                samples.append(make_sample(frame, index, fps, mode))
            index += 1
            if show:
                cv2.imshow(path.name, frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
            ok, frame = capture.read()
        elapsed = time.perf_counter() - started
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    save_samples(path, samples)
    found = index - counts["lost"]
    print(f"{path.name}: {found}/{index} кадров ({found/index:.1%}), {index/elapsed:.1f} кадр/с -> {output}", flush=True)
    return {
        "video": path.name, "frames": index, "counts": counts,
        "found_percent": round(100 * found / index, 2),
        "processing_fps": round(index / elapsed, 2),
        "video_fps": fps, "size": size, "seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path, help="Без аргументов: три теста и своё видео")
    parser.add_argument("--label", help="Подпись латиницей")
    parser.add_argument("--corners", type=float, nargs=8, help="x1 y1 ... x4 y4 в первом кадре, по периметру")
    parser.add_argument("--show", action="store_true", help="Показать обработку; Q для выхода")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    videos = args.videos
    if not videos:
        videos = [VIDEOS / name for name in DEFAULT_VIDEOS]
    stats = {}
    for path in videos:
        cv2.setRNGSeed(42)
        if path.name == "IMG_8805.mp4":
            label = "Aristotle"
            corners = BOOK_CORNERS
        else:
            label = "Mona Lisa"
            corners = None
        if args.label:
            label = args.label
        if args.corners is not None:
            corners = args.corners
        try:
            stats[path.stem] = process(path, label, corners, args.show)
        except (ValueError, OSError, cv2.error) as error:
            parser.exit(1, f"Ошибка: {error}\n")

    stats_path = RESULTS / "stats.json"
    previous = {}
    if stats_path.exists():
        previous = json.loads(stats_path.read_text())
    previous.update(stats)
    stats_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
