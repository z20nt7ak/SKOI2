import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEOS, RESULTS = ROOT / "task/test-videos", ROOT / "results"
DEFAULT_VIDEOS = ["mona-lisa.mp4", "mona-lisa-blur.mp4", "mona-lisa-blur-extra-credit.mp4", "IMG_8805.mp4"]
BOOK_CORNERS = [210, 286, 940, 286, 928, 1446, 173, 1428]
MIN_POINTS = 10
RATIO= 0.75 
MAX_SIDE = 1280

class Tracker:
    def __init__(self, frame, corners):
        self.corners = np.float32(corners).reshape(4, 2) 
        h, w = frame.shape[:2]
        if (not np.isfinite(self.corners).all() or not cv2.isContourConvex(self.corners)
                or cv2.contourArea(self.corners) < 100 or (self.corners < 0).any()
                or (self.corners[:, 0] >= w).any() or (self.corners[:, 1] >= h).any()):
            raise ValueError("Укажите четыре угла объекта по периметру внутри первого кадра")
        self.sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(self.corners), 255)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(gray, mask)
        if self.descriptors is None or len(self.keypoints) < MIN_POINTS:
            raise ValueError("На выбранном объекте слишком мало ключевых точек")
        self.previous_gray = self.homography = self.points = None

    def estimate(self, source, target, shape):
        if len(source) < MIN_POINTS:
            return None, 0, 0
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None, 0, 0
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 8 or count / len(source) < 0.4:
            return None, 0, 0
        coverage = cv2.contourArea(cv2.convexHull(source[inliers])) / cv2.contourArea(self.corners)
        projected = np.column_stack([self.corners, np.ones(4)]) @ matrix.T
        z = projected[:, 2]
        if coverage < 0.025 or not ((z > 1e-6).all() or (z < -1e-6).all()):
            return None, 0, 0
        quad = (projected[:, :2] / z[:, None]).astype(np.float32)
        h, w = shape
        area = cv2.contourArea(quad)
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or not 100 < area < 4*h*w
                or sides.min() < 5 or (np.abs(quad) > 4*max(h, w)).any()):
            return None, 0, 0
        screen = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        if cv2.intersectConvexConvex(quad, screen)[0] < 0.1*area:
            return None, 0, 0
        return matrix, count, coverage

    def detect(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0, 0
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO*p[1].distance]
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        source = np.float32([self.keypoints[m.queryIdx].pt for m in unique.values()])
        target = np.float32([keypoints[m.trainIdx].pt for m in unique.values()])
        return self.estimate(source, target, gray.shape)

    def follow(self, gray):
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.points, None, **params)
        if forward is None or status is None or error is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, forward, None, **params)
        if backward is None or back_status is None:
            return None
        distance = np.linalg.norm(backward - self.points, axis=2).ravel()
        keep = (status.ravel() == 1) & (back_status.ravel() == 1) & (distance < 1) & (error.ravel() < 20)
        h, w = gray.shape
        points = forward.reshape(-1, 2)
        keep &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        if keep.sum() < MIN_POINTS:
            return None
        try:
            source = cv2.perspectiveTransform(self.points[keep], np.linalg.inv(self.homography)).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return None
        matrix, _, _ = self.estimate(source, points[keep], gray.shape)
        if matrix is not None:
            old = cv2.perspectiveTransform(self.corners[:, None], self.homography)
            new = cv2.perspectiveTransform(self.corners[:, None], matrix)
            if not 0.5 < cv2.contourArea(new) / cv2.contourArea(old) < 2:
                return None
        return matrix

    def locate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix, inliers, coverage = self.detect(gray)
        mode = "detect"
        if (matrix is None or inliers < 20 or coverage < 0.1) and self.points is not None:
            flow = self.follow(gray)
            if flow is not None:
                matrix, mode = flow, "flow"
        self.homography, self.previous_gray, self.points = matrix, gray, None
        if matrix is None:
            return None, "lost"
        quad = cv2.perspectiveTransform(self.corners[:, None], matrix)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(np.round(quad.reshape(4, 2))), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        self.points = cv2.goodFeaturesToTrack(gray, 200, 0.01, 3, mask=mask, blockSize=3)
        return quad, mode


def draw(frame, quad, label, mode):
    color = (40, 230, 70) if mode == "detect" else (0, 210, 255)
    cv2.polylines(frame, [np.int32(np.round(quad))], True, color, 2, cv2.LINE_AA)
    x, y = quad.reshape(4, 2)[np.argmin(quad[:, 0, 1])]
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    position = (int(np.clip(x, 5, max(5, frame.shape[1]-text_width-5))), int(np.clip(y-8, 25, frame.shape[0]-5)))
    for thickness, text_color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness, cv2.LINE_AA)


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
        h, w = frame.shape[:2]
        scale = min(1, MAX_SIDE/max(h, w))
        size = (round(w*scale/2)*2, round(h*scale/2)*2)
        source_corners = np.float32(corners).reshape(4, 2) if corners else np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        source_corners *= np.float32([(size[0]-1)/(w-1), (size[1]-1)/(h-1)])
        tracker = Tracker(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), source_corners)
        RESULTS.mkdir(exist_ok=True)
        output = RESULTS / f"{path.stem}_tracked.mp4"
        if output.resolve() == path.resolve():
            raise ValueError("Выходной файл совпадает с входным")
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                break
            writer.release()
        if not writer.isOpened():
            raise ValueError("Не удалось создать выходной MP4")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # В тестах размытие кратковременно: равномерная выборка его пропускает.
        sample_indices = {
            "mona-lisa-blur.mp4": {9, 28, 29},
            "mona-lisa-blur-extra-credit.mp4": {10, 29, 30, 390, 399, 400},
        }.get(path.name, set(np.linspace(0, max(0, total-12), 6).astype(int)))
        counts, samples, index = {"detect": 0, "flow": 0, "lost": 0}, [], 0
        started = time.perf_counter()
        while ok:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            quad, mode = tracker.locate(frame)
            if quad is not None:
                draw(frame, quad, label, mode)
            counts[mode] += 1
            writer.write(frame)
            if index in sample_indices:
                ratio = min(400/size[0], 320/size[1])
                thumb = cv2.resize(frame, (round(size[0]*ratio), round(size[1]*ratio)))
                for thickness, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
                    cv2.putText(thumb, f"#{index} {index/fps:.2f}s {mode}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                samples.append(thumb)
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
    if samples:
        while len(samples) % 3:
            samples.append(np.zeros_like(samples[0]))
        cv2.imwrite(str(RESULTS / f"{path.stem}.jpg"), np.vstack([np.hstack(samples[i:i+3]) for i in range(0, len(samples), 3)]))
    found = index - counts["lost"]
    print(f"{path.name}: {found}/{index} кадров ({found/index:.1%}), {index/elapsed:.1f} кадр/с -> {output}", flush=True)
    return dict(video=path.name, frames=index, counts=counts, found_percent=round(100*found/index, 2),
                processing_fps=round(index/elapsed, 2), video_fps=fps, size=size, seconds=round(elapsed, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path, help="Без аргументов: три теста и своё видео")
    parser.add_argument("--label", help="Подпись латиницей")
    parser.add_argument("--corners", type=float, nargs=8, help="x1 y1 ... x4 y4 в исходном первом кадре, по периметру")
    parser.add_argument("--show", action="store_true", help="Показать обработку; Q для выхода")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    stats = {}
    for path in args.videos or [VIDEOS / name for name in DEFAULT_VIDEOS]:
        cv2.setRNGSeed(42)
        own = path.name == "IMG_8805.mp4"
        label = args.label or ("Aristotle" if own else "Mona Lisa")
        corners = args.corners or (BOOK_CORNERS if own else None)
        try:
            stats[path.stem] = process(path, label, corners, args.show)
        except (ValueError, OSError, cv2.error) as error:
            parser.exit(1, f"Ошибка: {error}\n")
    stats_path = RESULTS / "stats.json"
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats_path.write_text(json.dumps(previous | stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
