import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np


def load_gray(path):
    """
    Safely load an image from either a string or pathlib.Path.
    """
    path = str(path)

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    return image.astype(np.float32) / 255.0


def edge_image(image):
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    return cv2.magnitude(gx, gy)


def rotate_image(image, angle, scale=1.0):
    h, w = image.shape

    center = (
        (w - 1) / 2.0,
        (h - 1) / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def find_peaks(
    score,
    minimum_distance=25,
    max_candidates=20
):
    """
    Extract several spatially separated strong candidates.
    """

    work = score.copy()
    candidates = []

    for _ in range(max_candidates):

        _, maximum, _, location = cv2.minMaxLoc(
            work
        )

        if maximum <= -1:
            break

        x, y = location

        candidates.append(
            (
                float(maximum),
                int(x),
                int(y)
            )
        )

        x0 = max(
            0,
            x - minimum_distance
        )

        x1 = min(
            work.shape[1],
            x + minimum_distance + 1
        )

        y0 = max(
            0,
            y - minimum_distance
        )

        y1 = min(
            work.shape[0],
            y + minimum_distance + 1
        )

        work[
            y0:y1,
            x0:x1
        ] = -1

    return candidates


def subpixel_peak(score, x, y):

    if (
        x <= 0
        or y <= 0
        or x >= score.shape[1] - 1
        or y >= score.shape[0] - 1
    ):
        return float(x), float(y)

    def parabolic_offset(a, b, c):

        denominator = (
            a - 2.0 * b + c
        )

        if abs(denominator) < 1e-8:
            return 0.0

        return 0.5 * (
            a - c
        ) / denominator

    dx = parabolic_offset(
        score[y, x - 1],
        score[y, x],
        score[y, x + 1]
    )

    dy = parabolic_offset(
        score[y - 1, x],
        score[y, x],
        score[y + 1, x]
    )

    dx = float(
        np.clip(dx, -0.5, 0.5)
    )

    dy = float(
        np.clip(dy, -0.5, 0.5)
    )

    return (
        x + dx,
        y + dy
    )


def match(
    reference_path,
    search_path
):
    """
    Main localization function.

    Input:
        reference_path -> high magnification reference image
        search_path    -> 1000x1000 search image

    Output:
        dictionary containing predicted center coordinates.
    """

    reference = load_gray(
        reference_path
    )

    search = load_gray(
        search_path
    )

    # ---------------------------------------------------------
    # Reference is approximately 10x larger than its appearance
    # in the low-magnification search image.
    # ---------------------------------------------------------

    reference_small = cv2.resize(
        reference,
        (100, 100),
        interpolation=cv2.INTER_AREA
    )

    reference_edge = edge_image(
        reference_small
    )

    search_edge = edge_image(
        search
    )

    candidates = []

    # Small scale range around expected 100x100 target.
    scales = [
        0.94,
        0.96,
        0.98,
        1.00,
        1.02,
        1.04,
        1.06
    ]

    # Generator currently uses approximately +/- 6 degrees.
    angles = [
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6
    ]

    for scale in scales:

        size = int(
            round(
                100 * scale
            )
        )

        if size < 70:
            continue

        template = cv2.resize(
            reference_small,
            (size, size),
            interpolation=cv2.INTER_LINEAR
        )

        template_edge = edge_image(
            template
        )

        for angle in angles:

            rotated = rotate_image(
                template,
                angle,
                1.0
            )

            rotated_edge = rotate_image(
                template_edge,
                angle,
                1.0
            )

            # -------------------------------------------------
            # Intensity similarity
            # -------------------------------------------------

            intensity = cv2.matchTemplate(
                search,
                rotated,
                cv2.TM_CCOEFF_NORMED
            )

            # -------------------------------------------------
            # Edge similarity
            # -------------------------------------------------

            edge_score = cv2.matchTemplate(
                search_edge,
                rotated_edge,
                cv2.TM_CCOEFF_NORMED
            )

            # Combined score.
            combined = (
                0.70 * intensity
                + 0.30 * edge_score
            )

            peaks = find_peaks(
                combined,
                minimum_distance=25,
                max_candidates=10
            )

            for value, px, py in peaks:

                center_x = (
                    px + size / 2.0
                )

                center_y = (
                    py + size / 2.0
                )

                candidates.append({
                    "score": float(value),
                    "x": float(center_x),
                    "y": float(center_y),
                    "size": int(size),
                    "scale": float(scale),
                    "angle": float(angle),
                    "score_map": combined,
                    "peak_x": int(px),
                    "peak_y": int(py)
                })

    if not candidates:
        raise RuntimeError(
            "No matching candidate was found."
        )

    # Highest scoring candidate.
    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    best_score = candidates[0]["score"]

    # ---------------------------------------------------------
    # Periodic-layout tie handling.
    #
    # If candidates have nearly identical scores, use the
    # official challenge rule:
    #
    #     choose the one closest to search-image center.
    # ---------------------------------------------------------

    tie_threshold = 0.025

    tied = [
        candidate
        for candidate in candidates
        if candidate["score"]
        >= best_score - tie_threshold
    ]

    search_center_x = (
        search.shape[1] - 1
    ) / 2.0

    search_center_y = (
        search.shape[0] - 1
    ) / 2.0

    best = min(
        tied,
        key=lambda candidate:
        math.hypot(
            candidate["x"]
            - search_center_x,

            candidate["y"]
            - search_center_y
        )
    )

    # ---------------------------------------------------------
    # Subpixel refinement.
    # ---------------------------------------------------------

    refined_x, refined_y = subpixel_peak(
        best["score_map"],
        best["peak_x"],
        best["peak_y"]
    )

    predicted_x = (
        refined_x
        + best["size"] / 2.0
    )

    predicted_y = (
        refined_y
        + best["size"] / 2.0
    )

    return {
        "x": float(predicted_x),
        "y": float(predicted_y),
        "score": float(best["score"]),
        "scale": float(best["scale"]),
        "angle": float(best["angle"])
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Drift-Sense navigation-error recovery"
        )
    )

    parser.add_argument(
        "reference",
        help="Path to reference image"
    )

    parser.add_argument(
        "search",
        help="Path to search image"
    )

    parser.add_argument(
        "--details",
        action="store_true"
    )

    args = parser.parse_args()

    start = time.perf_counter()

    result = match(
        args.reference,
        args.search
    )

    runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    if args.details:

        print(
            f"x={result['x']:.2f} "
            f"y={result['y']:.2f} "
            f"score={result['score']:.4f} "
            f"scale={result['scale']:.2f} "
            f"angle={result['angle']:.1f} "
            f"runtime_ms={runtime_ms:.1f}"
        )

    else:

        # Official inference output.
        print(
            f"{result['x']:.2f} "
            f"{result['y']:.2f}"
        )


if __name__ == "__main__":
    main()