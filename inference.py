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


def detect_fiducial(
    reference,
    threshold=0.55,
    erode_ksize=9,
    min_area=30,
    border_margin=40
):
    """
    Locate the unique alignment fiducial embedded in the reference image.

    The dataset generator draws periodic lines up to ~5.5px wide and via
    dots up to ~6.4px across, then stamps a single much thicker (14-20px)
    cross-shaped fiducial somewhere in the frame specifically so the true
    match location is distinguishable from other, visually-identical
    occurrences of the periodic pattern. Eroding a brightness mask with a
    kernel wider than any periodic feature but narrower than the fiducial
    strips away the repeating structure and leaves only the fiducial,
    which is then located by its connected-component centroid.

    Border-adjacent components are skipped: the small rotation applied
    to the reference during generation uses reflective borders, which
    can occasionally leave a large bright artifact right at the edge
    that would otherwise survive erosion. The generator never places
    a real fiducial within `border_margin` of the edge, so any
    component that close is treated as a reflection artifact, not a
    fiducial.

    Returns (x, y) in reference-image pixel coordinates, or None if no
    fiducial-sized blob survives erosion (e.g. a reference image that
    was not generated with this convention).
    """

    mask = (reference > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erode_ksize, erode_ksize)
    )

    eroded = cv2.erode(mask, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded,
        connectivity=8
    )

    if count <= 1:
        return None

    h, w = reference.shape

    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    for label in order:

        if stats[label, cv2.CC_STAT_AREA] < min_area:
            break

        cx, cy = centroids[label]

        if (
            cx < border_margin
            or cy < border_margin
            or cx > w - border_margin
            or cy > h - border_margin
        ):
            continue

        return float(cx), float(cy)

    return None


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

    # ---------------------------------------------------------
    # Fiducial-aware disambiguation.
    #
    # The periodic background alone repeats identically at every
    # occurrence, so it cannot distinguish the true match from other
    # tiles. If a fiducial is present in the reference, crop it (from
    # the full-resolution reference, before detail is lost to the 10x
    # downsample below) so it can later be matched against candidate
    # locations to break periodic ties on real evidence.
    # ---------------------------------------------------------

    fiducial_info = None

    fiducial_center = detect_fiducial(reference)

    if fiducial_center is not None:

        fx, fy = fiducial_center
        margin = 100

        rx0 = max(0, int(round(fx)) - margin)
        rx1 = min(reference.shape[1], int(round(fx)) + margin)
        ry0 = max(0, int(round(fy)) - margin)
        ry1 = min(reference.shape[0], int(round(fy)) + margin)

        fiducial_crop = reference[ry0:ry1, rx0:rx1]

        if fiducial_crop.size > 0:
            fiducial_info = {
                "crop": fiducial_crop,
                "fx": fx,
                "fy": fy,
                "ref_h": reference.shape[0],
                "ref_w": reference.shape[1]
            }

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

    # Generator uses +/- 8 degrees (generate_dataset.py: rng.uniform(-8, 8)).
    angles = [
        -8,
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6,
        8
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

        # -------------------------------------------------
        # Scale the fiducial crop and its anchor marker to
        # match this candidate template's scale. The template
        # is effectively reference resized to `size`, so the
        # fiducial (cropped from the full-resolution reference)
        # is resized by the same size/reference_width factor.
        # -------------------------------------------------

        scaled_fiducial = None

        if fiducial_info is not None:

            fid_scale = size / float(fiducial_info["ref_w"])
            crop_h, crop_w = fiducial_info["crop"].shape
            fid_dim = int(round(max(crop_w, crop_h) * fid_scale))

            if 6 <= fid_dim < size:

                fid_template = cv2.resize(
                    fiducial_info["crop"],
                    (fid_dim, fid_dim),
                    interpolation=cv2.INTER_AREA
                )

                marker = np.zeros((size, size), np.float32)
                mx = int(np.clip(round(fiducial_info["fx"] * fid_scale), 0, size - 1))
                my = int(np.clip(round(fiducial_info["fy"] * fid_scale), 0, size - 1))
                marker[my, mx] = 1.0

                scaled_fiducial = {
                    "template": fid_template,
                    "dim": fid_dim,
                    "marker": marker
                }

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
            # Fiducial score map + expected in-template anchor
            # position, rotated with the exact same warpAffine
            # call used for the template itself so the anchor
            # stays consistent with the rotation actually applied.
            # -------------------------------------------------

            fiducial_score_map = None
            anchor_x = anchor_y = None
            fid_dim = 0
            fiducial_peaks = []

            if scaled_fiducial is not None:

                rotated_fid = rotate_image(
                    scaled_fiducial["template"],
                    angle,
                    1.0
                )

                rotated_marker = rotate_image(
                    scaled_fiducial["marker"],
                    angle,
                    1.0
                )

                fid_dim = scaled_fiducial["dim"]

                if fid_dim < search.shape[0] and fid_dim < search.shape[1]:

                    fiducial_score_map = cv2.matchTemplate(
                        search,
                        rotated_fid,
                        cv2.TM_CCOEFF_NORMED
                    )

                    anchor_y, anchor_x = np.unravel_index(
                        np.argmax(rotated_marker),
                        rotated_marker.shape
                    )

                    # The fiducial is far more discriminative than the
                    # periodic background, but the whole-tile peaks
                    # found below (max 10 per scale/angle) are so
                    # crowded by periodic sidelobes that the true tile
                    # is frequently not among them at all. Search the
                    # fiducial's own score map for its strongest,
                    # spatially-distinct matches directly, so the true
                    # location gets a chance to enter the candidate
                    # pool even when the whole-tile score alone
                    # wouldn't have surfaced it.
                    fiducial_peaks = find_peaks(
                        fiducial_score_map,
                        minimum_distance=max(10, fid_dim),
                        max_candidates=5
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

                fid_score = None
                fid_peak_x = fid_peak_y = None

                if fiducial_score_map is not None:

                    # Expected fiducial location for this candidate:
                    # candidate's top-left plus the anchor position
                    # found within the identically-rotated template.
                    expected_x = px + anchor_x - fid_dim / 2.0
                    expected_y = py + anchor_y - fid_dim / 2.0

                    ix = int(round(expected_x))
                    iy = int(round(expected_y))

                    win = 3
                    x0 = max(0, ix - win)
                    x1 = min(fiducial_score_map.shape[1], ix + win + 1)
                    y0 = max(0, iy - win)
                    y1 = min(fiducial_score_map.shape[0], iy + win + 1)

                    if x1 > x0 and y1 > y0:
                        window = fiducial_score_map[y0:y1, x0:x1]
                        wy, wx = np.unravel_index(np.argmax(window), window.shape)
                        fid_score = float(window[wy, wx])
                        fid_peak_x = x0 + wx
                        fid_peak_y = y0 + wy

                candidates.append({
                    "score": float(value),
                    "x": float(center_x),
                    "y": float(center_y),
                    "size": int(size),
                    "scale": float(scale),
                    "angle": float(angle),
                    "score_map": combined,
                    "peak_x": int(px),
                    "peak_y": int(py),
                    "fid_score": fid_score,
                    "fid_score_map": fiducial_score_map,
                    "fid_peak_x": fid_peak_x,
                    "fid_peak_y": fid_peak_y,
                    "fid_dim": fid_dim
                })

            # -------------------------------------------------
            # Candidates proposed directly from the fiducial's own
            # peaks, back-projected to a whole-tile position so they
            # can be scored/refined with the same machinery as any
            # other candidate.
            # -------------------------------------------------

            for fvalue, fpx, fpy in fiducial_peaks:

                bpx = int(round(fpx - anchor_x + fid_dim / 2.0))
                bpy = int(round(fpy - anchor_y + fid_dim / 2.0))

                bpx = int(np.clip(bpx, 0, combined.shape[1] - 1))
                bpy = int(np.clip(bpy, 0, combined.shape[0] - 1))

                candidates.append({
                    "score": float(combined[bpy, bpx]),
                    "x": float(bpx + size / 2.0),
                    "y": float(bpy + size / 2.0),
                    "size": int(size),
                    "scale": float(scale),
                    "angle": float(angle),
                    "score_map": combined,
                    "peak_x": bpx,
                    "peak_y": bpy,
                    "fid_score": float(fvalue),
                    "fid_score_map": fiducial_score_map,
                    "fid_peak_x": int(fpx),
                    "fid_peak_y": int(fpy),
                    "fid_dim": fid_dim
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
    # The periodic background alone scores nearly identically at
    # every occurrence -- including at candidates proposed by the
    # fiducial's own peaks above, whose whole-tile score can be far
    # from the global best even when they mark the true location.
    # So: a candidate with strong, spatially-specific fiducial
    # correlation is trusted over the periodic whole-tile score,
    # gated only by a loose plausibility check (reject scale/angle
    # combinations whose background match is clearly wrong). If no
    # candidate has confident fiducial evidence -- e.g. this
    # reference has no detectable fiducial -- fall back exactly to
    # the original closest-to-search-center heuristic among
    # narrowly tied whole-tile scores.
    # ---------------------------------------------------------

    fiducial_confidence = 0.5

    # A sane absolute floor on the whole-tile score, not a margin
    # relative to best_score: best_score is often held by some other
    # tile whose periodic background happens to correlate strongly by
    # chance, which is exactly the ambiguity fiducial evidence exists
    # to overrule. This floor only rejects scale/angle combinations
    # whose background match is closer to noise than a real match.
    plausible_score_floor = 0.10

    fiducial_candidates = [
        candidate
        for candidate in candidates
        if candidate["fid_score"] is not None
        and candidate["fid_score"] >= fiducial_confidence
        and candidate["score"] >= plausible_score_floor
    ]

    selected_via_fiducial = bool(fiducial_candidates)

    if fiducial_candidates:

        best = max(
            fiducial_candidates,
            key=lambda candidate: candidate["fid_score"]
        )

    else:

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
    #
    # Ground truth is the fiducial's own position, which is not
    # necessarily the matched tile's geometric center -- the dataset
    # generator places it anywhere within the tile. So when a
    # candidate was trusted specifically because of its fiducial
    # evidence, refine and report the fiducial's own matched position
    # rather than the whole-tile center; otherwise this is identical
    # to the original whole-tile refinement.
    # ---------------------------------------------------------

    if (
        selected_via_fiducial
        and best["fid_peak_x"] is not None
        and best["fid_peak_y"] is not None
    ):

        refined_x, refined_y = subpixel_peak(
            best["fid_score_map"],
            best["fid_peak_x"],
            best["fid_peak_y"]
        )

        predicted_x = (
            refined_x
            + best["fid_dim"] / 2.0
        )

        predicted_y = (
            refined_y
            + best["fid_dim"] / 2.0
        )

    else:

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