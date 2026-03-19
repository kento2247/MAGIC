import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw


class PingToSVG:
    DEFAULT_SAM_ID = "facebook/sam-vit-base"
    OCR_CONF_THRESH = 0.35
    CROP_PADDING = 6

    # SAM filtering
    SAM_MIN_AREA_RATIO = 0.003
    SAM_MAX_AREA_RATIO = 0.60
    SAM_MAX_ASPECT = 6.0
    SAM_IOU_DEDUP = 0.50
    SAM_MAX_COMPONENTS = 80

    # OCR/SAM merge
    TEXT_IN_BOX_THRESH = 0.60

    # Arrow detection
    ARROW_S_MAX = 50
    ARROW_V_MIN = 40
    ARROW_V_MAX = 130
    ARROW_MIN_AREA = 0
    ARROW_MAX_AREA = 5000
    ARROW_ERODE_PX = 2
    ARROW_SOLIDITY_MIN = 0.1

    def __init__(
        self,
        sam_model_id: str = DEFAULT_SAM_ID,
        device: str | None = None,
        use_sam: bool = True,
        use_arrows: bool = True,
        use_ocr_overlay: bool = True,
        erase_text_before_sam: bool = True,
        ocr_conf_thresh: float = OCR_CONF_THRESH,
        show_outline: bool = False,
        debug: bool = False,
    ):
        self.sam_model_id = sam_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sam = use_sam
        self.use_arrows = use_arrows
        self.use_ocr_overlay = use_ocr_overlay
        self.erase_text_before_sam = erase_text_before_sam
        self.ocr_conf_thresh = ocr_conf_thresh
        self.show_outline = show_outline
        self.debug = debug

    def convert(self, image: Image.Image) -> str:
        """
        Convert PIL Image (PNG diagram) to SVG string.

        Returns:
            SVG string
        """
        pipeline = self._run_pipeline(image)

        # Build SVG
        svg = self._build_svg(
            image=pipeline["image_after_arrow_erase"],
            sam_components=pipeline["merged_components"],
            arrows=pipeline["arrows"],
            all_texts=pipeline["texts"],
            arrow_image=pipeline["image_before_arrow_erase"],
            ocr_overlay=self.use_ocr_overlay,
            show_outline=self.show_outline,
        )
        return svg

    def convert_with_trace(
        self,
        image: Image.Image,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        pipeline = self._run_pipeline(image, allow_partial=allow_partial)
        svg = self._build_svg(
            image=pipeline["image_after_arrow_erase"],
            sam_components=pipeline["merged_components"],
            arrows=pipeline["arrows"],
            all_texts=pipeline["texts"],
            arrow_image=pipeline["image_before_arrow_erase"],
            ocr_overlay=self.use_ocr_overlay,
            show_outline=self.show_outline,
        )
        width, height = pipeline["original_image"].size
        warnings = pipeline["warnings"]

        stages = [
            {
                "key": "ocr",
                "label": "OCR: 文字候補を走査しています。",
                "metric": warnings["ocr"] or f"{len(pipeline['texts'])} text regions",
                "image": self._render_ocr_overlay((width, height), pipeline["texts"]),
                "texts": self._copy_texts(pipeline["texts"]),
            },
            {
                "key": "erase_text",
                "label": "Erase text: 検出した文字領域を消去しています。",
                "metric": (
                    "skipped: OCR unavailable"
                    if warnings["ocr"]
                    else f"{len(pipeline['texts'])} regions removed"
                ),
                "image": pipeline["image_after_text_erase"],
                "texts": self._copy_texts(pipeline["texts"]),
            },
            {
                "key": "detect_arrows",
                "label": "Detect arrows: 矢印の候補を検出しています。",
                "metric": warnings["arrows"] or f"{len(pipeline['arrows'])} arrows",
                "image": self._render_arrow_overlay(
                    (width, height), pipeline["arrows"]
                ),
                "arrows": self._copy_arrows(pipeline["arrows"]),
            },
            {
                "key": "erase_arrows",
                "label": "Erase arrows before SAM: SAM 前処理として矢印を消去しています。",
                "metric": (
                    "skipped: arrow detection unavailable"
                    if warnings["arrows"]
                    else f"{len(pipeline['arrows'])} arrows removed"
                ),
                "image": pipeline["image_after_arrow_erase"],
                "arrows": self._copy_arrows(pipeline["arrows"]),
            },
            {
                "key": "sam",
                "label": "SAM: 領域マスクを推定しています。",
                "metric": warnings["sam"]
                or f"{len(pipeline['sam_components'])} components",
                "image": self._render_sam_overlay(
                    (width, height), pipeline["sam_components"]
                ),
                "components": self._copy_components(pipeline["sam_components"]),
            },
            {
                "key": "merge",
                "label": "Merge: 各結果を統合して SVG export を組み立てています。",
                "metric": (
                    f"{len(pipeline['merged_components'])} blocks / "
                    f"{len(pipeline['texts'])} texts / "
                    f"{len(pipeline['orphan_texts'])} orphan texts"
                ),
                "image": self._render_merge_overlay(
                    (width, height),
                    pipeline["merged_components"],
                    pipeline["texts"],
                    pipeline["orphan_texts"],
                    pipeline["arrows"],
                ),
                "components": self._copy_components(pipeline["merged_components"]),
                "texts": self._copy_texts(pipeline["texts"]),
                "orphan_texts": self._copy_texts(pipeline["orphan_texts"]),
            },
        ]

        return {
            "svg": svg,
            "width": width,
            "height": height,
            "stages": stages,
            "warnings": [warning for warning in warnings.values() if warning],
        }

    def _run_pipeline(
        self,
        image: Image.Image,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        image = image.convert("RGB")
        warnings: dict[str, str | None] = {"ocr": None, "arrows": None, "sam": None}

        try:
            texts = self._run_ocr(image, self.device, conf_thresh=self.ocr_conf_thresh)
        except Exception as exc:
            if not allow_partial:
                raise
            texts = []
            warnings["ocr"] = f"OCR unavailable: {exc}"

        if self.erase_text_before_sam and texts:
            image_after_text_erase = self._erase_text(image, texts)
        else:
            image_after_text_erase = image.copy()

        try:
            if self.use_arrows:
                arrows = self._decorate_arrows(
                    self._detect_arrows(image_after_text_erase)
                )
            else:
                arrows = []
        except Exception as exc:
            if not allow_partial:
                raise
            arrows = []
            warnings["arrows"] = f"Arrow detection unavailable: {exc}"

        image_before_arrow_erase = image_after_text_erase.copy()
        if arrows:
            image_after_arrow_erase = self._erase_arrows(image_after_text_erase, arrows)
        else:
            image_after_arrow_erase = image_after_text_erase.copy()

        try:
            if self.use_sam:
                sam_components = self._run_sam(
                    image_after_arrow_erase, self.sam_model_id, self.device
                )
            else:
                sam_components = []
        except Exception as exc:
            if not allow_partial:
                raise
            sam_components = []
            warnings["sam"] = f"SAM unavailable: {exc}"

        merged_components = self._copy_components(sam_components)
        merged_components, orphan_texts = self._merge_ocr_sam(texts, merged_components)

        return {
            "original_image": image,
            "texts": self._copy_texts(texts),
            "image_after_text_erase": image_after_text_erase,
            "arrows": self._copy_arrows(arrows),
            "image_before_arrow_erase": image_before_arrow_erase,
            "image_after_arrow_erase": image_after_arrow_erase,
            "sam_components": self._copy_components(sam_components),
            "merged_components": self._copy_components(merged_components),
            "orphan_texts": self._copy_texts(orphan_texts),
            "warnings": warnings,
        }

    @staticmethod
    def _copy_texts(texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied: list[dict[str, Any]] = []
        for text in texts:
            copied.append(
                {
                    "text": str(text["text"]),
                    "bbox": [int(v) for v in text["bbox"]],
                    "conf": float(text["conf"]),
                    "cx": float(text["cx"]),
                    "cy": float(text["cy"]),
                    "font_size": int(text["font_size"]),
                }
            )
        return copied

    @staticmethod
    def _copy_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied: list[dict[str, Any]] = []
        for component in components:
            copied_component: dict[str, Any] = {
                "id": str(component["id"]),
                "bbox": [int(v) for v in component["bbox"]],
            }
            if "texts" in component:
                copied_component["texts"] = PingToSVG._copy_texts(component["texts"])
            copied.append(copied_component)
        return copied

    @staticmethod
    def _copy_arrows(arrows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied: list[dict[str, Any]] = []
        for arrow in arrows:
            copied.append(
                {
                    "id": str(arrow["id"]),
                    "bbox": [int(v) for v in arrow["bbox"]],
                    "orientation": str(arrow["orientation"]),
                    "x1": int(arrow["x1"]),
                    "y1": int(arrow["y1"]),
                    "x2": int(arrow["x2"]),
                    "y2": int(arrow["y2"]),
                }
            )
        return copied

    @staticmethod
    def _measure_text(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text)
            return right - left, bottom - top
        except Exception:
            return max(12, len(text) * 6), 12

    @classmethod
    def _draw_badge(
        cls,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        fill: tuple[int, int, int, int],
        text_fill: tuple[int, int, int, int],
    ) -> None:
        text_width, text_height = cls._measure_text(draw, text)
        x, y = xy
        padding_x = 8
        padding_y = 4
        draw.rounded_rectangle(
            (
                x,
                y,
                x + text_width + padding_x * 2,
                y + text_height + padding_y * 2,
            ),
            radius=10,
            fill=fill,
        )
        draw.text((x + padding_x, y + padding_y), text, fill=text_fill)

    def _render_ocr_overlay(
        self,
        size: tuple[int, int],
        texts: list[dict[str, Any]],
    ) -> Image.Image:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for index, text in enumerate(texts):
            x1, y1, x2, y2 = [int(v) for v in text["bbox"]]
            outline = (215, 129, 42, 235) if index % 2 == 0 else (67, 121, 212, 235)
            fill = (255, 242, 222, 76) if index % 2 == 0 else (226, 236, 255, 70)
            draw.rounded_rectangle(
                (x1, y1, x2, y2), radius=10, outline=outline, width=3, fill=fill
            )
            label = text["text"][:18] or "OCR"
            badge_y = max(0, y1 - 22)
            self._draw_badge(
                draw, (x1, badge_y), label, (18, 25, 35, 208), (255, 253, 249, 255)
            )

        return overlay

    @staticmethod
    def _decorate_arrows(arrows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decorated: list[dict[str, Any]] = []
        for arrow in arrows:
            x1, y1, x2, y2 = [int(v) for v in arrow["bbox"]]
            width = x2 - x1
            height = y2 - y1
            orientation = "horizontal" if width >= height else "vertical"
            if orientation == "horizontal":
                decorated.append(
                    {
                        "id": str(arrow["id"]),
                        "bbox": [x1, y1, x2, y2],
                        "orientation": orientation,
                        "x1": x1,
                        "y1": int((y1 + y2) / 2),
                        "x2": x2,
                        "y2": int((y1 + y2) / 2),
                    }
                )
            else:
                decorated.append(
                    {
                        "id": str(arrow["id"]),
                        "bbox": [x1, y1, x2, y2],
                        "orientation": orientation,
                        "x1": int((x1 + x2) / 2),
                        "y1": y1,
                        "x2": int((x1 + x2) / 2),
                        "y2": y2,
                    }
                )
        return decorated

    def _render_arrow_overlay(
        self,
        size: tuple[int, int],
        arrows: list[dict[str, Any]],
    ) -> Image.Image:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for arrow in arrows:
            x1, y1, x2, y2 = [int(v) for v in arrow["bbox"]]
            draw.rounded_rectangle(
                (x1, y1, x2, y2),
                radius=10,
                outline=(58, 150, 98, 235),
                width=3,
                fill=(140, 205, 164, 54),
            )
            self._draw_badge(
                draw,
                (x1, max(0, y1 - 22)),
                arrow["id"],
                (18, 25, 35, 208),
                (235, 255, 241, 255),
            )

        return overlay

    def _render_sam_overlay(
        self,
        size: tuple[int, int],
        components: list[dict[str, Any]],
    ) -> Image.Image:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        palette = [
            ((54, 122, 201, 220), (173, 212, 255, 68)),
            ((84, 168, 112, 220), (182, 233, 192, 70)),
            ((180, 98, 178, 220), (230, 192, 231, 68)),
            ((215, 129, 42, 220), (255, 225, 181, 70)),
        ]

        for index, component in enumerate(components):
            outline, fill = palette[index % len(palette)]
            x1, y1, x2, y2 = [int(v) for v in component["bbox"]]
            draw.rounded_rectangle(
                (x1, y1, x2, y2), radius=12, outline=outline, width=3, fill=fill
            )
            self._draw_badge(
                draw,
                (x1, max(0, y1 - 22)),
                str(component["id"]),
                (18, 25, 35, 208),
                (255, 253, 249, 255),
            )

        return overlay

    def _render_merge_overlay(
        self,
        size: tuple[int, int],
        components: list[dict[str, Any]],
        texts: list[dict[str, Any]],
        orphan_texts: list[dict[str, Any]],
        arrows: list[dict[str, Any]],
    ) -> Image.Image:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for component in components:
            x1, y1, x2, y2 = [int(v) for v in component["bbox"]]
            draw.rounded_rectangle(
                (x1, y1, x2, y2),
                radius=12,
                outline=(54, 122, 201, 235),
                width=3,
                fill=(173, 212, 255, 48),
            )

        for text in texts:
            x1, y1, x2, y2 = [int(v) for v in text["bbox"]]
            color = (215, 129, 42, 230)
            if any(orphan["bbox"] == text["bbox"] for orphan in orphan_texts):
                color = (204, 51, 0, 230)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=10, outline=color, width=2)

        for arrow in arrows:
            x1, y1, x2, y2 = [int(v) for v in arrow["bbox"]]
            draw.rounded_rectangle(
                (x1, y1, x2, y2),
                radius=10,
                outline=(84, 168, 112, 210),
                width=2,
            )

        return overlay

    # -------------------------------------------------------------------------
    # bbox utils
    # -------------------------------------------------------------------------
    @staticmethod
    def _area(b: list[int]) -> float:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    @classmethod
    def _iou(cls, a: list[int], b: list[int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = cls._area(a) + cls._area(b) - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _contain_ratio(cls, inner: list[int], outer: list[int]) -> float:
        ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a_inner = cls._area(inner)
        return inter / a_inner if a_inner > 0 else 0.0

    @classmethod
    def _deduplicate_leaf_first(
        cls, components: list[dict[str, Any]], thresh: float = 0.50
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for comp in sorted(components, key=lambda c: cls._area(c["bbox"])):
            if all(cls._iou(comp["bbox"], k["bbox"]) < thresh for k in kept):
                kept.append(comp)
        return kept

    # -------------------------------------------------------------------------
    # OCR
    # -------------------------------------------------------------------------
    def _run_ocr(
        self,
        image: Image.Image,
        device: str,
        conf_thresh: float,
    ) -> list[dict[str, Any]]:
        iw, ih = image.size
        reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
        raw = reader.readtext(np.array(image), batch_size=8)

        texts: list[dict[str, Any]] = []
        for pts, text, conf in raw:
            if conf < conf_thresh or not text.strip():
                continue

            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            h_box = y2 - y1
            texts.append(
                {
                    "text": text.strip(),
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "font_size": max(8, int(h_box * 0.75)),
                }
            )

        texts.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
        return texts

    def _erase_text(
        self,
        image: Image.Image,
        texts: list[dict[str, Any]],
        mask_padding: int = 2,
        dark_thresh: int = 60,
        dilation: int = 2,
    ) -> Image.Image:
        if not texts:
            return image

        img = np.array(image).copy()
        ih, iw = img.shape[:2]

        for txt in texts:
            x1, y1, x2, y2 = txt["bbox"]
            x1 = max(0, x1 - mask_padding)
            y1 = max(0, y1 - mask_padding)
            x2 = min(iw, x2 + mask_padding)
            y2 = min(ih, y2 + mask_padding)
            if x2 <= x1 or y2 <= y1:
                continue

            region = img[y1:y2, x1:x2]
            core_mask = np.all(region < dark_thresh, axis=2)

            dilated = core_mask.copy()
            for _ in range(dilation):
                dilated[1:] |= dilated[:-1]
                dilated[:-1] |= dilated[1:]
                dilated[:, 1:] |= dilated[:, :-1]
                dilated[:, :-1] |= dilated[:, 1:]

            bg_pixels = region[~dilated]
            fill = (
                np.median(bg_pixels, axis=0).astype(np.uint8)
                if len(bg_pixels) > 0
                else np.array([255, 255, 255], dtype=np.uint8)
            )

            region[dilated] = fill

        return Image.fromarray(img)

    # -------------------------------------------------------------------------
    # Arrow processing
    # -------------------------------------------------------------------------
    def _erase_arrows(
        self,
        image: Image.Image,
        arrows: list[dict[str, Any]],
        dilation: int = 2,
    ) -> Image.Image:
        if not arrows:
            return image

        img = np.array(image).copy()
        ih, iw = img.shape[:2]

        for arrow in arrows:
            x1, y1, x2, y2 = arrow["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(iw, x2)
            y2 = min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            region = img[y1:y2, x1:x2]
            region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            s = region_hsv[:, :, 1]
            v = region_hsv[:, :, 2]
            gray_mask = (
                (s < self.ARROW_S_MAX)
                & (v >= self.ARROW_V_MIN)
                & (v <= self.ARROW_V_MAX)
            )

            dilated = gray_mask.copy()
            for _ in range(dilation):
                dilated[1:] |= dilated[:-1]
                dilated[:-1] |= dilated[1:]
                dilated[:, 1:] |= dilated[:, :-1]
                dilated[:, :-1] |= dilated[:, 1:]

            border = dilated.copy()
            border[1:] |= border[:-1]
            border[:-1] |= border[1:]
            border[:, 1:] |= border[:, :-1]
            border[:, :-1] |= border[:, 1:]
            border_only = border & ~dilated

            edge_pixels = region[border_only]
            if len(edge_pixels) > 0:
                fill = np.median(edge_pixels, axis=0).astype(np.uint8)
            else:
                bg_pixels = region[~dilated]
                fill = (
                    np.median(bg_pixels, axis=0).astype(np.uint8)
                    if len(bg_pixels) > 0
                    else np.array([255, 255, 255], dtype=np.uint8)
                )

            region[dilated] = fill

        return Image.fromarray(img)

    def _find_arrowhead_blobs(self, mask: np.ndarray) -> list[tuple[int, int]]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.ARROW_ERODE_PX, self.ARROW_ERODE_PX)
        )
        eroded = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(eroded) == 0:
            return []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )

        heads: list[tuple[int, int]] = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < 20:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.07 * peri, True)
                if not (3 <= len(approx) <= 8):
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                if hull_area > 0 and a / hull_area >= self.ARROW_SOLIDITY_MIN:
                    heads.append((int(centroids[i][0]), int(centroids[i][1])))
                    break
        return heads

    def _detect_arrows(self, image: Image.Image) -> list[dict[str, Any]]:
        img_np = np.array(image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        gray_bin = (
            (s < self.ARROW_S_MAX) & (v >= self.ARROW_V_MIN) & (v <= self.ARROW_V_MAX)
        ).astype(np.uint8) * 255

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray_bin, cv2.MORPH_CLOSE, k3, iterations=2)

        head_positions = self._find_arrowhead_blobs(closed)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )

        label_head_count: dict[int, int] = defaultdict(int)
        for hx, hy in head_positions:
            if 0 <= hy < labels.shape[0] and 0 <= hx < labels.shape[1]:
                lbl = int(labels[hy, hx])
                if lbl != 0:
                    label_head_count[lbl] += 1

        arrows: list[dict[str, Any]] = []
        for lbl, head_count in sorted(label_head_count.items()):
            if head_count >= 4:
                continue

            x, y, w, h, comp_area = stats[lbl]
            if comp_area < self.ARROW_MIN_AREA:
                continue
            if comp_area > self.ARROW_MAX_AREA:
                continue

            arrows.append(
                {"id": f"arrow_{len(arrows)+1}", "bbox": [x, y, x + w, y + h]}
            )

        return arrows

    # -------------------------------------------------------------------------
    # SAM
    # -------------------------------------------------------------------------
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> list[int] | None:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1 = int(np.where(rows)[0][0])
        y2 = int(np.where(rows)[0][-1]) + 1
        x1 = int(np.where(cols)[0][0])
        x2 = int(np.where(cols)[0][-1]) + 1
        return [x1, y1, x2, y2]

    def _run_sam(
        self,
        image: Image.Image,
        sam_model_id: str,
        device: str,
    ) -> list[dict[str, Any]]:
        from transformers import pipeline as hf_pipeline

        if not self._is_hf_model_available_locally(sam_model_id):
            raise RuntimeError(f"SAM model not available locally: {sam_model_id}")

        iw, ih = image.size
        total_area = iw * ih
        device_idx = 0 if device == "cuda" else -1

        mask_gen = hf_pipeline(
            "mask-generation",
            model=sam_model_id,
            device=device_idx,
            local_files_only=True,
        )
        outputs = mask_gen(
            image,
            points_per_batch=64,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.70,
            crops_n_layers=0,
        )

        raw_masks: list[Any] = []
        if isinstance(outputs, dict):
            raw_masks = outputs.get("masks", [])
        elif isinstance(outputs, list):
            for m in outputs:
                if isinstance(m, dict):
                    arr = m.get("mask") or m.get("segmentation")
                    if arr is not None:
                        raw_masks.append(arr)

        components: list[dict[str, Any]] = []
        for i, mask in enumerate(raw_masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_np = np.asarray(mask, dtype=bool)
            if mask_np.sum() == 0:
                continue

            bbox = self._mask_to_bbox(mask_np)
            if bbox is None:
                continue

            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ratio = (bw * bh) / total_area
            aspect = max(bw, bh) / max(min(bw, bh), 1)

            if ratio < self.SAM_MIN_AREA_RATIO or ratio > self.SAM_MAX_AREA_RATIO:
                continue
            if aspect > self.SAM_MAX_ASPECT:
                continue

            components.append({"id": f"s{i+1}", "bbox": bbox})

        components = self._deduplicate_leaf_first(components, self.SAM_IOU_DEDUP)
        if len(components) > self.SAM_MAX_COMPONENTS:
            components = components[: self.SAM_MAX_COMPONENTS]

        components.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        return components

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------
    def _merge_ocr_sam(
        self,
        texts: list[dict[str, Any]],
        sam_components: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        for blk in sam_components:
            blk["texts"] = []

        orphan_texts: list[dict[str, Any]] = []

        for txt in texts:
            best_ratio = 0.0
            best_blk = None
            for blk in sam_components:
                r = self._contain_ratio(txt["bbox"], blk["bbox"])
                if r > best_ratio:
                    best_ratio = r
                    best_blk = blk

            if best_blk is not None and best_ratio >= self.TEXT_IN_BOX_THRESH:
                best_blk["texts"].append(txt)
            else:
                orphan_texts.append(txt)

        return sam_components, orphan_texts

    # -------------------------------------------------------------------------
    # SVG builders
    # -------------------------------------------------------------------------
    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _esc(t: object) -> str:
        return (
            str(t)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    @staticmethod
    def _is_hf_model_available_locally(model_id: str) -> bool:
        model_path = Path(model_id)
        if model_path.exists():
            return True

        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if not model_dir.exists():
            return False

        for snapshot in model_dir.iterdir():
            if snapshot.is_dir() and (snapshot / "config.json").exists():
                return True
        return False

    def _crop_b64(
        self, image: Image.Image, bbox: list[int], width: int, height: int
    ) -> tuple[int, int, int, int, str]:
        x1, y1, x2, y2 = bbox
        cx1 = max(0, x1 - self.CROP_PADDING)
        cy1 = max(0, y1 - self.CROP_PADDING)
        cx2 = min(width, x2 + self.CROP_PADDING)
        cy2 = min(height, y2 + self.CROP_PADDING)
        return cx1, cy1, cx2, cy2, self._pil_to_b64(image.crop((cx1, cy1, cx2, cy2)))

    def _arrow_crop_b64(
        self,
        image: Image.Image,
        bbox: list[int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, str]:
        x1, y1, x2, y2 = bbox
        cx1 = max(0, x1 - self.CROP_PADDING)
        cy1 = max(0, y1 - self.CROP_PADDING)
        cx2 = min(width, x2 + self.CROP_PADDING)
        cy2 = min(height, y2 + self.CROP_PADDING)

        crop_rgb = np.array(image.crop((cx1, cy1, cx2, cy2)))
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        arrow_mask = (
            (s < self.ARROW_S_MAX) & (v >= self.ARROW_V_MIN) & (v <= self.ARROW_V_MAX)
        )

        rgba = np.zeros((*crop_rgb.shape[:2], 4), dtype=np.uint8)
        rgba[arrow_mask, :3] = crop_rgb[arrow_mask]
        rgba[arrow_mask, 3] = 255

        return cx1, cy1, cx2, cy2, self._pil_to_b64(Image.fromarray(rgba, "RGBA"))

    def _build_svg(
        self,
        image: Image.Image,
        sam_components: list[dict[str, Any]],
        arrows: list[dict[str, Any]],
        all_texts: list[dict[str, Any]],
        arrow_image: Image.Image | None = None,
        ocr_overlay: bool = True,
        show_outline: bool = False,
    ) -> str:
        width, height = image.size
        if arrow_image is None:
            arrow_image = image

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg"',
            f'     xmlns:xlink="http://www.w3.org/1999/xlink"',
            f'     width="{width}" height="{height}"',
            f'     viewBox="0 0 {width} {height}">',
            '  <rect width="100%" height="100%" fill="white"/>',
        ]

        if sam_components:
            lines.append("  <!-- Layer 1: SAM visual blocks -->")
            lines.append('  <g id="visual-blocks">')
            for blk in sam_components:
                bid = self._esc(blk["id"])
                x1, y1, x2, y2 = blk["bbox"]
                w, h = x2 - x1, y2 - y1
                cx1, cy1, cx2, cy2, b64 = self._crop_b64(
                    image, blk["bbox"], width, height
                )

                lines.append(f'    <g id="{bid}">')
                lines.append(
                    f'      <image x="{cx1}" y="{cy1}" width="{cx2-cx1}" height="{cy2-cy1}" '
                    f'href="data:image/png;base64,{b64}"/>'
                )
                if show_outline:
                    lines.append(
                        f'      <rect x="{x1}" y="{y1}" width="{w}" height="{h}" '
                        f'fill="none" stroke="#0055cc" stroke-width="1.5" stroke-dasharray="4 2"/>'
                    )
                lines.append("    </g>")
            lines.append("  </g>")
            lines.append("")

        if arrows:
            lines.append("  <!-- Layer 1.5: Arrow components -->")
            lines.append('  <g id="arrow-layer">')
            for arrow in arrows:
                aid = self._esc(arrow["id"])
                x1, y1, x2, y2 = arrow["bbox"]
                w, h = x2 - x1, y2 - y1
                cx1, cy1, cx2, cy2, b64 = self._arrow_crop_b64(
                    arrow_image, arrow["bbox"], width, height
                )

                lines.append(f'    <g id="{aid}">')
                lines.append(
                    f'      <image x="{cx1}" y="{cy1}" width="{cx2-cx1}" height="{cy2-cy1}" '
                    f'href="data:image/png;base64,{b64}"/>'
                )
                if show_outline:
                    lines.append(
                        f'      <rect x="{x1}" y="{y1}" width="{w}" height="{h}" '
                        f'fill="none" stroke="#00aa44" stroke-width="1.5" stroke-dasharray="4 2"/>'
                    )
                lines.append("    </g>")
            lines.append("  </g>")
            lines.append("")

        if ocr_overlay and all_texts:
            lines.append("  <!-- Layer 2: OCR text -->")
            lines.append(
                '  <g id="text-layer" font-family="Arial,Helvetica,sans-serif">'
            )
            for txt in all_texts:
                x1, y1, x2, y2 = txt["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                fs = txt["font_size"]
                conf = txt["conf"]
                alpha = min(1.0, max(0.5, conf))
                opacity = f"{alpha:.2f}"
                escaped = self._esc(txt["text"])

                if show_outline:
                    w, h = x2 - x1, y2 - y1
                    lines.append(
                        f'    <rect x="{x1}" y="{y1}" width="{w}" height="{h}" '
                        f'fill="none" stroke="#cc3300" stroke-width="1" stroke-dasharray="3 2"/>'
                    )

                lines.append(
                    f'    <text x="{cx:.1f}" y="{cy:.1f}" font-size="{fs}" text-anchor="middle" '
                    f'dominant-baseline="middle" opacity="{opacity}" fill="#111111">{escaped}</text>'
                )
            lines.append("  </g>")
            lines.append("")

        lines.append("</svg>")
        return "\n".join(lines)
