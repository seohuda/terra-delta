"""Verbatim official mask-to-polygon reference; do not optimize this module.

Source: baseline/original/03_illegal structure submission/predict.ipynb,
setup cell (cell index 1). The function and constants below are copied unchanged.
"""

import json

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

H, W = 256, 256
BATCH = 16
CLASSES = ("new_building", "tree_removal")       # 모델 출력 채널 1, 2 (0 은 배경)
MIN_AREA = 30            # 이 면적(px^2) 미만의 조각은 지웁니다 (학습 때와 같은 후처리)
MIN_POS_AREA = 20.0      # 조각을 지운 뒤 남은 총면적이 이 값 미만이면 빈 예측으로 냅니다 (채점 규칙과 동일)
SIMPLIFY_PX = 0.5        # 계단형 외곽선을 이 허용 오차로 단순화합니다 (위상 보존). 채점 허용 오차 1px 안입니다
NDIGITS = 2              # 좌표 소수 자릿수


def mask_to_polygons(mask: np.ndarray) -> str:
    """(H,W) 이진 마스크 -> 폴리곤 목록 JSON 문자열(폴리곤 = 점 목록, 구멍 없음). 변화가 없으면 빈 문자열.

    행 단위 런(run)을 화소 경계 사각형으로 만들어 합집합하면 마스크의 외곽선을 그대로 따르는 폴리곤이 나옵니다.
    좌표는 화소 모서리 기준(화소 (r, c) 는 [c, c+1] x [r, r+1])이라 채점 규약과 같습니다.
    """
    m = np.asarray(mask, dtype=bool)
    boxes = []
    for r in np.flatnonzero(m.any(axis=1)):
        pad = np.concatenate(([0], m[r].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(pad))
        for s, e in zip(edges[::2], edges[1::2]):
            boxes.append(box(float(s), float(r), float(e), float(r + 1)))
    if not boxes:
        return ""
    g = unary_union(boxes)
    parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    parts = [p for p in parts if isinstance(p, Polygon) and p.area >= MIN_AREA]     # 작은 조각 제거
    if not parts or sum(p.area for p in parts) < MIN_POS_AREA:
        return ""
    out = []
    for p in parts:
        p = p.simplify(SIMPLIFY_PX, preserve_topology=True)
        if p.is_empty or p.area <= 0:
            continue
        # 바깥 경계만 씁니다(제출 형식에 구멍이 없습니다). 마지막 점은 첫 점과 같으므로 뺍니다
        out.append([[round(float(x), NDIGITS), round(float(y), NDIGITS)] for x, y in list(p.exterior.coords)[:-1]])
    return json.dumps(out, separators=(",", ":")) if out else ""
