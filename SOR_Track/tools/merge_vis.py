# tools/merge_vis.py
# 修改点：
#   1. get_orig() 优先读 _raw.jpg，消除双重绘制
#   2. draw_bbox() 删除 label 参数（不在框旁写字）
#   3. 新增 draw_legend_corner() 统一管理图例
#   4. 三种布局函数均改用新图例函数

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

#  颜色定义（BGR）
COLOR_PRED_A = (0,   80,  255)   # 红
COLOR_PRED_B = (255, 100,   0)   # 蓝橙
COLOR_GT     = (0,   210,   0)   # 绿

#  基础工具 

def load_bboxes(txt_path):
    bboxes = {}
    if not os.path.exists(txt_path):
        return bboxes
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            fid  = int(parts[0])
            pred = [float(v) for v in parts[1:5]]
            gt   = ([float(v) for v in parts[5:9]]
                    if len(parts) >= 9 and parts[5] != 'None'
                    else None)
            bboxes[fid] = (pred, gt)
    return bboxes


def draw_bbox(img, bbox, color, thickness=2):
    """只画框，不在框旁写文字（文字统一由图例函数处理）"""
    if bbox is None:
        return img
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    return img


def draw_legend_corner(img, items,
                        margin=10, box_size=14,
                        font_scale=0.42, line_h=22):
    """
    右下角半透明图例。
    items: [(BGR_color, label_str), ...]
    """
    n = len(items)
    max_tw = max(
        cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for _, lbl in items
    )
    pad    = 8
    lw     = pad + box_size + 6 + max_tw + pad
    lh     = pad + n * line_h + pad

    ih, iw = img.shape[:2]
    x0 = iw - lw - margin
    y0 = ih - lh - margin

    # 半透明黑底
    roi = img[y0:y0+lh, x0:x0+lw]
    bg  = np.zeros_like(roi)
    img[y0:y0+lh, x0:x0+lw] = cv2.addWeighted(roi, 0.3, bg, 0.7, 0)

    for i, (color, label) in enumerate(items):
        iy = y0 + pad + i * line_h
        cv2.rectangle(img,
                      (x0+pad,            iy),
                      (x0+pad+box_size,   iy+box_size),
                      color, -1)
        cv2.putText(img, label,
                    (x0+pad+box_size+6,   iy+box_size-1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * target_h / h), target_h))


def add_top_banner(img, text, font_scale=0.52,
                   bg=(30, 30, 30), fg=(255, 255, 255)):
    out   = img.copy()
    h_bar = 26
    cv2.rectangle(out, (0, 0), (img.shape[1], h_bar), bg, -1)
    cv2.putText(out, text, (8, h_bar-7),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 1, cv2.LINE_AA)
    return out


def get_orig_frame(debug_dir, fname):
    """
    优先读 XXXXX_raw.jpg（无框纯原图）；
    若不存在则回退到 XXXXX.jpg 并裁取左1/3（兼容旧格式）。
    返回 RGB numpy array。
    """
    stem     = os.path.splitext(fname)[0]          # e.g. "00010"
    raw_path = os.path.join(debug_dir, stem + '_raw.jpg')

    if os.path.exists(raw_path):
        bgr = cv2.imread(raw_path)
    else:
        # 兼容旧格式：从拼接图裁取左1/3
        full = cv2.imread(os.path.join(debug_dir, fname))
        if full is None:
            return np.zeros((360, 480, 3), dtype=np.uint8)
        w3  = full.shape[1] // 3
        bgr = full[:, :w3, :]

    if bgr is None:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


#  布局函数 

def merge_side_by_side(orig_a, orig_b, pred_a, pred_b, gt,
                        label_a, label_b, target_h=360):
    """
    左面板：orig_a + pred_a(红) + gt(绿) + 右下角图例
    右面板：orig_b + pred_b(蓝) + gt(绿) + 右下角图例
    """
    left  = cv2.cvtColor(orig_a.copy(), cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(orig_b.copy(), cv2.COLOR_RGB2BGR)

    draw_bbox(left,  gt,     COLOR_GT,     thickness=2)
    draw_bbox(left,  pred_a, COLOR_PRED_A, thickness=2)
    draw_bbox(right, gt,     COLOR_GT,     thickness=2)
    draw_bbox(right, pred_b, COLOR_PRED_B, thickness=2)

    # # 图例：每个面板只显示自己模型的颜色
    # draw_legend_corner(left,  [(COLOR_GT, 'GT'), (COLOR_PRED_A, label_a)])
    # draw_legend_corner(right, [(COLOR_GT, 'GT'), (COLOR_PRED_B, label_b)])

    left  = resize_to_height(left,  target_h)
    right = resize_to_height(right, target_h)
    left  = add_top_banner(left,  label_a)
    right = add_top_banner(right, label_b)

    div = np.full((target_h, 4, 3), 180, dtype=np.uint8)
    return np.hstack([left, div, right])


def merge_overlay(orig_a, pred_a, pred_b, gt,
                   label_a, label_b, target_h=360):
    """
    单图叠加：gt(绿) + pred_a(红) + pred_b(蓝) + 右下角图例
    """
    canvas = cv2.cvtColor(orig_a.copy(), cv2.COLOR_RGB2BGR)

    draw_bbox(canvas, gt,     COLOR_GT,     thickness=2)
    draw_bbox(canvas, pred_a, COLOR_PRED_A, thickness=2)
    draw_bbox(canvas, pred_b, COLOR_PRED_B, thickness=2)

    # draw_legend_corner(canvas, [
    #     (COLOR_GT,     'GT'),
    #     (COLOR_PRED_A, label_a),
    #     (COLOR_PRED_B, label_b),
    # ])

    canvas = resize_to_height(canvas, target_h)
    return canvas


def merge_quad(orig_a, orig_b, debug_dir_a, debug_dir_b, fname,
               pred_a, pred_b, gt, label_a, label_b, target_h=320):
    """
    四格：
    [frame_a+bbox] | [frame_b+bbox]
    [scoreA heatmap] | [scoreB heatmap]
    """
    tl = cv2.cvtColor(orig_a.copy(), cv2.COLOR_RGB2BGR)
    tr = cv2.cvtColor(orig_b.copy(), cv2.COLOR_RGB2BGR)

    draw_bbox(tl, gt,     COLOR_GT,     thickness=2)
    draw_bbox(tl, pred_a, COLOR_PRED_A, thickness=2)
    draw_bbox(tr, gt,     COLOR_GT,     thickness=2)
    draw_bbox(tr, pred_b, COLOR_PRED_B, thickness=2)

    # draw_legend_corner(tl, [(COLOR_GT, 'GT'), (COLOR_PRED_A, label_a)])
    # draw_legend_corner(tr, [(COLOR_GT, 'GT'), (COLOR_PRED_B, label_b)])

    tl = resize_to_height(tl, target_h)
    tr = resize_to_height(tr, target_h)
    tl = add_top_banner(tl, label_a)
    tr = add_top_banner(tr, label_b)
    panel_w = tl.shape[1]

    def load_score(debug_dir, fname, out_wh):
        full_path = os.path.join(debug_dir, fname)
        full = cv2.imread(full_path)
        if full is None:
            return np.zeros((out_wh[1], out_wh[0], 3), dtype=np.uint8)
        sw = full.shape[1] // 3
        return cv2.resize(full[:, 2*sw:, :], out_wh)

    bl = load_score(debug_dir_a, fname, (panel_w, target_h))
    br = load_score(debug_dir_b, fname, (panel_w, target_h))
    bl = add_top_banner(bl, f"ScoreMap: {label_a}")
    br = add_top_banner(br, f"ScoreMap: {label_b}")

    div_v = np.full((target_h, 4, 3), 180, dtype=np.uint8)
    div_h = np.full((4, panel_w*2+4, 3), 180, dtype=np.uint8)
    top = np.hstack([tl, div_v, tr])
    bot = np.hstack([bl, div_v, br])
    return np.vstack([top, div_h, bot])


#  主流程 

def merge_sequence(dir_a, dir_b, out_dir, label_a, label_b,
                   layout='side_by_side', target_h=360,
                   output_video=False, fps=10):

    os.makedirs(out_dir, exist_ok=True)

    frames_a = sorted([f for f in os.listdir(dir_a)
                        if f.endswith('.jpg')
                        and not f.startswith('_')
                        and not f.endswith('_raw.jpg')])   # ← 排除 _raw.jpg
    frames_b = sorted([f for f in os.listdir(dir_b)
                        if f.endswith('.jpg')
                        and not f.startswith('_')
                        and not f.endswith('_raw.jpg')])

    common = sorted(set(frames_a) & set(frames_b))

    if not common:
        print(f"[ERROR] No common frames:\n  {dir_a}\n  {dir_b}")
        return

    bboxes_a = load_bboxes(os.path.join(dir_a, '_pred_bboxes.txt'))
    bboxes_b = load_bboxes(os.path.join(dir_b, '_pred_bboxes.txt'))

    canvas_list = []

    for fname in tqdm(common, desc=f"{label_a} vs {label_b}"):
        fid = int(os.path.splitext(fname)[0])

        # ← 核心修复：使用无框原图
        orig_a = get_orig_frame(dir_a, fname)
        orig_b = get_orig_frame(dir_b, fname)

        pred_a, gt = bboxes_a.get(fid, (None, None))
        pred_b, _  = bboxes_b.get(fid, (None, None))

        if layout == 'side_by_side':
            canvas = merge_side_by_side(
                orig_a, orig_b, pred_a, pred_b, gt,
                label_a, label_b, target_h)
        elif layout == 'overlay':
            canvas = merge_overlay(
                orig_a, pred_a, pred_b, gt,
                label_a, label_b, target_h)
        elif layout == 'quad':
            canvas = merge_quad(
                orig_a, orig_b, dir_a, dir_b, fname,
                pred_a, pred_b, gt,
                label_a, label_b, target_h)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # 帧号水印（右下角图例上方）
        cv2.putText(canvas,
                    f"frame {fid:05d}",
                    (canvas.shape[1] - 105, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, fname), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        canvas_list.append(canvas)

    print(f"[✓] {len(common)} frames → {out_dir}")

    if output_video and canvas_list:
        h, w   = canvas_list[0].shape[:2]
        vpath  = os.path.join(out_dir, '_compare.mp4')
        writer = cv2.VideoWriter(vpath,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))
        for f in canvas_list:
            writer.write(f)
        writer.release()
        print(f"[✓] Video → {vpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_a',   required=True)
    parser.add_argument('--dir_b',   required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--label_a', default='Base')
    parser.add_argument('--label_b', default='SOR')
    parser.add_argument('--layout',  default='side_by_side',
                        choices=['side_by_side', 'overlay', 'quad'])
    parser.add_argument('--height',  type=int, default=360)
    parser.add_argument('--video',   action='store_true')
    parser.add_argument('--fps',     type=int, default=10)
    args = parser.parse_args()

    merge_sequence(
        dir_a=args.dir_a,   dir_b=args.dir_b,
        out_dir=args.out_dir,
        label_a=args.label_a, label_b=args.label_b,
        layout=args.layout,   target_h=args.height,
        output_video=args.video, fps=args.fps,
    )

if __name__ == '__main__':
    main()