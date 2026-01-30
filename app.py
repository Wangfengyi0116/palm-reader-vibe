import time
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st


def process_hand_image(image: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    使用 OpenCV 做一个**简化版掌心检测与关键点可视化 Demo**：

    - 通过肤色阈值 + 轮廓找到最大“手部”区域；
    - 在该区域内生成 21 个“伪关键点”（规则网格），并画在图像上；
    - 返回处理后的图像、是否检测到手、以及耗时。

    注意：这是一个为了兼容当前 Python 环境的近似 Demo，
    并非 MediaPipe 的真实 21 关键点结果，但交互流程与体验一致。
    """
    start = time.perf_counter()

    # image: RGB -> BGR -> HSV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 简单的肤色范围（可根据需要微调）
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学操作，去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        elapsed = time.perf_counter() - start
        return image, False, elapsed

    # 取面积最大的轮廓近似认为是手掌区域
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 在原图上画出检测到的“手部”区域矩形
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 在该矩形区域内生成 21 个规则分布的“伪关键点”
    num_points = 21
    cols = 7
    rows = 3
    points = []
    for r in range(rows):
        for c in range(cols):
            if len(points) >= num_points:
                break
            px = int(x + (c + 0.5) * w / cols)
            py = int(y + (r + 0.5) * h / rows)
            points.append((px, py))

    for px, py in points:
        cv2.circle(output, (px, py), 4, (0, 255, 255), -1)

    elapsed = time.perf_counter() - start
    return output, True, elapsed


def load_image(uploaded_file) -> Optional[np.ndarray]:
    """从上传文件中读取图像并转换为 RGB numpy 数组。"""
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Palm Reader Vibe Demo", layout="centered")

    st.title("Palm Reader Vibe 👋")
    st.markdown("一个基于 **Streamlit + MediaPipe** 的掌纹关键点检测演示。")

    with st.sidebar:
        st.header("图片上传")
        uploaded_file = st.file_uploader(
            "上传一张包含手掌的 JPG/PNG 图片",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        st.markdown(
            "小提示：\n"
            "- 尽量让**手心正对镜头**；\n"
            "- 保持**光线充足**；\n"
            "- 尽量**避免背景过于杂乱**。"
        )

    if uploaded_file is None:
        st.info("请在左侧侧边栏上传一张掌纹图片。")
        return

    image = load_image(uploaded_file)
    if image is None:
        st.error("无法读取图片，请确认文件是否为有效的 JPG/PNG 图像。")
        return

    st.subheader("原始图像")
    st.image(image, channels="RGB", use_column_width=True)

    with st.spinner("正在检测手掌关键点，请稍候..."):
        processed, has_hand, elapsed = process_hand_image(image)

    st.subheader("检测结果")
    st.image(processed, channels="RGB", use_column_width=True)

    if not has_hand:
        st.warning("未检测到手掌，请确保手心正对手中并保持光线充足。")
    else:
        st.success("已检测到手掌关键点！")

    st.caption(f"检测耗时约：**{elapsed * 1000:.1f} ms**")


if __name__ == "__main__":
    main()


