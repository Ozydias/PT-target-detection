# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO


def yolov12_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(model_id)
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        result = results[0]
        
        # 检查是否有检测结果
        num_detections = len(result.boxes) if result.boxes is not None else 0
        print(f"检测到 {num_detections} 个目标，置信度阈值: {conf_threshold}")
        
        # 绘制检测框，确保boxes=True（默认就是True，但显式指定更安全）
        annotated_image = result.plot(
            boxes=True,      # 确保绘制边界框
            labels=True,     # 显示标签
            conf=True,       # 显示置信度
            line_width=2     # 增加线条宽度，使框更明显
        )
        
        # plot()返回的是RGB格式的numpy数组，Gradio的Image组件需要RGB格式
        # 所以不需要转换，直接返回即可
        # 如果annotated_image是PIL Image，需要转换为numpy数组
        if not isinstance(annotated_image, np.ndarray):
            annotated_image = np.array(annotated_image)
        
        return annotated_image, None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot(
                boxes=True,
                labels=True,
                conf=True,
                line_width=2
            )
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov12n.pt",
                        "yolov12s.pt",
                        "yolov12m.pt",
                        "yolov12l.pt",
                        "yolov12x.pt",
                        "runs/detect/yolov12s_pt_v2/weights/best.pt",  # 训练好的模型
                        "runs/detect/yolov12s_pt_v2/weights/last.pt",   # 最后一个epoch的模型
                    ],
                    value="runs/detect/yolov12s_pt_v2/weights/best.pt",  # 默认使用训练好的模型
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold (如果看不到框，尝试降低此值)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.1,  # 降低默认阈值，更容易看到检测框
                )
                yolov12_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov12_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov12_inference(None, video, model_id, image_size, conf_threshold)


        yolov12_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov12s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov12x.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov12_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv12: Attention-Centric Real-Time Object Detectors
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
