import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# --- Cấu hình trang ---
st.set_page_config(page_title="AI Vehicle Counter", layout="wide")
st.title("Hệ thống đếm phương tiện giao thông (YOLOv8)")

# --- Load Model (Cache để không load lại mỗi lần nhấn nút) ---
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()

# --- Sidebar ---
st.sidebar.header("Cấu hình")
conf_threshold = st.sidebar.slider("Độ tự tin (Confidence)", 0.0, 1.0, 0.5)
selected_classes = [2, 3, 5, 7] # Car, Motor, Bus, Truck

# --- Upload File ---
uploaded_file = st.file_uploader("Tải video lên (mp4, avi, mov)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Lưu file tạm để OpenCV có thể đọc
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')    
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    
    # Thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo đường dẫn file output
    output_path = "processed_video.mp4"
    # Dùng 'avc1' để có thể xem trực tiếp trên trình duyệt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if st.button("Bắt đầu xử lý"):
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        vehicle_count = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}
        detected_ids = set()
        
        status_text = st.empty()
        frame_placeholder = st.empty() # Để hiển thị preview

        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Xử lý với YOLO
            results = model.track(
                frame, persist=True, classes=selected_classes, 
                conf=conf_threshold, tracker="botsort.yaml", verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                
                for box, obj_id, cls_id in zip(boxes.xyxy, ids, classes):
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Vẽ khung
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Đếm
                    if obj_id not in detected_ids:
                        vehicle_count[label] = vehicle_count.get(label, 0) + 1
                        detected_ids.add(obj_id)

            # Vẽ bảng thống kê lên frame
            cv2.rectangle(frame, (20, 20), (250, 160), (0, 0, 0), -1)
            for i, (cls, cnt) in enumerate(vehicle_count.items()):
                cv2.putText(frame, f'{cls.capitalize()}: {cnt}', (30, 50 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Ghi vào file
            out.write(frame)
            
            # Cập nhật UI
            curr_frame += 1
            progress_bar.progress(curr_frame / frame_count)
            status_text.text(f"Đang xử lý frame {curr_frame}/{frame_count}")
            
            # Hiển thị preview (giảm kích thước để mượt hơn)
            if curr_frame % 5 == 0:
                preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(preview_frame, channels="RGB", use_container_width=True)

        cap.release()
        out.release()
        st.success("Xử lý hoàn tất!")
        
        # Cho phép tải video về
        with open(output_path, "rb") as file:
            st.download_button(label="Tải video kết quả", data=file, file_name="counted_video.mp4", mime="video/mp4")