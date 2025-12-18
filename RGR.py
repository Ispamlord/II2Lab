import dlib
import cv2
import numpy as np
import face_recognition
from collections import defaultdict, OrderedDict
from sklearn.metrics.pairwise import cosine_distances
import pickle
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import json

class FaceTracker:
    def __init__(self, max_disappeared=50, max_distance=0.6, tracking_method="dlib"):
        self.next_object_id = 0
        self.objects = OrderedDict() 
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracking_method = tracking_method
        
        self.trajectories = defaultdict(list)
        
        self.known_faces = []
        self.known_names = []
        
        # Загрузка предобученных моделей
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
        print("Трекер лиц инициализирован")
    
    def _rect_to_css(self, rect):
        return (rect.top(), rect.right(), rect.bottom(), rect.left())
    
    def _css_to_rect(self, css):
        return dlib.rectangle(css[3], css[0], css[1], css[2])
    
    def _get_centroid(self, face_location):
        """Вычисляет центр лица"""
        top, right, bottom, left = face_location
        x = (left + right) // 2
        y = (top + bottom) // 2
        return (x, y)
    
    def register(self, face_location, face_encoding=None):
        object_id = self.next_object_id
        self.next_object_id += 1
        
        centroid = self._get_centroid(face_location)
        
        self.objects[object_id] = {
            "face_location": face_location,
            "face_encoding": face_encoding,
            "centroid": centroid,
            "disappeared": 0,
            "active": True
        }

        
        self.disappeared[object_id] = 0
        
        # Инициализация траектории
        self.trajectories[object_id].append(centroid)
        
        return object_id
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    
    def update(self, face_locations, face_encodings=None):
        if len(face_locations) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                self.objects[object_id]["active"] = False 
                pass
            
            return self.objects
        
        input_centroids = [self._get_centroid(loc) for loc in face_locations]
        
        if len(self.objects) == 0:
            for i, location in enumerate(face_locations):
                encoding = face_encodings[i] if face_encodings is not None else None
                self.register(location, encoding)
        else:
            object_ids = sorted(self.objects.keys()) 
            object_centroids = [self.objects[obj_id]["centroid"] for obj_id in object_ids]
            
            if self.tracking_method == "centroid":
                D = np.zeros((len(object_centroids), len(input_centroids)))
                
                for i, obj_cent in enumerate(object_centroids):
                    for j, inp_cent in enumerate(input_centroids):
                        D[i, j] = np.linalg.norm(np.array(obj_cent) - np.array(inp_cent))
                
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)
                
                used_rows = set()
                used_cols = set()
                
                for row in rows:
                    col = cols[row]
                    
                    if row in used_rows or col in used_cols:
                        continue
                    
                    if D[row, col] < 50:
                        object_id = object_ids[row]
                        
                        self.objects[object_id]["face_location"] = face_locations[col]
                        self.objects[object_id]["centroid"] = input_centroids[col]
                        self.objects[object_id]["active"] = True
                        if face_encodings is not None:
                            self.objects[object_id]["face_encoding"] = face_encodings[col]
                        self.disappeared[object_id] = 0
                        
                        self.trajectories[object_id].append(input_centroids[col])
                        
                        used_rows.add(row)
                        used_cols.add(col)
            
            elif self.tracking_method == "dlib" and face_encodings is not None:
                object_encodings = [self.objects[obj_id]["face_encoding"] 
                                  for obj_id in object_ids]
                
                D = cosine_distances(object_encodings, face_encodings)
                
                for i, obj_id in enumerate(object_ids):
                    if len(face_encodings) > 0:
                        min_idx = np.argmin(D[i])
                        if D[i, min_idx] < self.max_distance:
                            self.objects[obj_id]["face_location"] = face_locations[min_idx]
                            self.objects[obj_id]["face_encoding"] = face_encodings[min_idx]
                            self.objects[obj_id]["centroid"] = input_centroids[min_idx]
                            self.disappeared[obj_id] = 0
                            
                            # Обновляем траекторию
                            self.trajectories[obj_id].append(input_centroids[min_idx])
            
            unused_cols = set(range(len(input_centroids))) - used_cols if 'used_cols' in locals() else set(range(len(input_centroids)))
            
            for col in unused_cols:
                encoding = face_encodings[col] if face_encodings is not None else None
                self.register(face_locations[col], encoding)
        
        for object_id in list(self.disappeared.keys()):
            if object_id not in [obj["face_location"] for obj in self.objects.values()]:
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects
    
    def _save_trajectory(self, object_id):
        if object_id in self.trajectories and len(self.trajectories[object_id]) > 1:
            if not os.path.exists("trajectories"):
                os.makedirs("trajectories")
            
            filename = f"trajectories/trajectory_{object_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            trajectory_data = {
                "object_id": object_id,
                "points": self.trajectories[object_id],
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "num_points": len(self.trajectories[object_id])
            }
            
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=4)
            
            print(f"Траектория для объекта {object_id} сохранена в {filename}")
    
    def get_trajectories(self):
        return dict(self.trajectories)


class RealTimeFaceTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система трекинга лиц в реальном времени")
        self.root.geometry("1200x800")
        
        self.video_source = 0  
        self.cap = None
        self.is_running = False
        self.tracker = None
        self.known_faces_loaded = False
        
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.setup_gui()
        
        self.tracker = FaceTracker(max_disappeared=30, max_distance=0.6, tracking_method="centroid")
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(left_frame)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(right_frame, text="Управление трекером лиц", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        self.start_button = ttk.Button(right_frame, text="Запуск трекера", 
                                      command=self.start_tracking)
        self.start_button.grid(row=1, column=0, pady=5, sticky=tk.EW)
        
        self.stop_button = ttk.Button(right_frame, text="Остановить", 
                                     command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=0, pady=5, sticky=tk.EW)
        
        source_frame = ttk.LabelFrame(right_frame, text="Источник видео", padding="10")
        source_frame.grid(row=3, column=0, pady=10, sticky=tk.EW)
        
        ttk.Label(source_frame, text="Источник:").grid(row=0, column=0, sticky=tk.W)
        
        self.source_var = tk.StringVar(value="0")
        sources = [("Веб-камера (по умолчанию)", "0"),
                  ("Веб-камера 2", "1"),
                  ("Видео файл", "file")]
        
        for i, (text, value) in enumerate(sources):
            ttk.Radiobutton(source_frame, text=text, variable=self.source_var, 
                           value=value).grid(row=i+1, column=0, sticky=tk.W)
        
        self.browse_button = ttk.Button(source_frame, text="Выбрать файл", 
                                       command=self.browse_video_file, state=tk.DISABLED)
        self.browse_button.grid(row=4, column=0, pady=5, sticky=tk.EW)
        
        self.source_var.trace('w', self.on_source_change)
        
        settings_frame = ttk.LabelFrame(right_frame, text="Настройки трекера", padding="10")
        settings_frame.grid(row=4, column=0, pady=10, sticky=tk.EW)
        
        ttk.Label(settings_frame, text="Метод трекинга:").grid(row=0, column=0, sticky=tk.W)
        self.tracking_method_var = tk.StringVar(value="centroid")
        ttk.Combobox(settings_frame, textvariable=self.tracking_method_var,
                    values=["centroid", "dlib"], state="readonly").grid(row=0, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Max disappeared:").grid(row=1, column=0, sticky=tk.W)
        self.max_disappeared_var = tk.IntVar(value=30)
        ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.max_disappeared_var,
                   width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Max distance:").grid(row=2, column=0, sticky=tk.W)
        self.max_distance_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(settings_frame, from_=0.1, to=1.0, increment=0.1,
                   textvariable=self.max_distance_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Button(settings_frame, text="Применить настройки", 
                  command=self.update_tracker_settings).grid(row=3, column=0, columnspan=2, pady=5)
        
        info_frame = ttk.LabelFrame(right_frame, text="Информация о трекинге", padding="10")
        info_frame.grid(row=5, column=0, pady=10, sticky=tk.EW)
        
        self.info_text = tk.Text(info_frame, height=10, width=30)
        self.info_text.grid(row=0, column=0)
        
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        traj_frame = ttk.LabelFrame(right_frame, text="Управление траекториями", padding="10")
        traj_frame.grid(row=6, column=0, pady=10, sticky=tk.EW)
        
        ttk.Button(traj_frame, text="Сохранить траектории", 
                  command=self.save_trajectories).grid(row=0, column=0, pady=5, sticky=tk.EW)
        
        ttk.Button(traj_frame, text="Очистить траектории", 
                  command=self.clear_trajectories).grid(row=1, column=0, pady=5, sticky=tk.EW)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def on_source_change(self, *args):
        if self.source_var.get() == "file":
            self.browse_button.config(state=tk.NORMAL)
        else:
            self.browse_button.config(state=tk.DISABLED)
    
    def browse_video_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_source = filename
            self.status_var.set(f"Выбран файл: {os.path.basename(filename)}")
    
    def update_tracker_settings(self):
        if self.tracker:
            self.tracker.max_disappeared = self.max_disappeared_var.get()
            self.tracker.max_distance = self.max_distance_var.get()
            self.tracker.tracking_method = self.tracking_method_var.get()
            self.status_var.set("Настройки трекера обновлены")
    
    def start_tracking(self):
        if not self.is_running:
            if self.source_var.get() == "file":
                source = self.video_source
            else:
                source = int(self.source_var.get())
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть видео поток")
                return
            
            self.update_tracker_settings()
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            self.is_running = True
            self.thread = threading.Thread(target=self.process_video, daemon=True)
            self.thread.start()
            
            self.update_gui()
            
            self.status_var.set("Трекинг запущен")
    
    def stop_tracking(self):
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_var.set("Трекинг остановлен")
        for object_id in self.tracker.trajectories.keys():
            self.tracker._save_trajectory(object_id)

    def process_video(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame, (800, 600))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame)
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            tracked_faces = self.tracker.update(face_locations, face_encodings)
            
            processed_frame = self.draw_tracking_results(frame, tracked_faces)
            
            if self.frame_queue.qsize() < 2:
                self.frame_queue.put(processed_frame)
        
        self.is_running = False
    
    def draw_tracking_results(self, frame, tracked_faces):
        display_frame = frame.copy()
        
        trajectories = self.tracker.get_trajectories()
        
        for object_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(display_frame, trajectory[i-1], trajectory[i], 
                            (0, 255, 0), 2)
        
        for object_id, face_data in tracked_faces.items():
            face_location = face_data["face_location"]
            centroid = face_data["centroid"]
            
            top, right, bottom, left = face_location
            
            cv2.rectangle(display_frame, (left, top), (right, bottom), 
                         (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"ID: {object_id}", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.circle(display_frame, centroid, 4, (0, 255, 0), -1)
        
        cv2.putText(display_frame, f"Обнаружено лиц: {len(tracked_faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def update_gui(self):
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=img)
                self.video_label.image = img
                
                self.update_info_panel()
        
        except Exception as e:
            print(f"Ошибка обновления GUI: {e}")
        
        if self.is_running:
            self.root.after(10, self.update_gui)
    
    def update_info_panel(self):
        if self.tracker:
            info_text = ""
            info_text += f"Текущих объектов: {len(self.tracker.objects)}\n"
            info_text += f"Всего объектов: {self.tracker.next_object_id}\n"
            info_text += f"Метод трекинга: {self.tracker.tracking_method}\n"
            info_text += f"Траекторий: {len(self.tracker.trajectories)}\n\n"
            
            for object_id, face_data in self.tracker.objects.items():
                centroid = face_data["centroid"]
                disappeared = self.tracker.disappeared[object_id]
                info_text += f"ID: {object_id} | Центр: {centroid} | Пропусков: {disappeared}\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
    
    def save_trajectories(self):
        if not self.tracker:
            return
        
        if not os.path.exists("trajectories"):
            os.makedirs("trajectories")
        
        for object_id in list(self.tracker.trajectories.keys()):
            self.tracker._save_trajectory(object_id)
        
        self.status_var.set("Траектории сохранены")
        messagebox.showinfo("Успех", "Все траектории сохранены в папку 'trajectories'")
    
    def clear_trajectories(self):
        if self.tracker:
            self.tracker.trajectories.clear()
            self.status_var.set("Траектории очищены")
    
    def on_closing(self):
        self.stop_tracking()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RealTimeFaceTrackerApp(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()


if __name__ == "__main__":
    main()