import os
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, simpledialog
from PIL import Image, ImageTk
import cv2

from config import Config
from training import FaceDetectorTrainer
from detector import FaceDetector
from evaluator import ModelEvaluator
from csv_parser import AnnotationParser


class ModernFaceDetectionGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection System ")
        self.root.geometry("900x750")
        
        self.trainer = None
        self.detector = None
        self.model_loaded = False
        
        self.message_queue = queue.Queue()
        
        self.create_ui()
        self.process_messages()
        
        if Config.model_exists():
            try:
                trained_model = FaceDetectorTrainer.load_model()
                self.detector = FaceDetector(trained_model)
                self.model_loaded = True
                self.log("✓ Auto-loaded: face_detector.pkl")
                self.update_status()
            except Exception as e:
                self.log(f"⚠️ Failed to auto-load model: {e}")
    
    def create_ui(self):
        title_label = tk.Label(
            self.root, 
            text="🎯 Face Detection System",
            font=("Arial", 20, "bold"),
            fg="darkblue"
        )
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(
            self.root,
            text="Classical Computer Vision: HOG + LBP + SVM",
            font=("Arial", 10, "italic"),
            fg="gray"
        )
        subtitle_label.pack()
        
        self.create_status_panel()
        self.create_button_panel()
        self.create_progress_panel()
        self.create_console_panel()
        
        self.log(f"✓ System ready")
        self.log(f"✓ Output directory: {Config.OUTPUT_DIR}\n")
    
    def create_status_panel(self):
        status_frame = tk.LabelFrame(
            self.root,
            text="System Status",
            font=("Arial", 10, "bold")
        )
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_text = tk.Text(
            status_frame,
            height=5,
            width=80,
            bg="lightyellow",
            font=("Consolas", 9),
            relief=tk.FLAT
        )
        self.status_text.pack(padx=10, pady=10)
        self.update_status()
    
    def create_button_panel(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)
        
        tk.Button(
            button_frame,
            text="⚙️ View Config",
            command=self.view_config,
            width=20,
            height=2,
            bg="lightblue",
            font=("Arial", 10)
        ).grid(row=0, column=0, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="📊 Dataset Info",
            command=self.view_dataset_info,
            width=20,
            height=2,
            bg="lightcyan",
            font=("Arial", 10)
        ).grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_train = tk.Button(
            button_frame,
            text="🚀 Train Model",
            command=self.train_threaded,
            width=20,
            height=2,
            bg="lightgreen",
            font=("Arial", 10)
        )
        self.btn_train.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_detect = tk.Button(
            button_frame,
            text="🔍 Detect Single Image",
            command=self.detect_threaded,
            width=20,
            height=2,
            bg="lightcoral",
            font=("Arial", 10)
        )
        self.btn_detect.grid(row=1, column=0, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="📁 Batch Detection",
            command=self.batch_threaded,
            width=20,
            height=2,
            bg="lightpink",
            font=("Arial", 10)
        ).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="📈 Evaluate Model",
            command=self.evaluate_threaded,
            width=20,
            height=2,
            bg="plum",
            font=("Arial", 10)
        ).grid(row=1, column=2, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="📂 Open Output Folder",
            command=self.open_output,
            width=31,
            height=2,
            bg="lightsteelblue",
            font=("Arial", 10)
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="❌ Exit",
            command=self.root.quit,
            width=20,
            height=2,
            bg="lightgray",
            font=("Arial", 10)
        ).grid(row=2, column=2, padx=5, pady=5)
    
    def create_progress_panel(self):
        self.progress_frame = tk.Frame(self.root)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=("Arial", 9)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=500
        )
    
    def create_console_panel(self):
        log_frame = tk.LabelFrame(
            self.root,
            text="Console Output",
            font=("Arial", 10, "bold")
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            bg="black",
            fg="lightgreen",
            font=("Consolas", 9),
            relief=tk.FLAT
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_status(self):
        self.status_text.delete(1.0, tk.END)
        
        status = f"Method: HOG + LBP + SVM\n"
        status += f"Window Size: {Config.WINDOW_SIZE}\n"
        status += f"Model Trained: {'✓' if Config.model_exists() else '✗'}\n"
        
        if self.model_loaded and self.detector:
            status += f"Model Loaded: ✓ (face_detector.pkl)\n"
        else:
            status += f"Model Loaded: ✗\n"
        
        status += f"Output: {Config.OUTPUT_DIR}"
        
        self.status_text.insert(1.0, status)
    
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def show_progress(self, msg="Processing..."):
        self.progress_label.config(text=msg)
        self.progress_bar.pack(pady=5)
        self.progress_bar.start(10)
    
    def hide_progress(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
    
    def process_messages(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "log":
                    self.log(data)
                elif msg_type == "status":
                    self.update_status()
                elif msg_type == "progress_show":
                    self.show_progress(data)
                elif msg_type == "progress_hide":
                    self.hide_progress()
                elif msg_type == "msgbox":
                    title, msg, typ = data
                    if typ == "info":
                        messagebox.showinfo(title, msg)
                    elif typ == "error":
                        messagebox.showerror(title, msg)
                    elif typ == "warning":
                        messagebox.showwarning(title, msg)
                elif msg_type == "result_comparison":
                    self.show_comparison_view(*data)
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_messages)
    
    def view_config(self):
        self.log("\n⚙️ CONFIGURATION\n" + "="*60)
        self.log(f"Train Images: {Config.TRAIN_IMAGES_DIR}")
        self.log(f"Test Images: {Config.TEST_IMAGES_DIR}")
        self.log(f"Model: {Config.MODEL_PATH}")
        self.log(f"Output: {Config.OUTPUT_DIR}")
        self.log(f"\n🎯 Feature Extraction:")
        self.log(f"  Window Size: {Config.WINDOW_SIZE}")
        self.log(f"  HOG Orientations: {Config.HOG_ORIENTATIONS}")
        self.log(f"  HOG Pixels/Cell: {Config.HOG_PIXELS_PER_CELL}")
        self.log(f"  HOG Cells/Block: {Config.HOG_CELLS_PER_BLOCK}")
        self.log(f"  LBP Points: {Config.LBP_POINTS}")
        self.log(f"  LBP Radius: {Config.LBP_RADIUS}")
        self.log(f"\n🚀 Training:")
        self.log(f"  Max Train Images: {Config.MAX_TRAIN_IMAGES}")
        self.log(f"  SVM Kernel: {Config.SVM_KERNEL}")
        self.log(f"  SVM C: {Config.SVM_C}")
        self.log(f"\n🔍 Detection:")
        self.log(f"  Scale Factor: {Config.SCALE_FACTOR}")
        self.log(f"  Slide Step: {Config.SLIDE_STEP}")
        self.log(f"  Detection Threshold: {Config.DETECTION_THRESHOLD}")
        self.log(f"  NMS IoU Threshold: {Config.NMS_IOU_THRESHOLD}\n")
    
    def view_dataset_info(self):
        self.log("\n📊 DATASET INFORMATION\n" + "="*60)
        
        self.log("\n📁 Training Set:")
        try:
            parser = AnnotationParser(Config.TRAIN_ANNOTATIONS, Config.TRAIN_IMAGES_DIR)
            parser.parse()
            stats = parser.get_stats()
            self.log(f"  Images: {stats['total_images']}")
            self.log(f"  Faces: {stats['total_faces']}")
            self.log(f"  Avg faces/image: {stats['avg_faces_per_image']:.2f}")
        except Exception as e:
            self.log(f"  ❌ Error: {e}")
        
        self.log("\n📁 Test Set:")
        try:
            parser = AnnotationParser(Config.TEST_ANNOTATIONS, Config.TEST_IMAGES_DIR)
            parser.parse()
            stats = parser.get_stats()
            self.log(f"  Images: {stats['total_images']}")
            self.log(f"  Faces: {stats['total_faces']}")
            self.log(f"  Avg faces/image: {stats['avg_faces_per_image']:.2f}\n")
        except Exception as e:
            self.log(f"  ❌ Error: {e}\n")
    
    def train_threaded(self):
        if Config.model_exists():
            if not messagebox.askyesno(
                "Overwrite Model",
                "Existing model found. Overwrite?"
            ):
                return
        
        max_images = simpledialog.askinteger(
            "Training Images",
            f"Number of training images (default: {Config.MAX_TRAIN_IMAGES}):",
            initialvalue=Config.MAX_TRAIN_IMAGES,
            minvalue=10,
            maxvalue=1000
        )
        
        if not max_images:
            return
        
        if not messagebox.askyesno(
            "Start Training",
            f"Train with {max_images} images?\nThis may take several minutes."
        ):
            return
        
        self.btn_train.config(state=tk.DISABLED, text="⏳ Training...")
        threading.Thread(target=self._train_worker, args=(max_images,), daemon=True).start()
    
    def _train_worker(self, max_images):
        try:
            self.message_queue.put(("progress_show", "Training model..."))
            self.message_queue.put(("log", "\n🚀 TRAINING STARTED\n" + "="*60))
            
            trainer = FaceDetectorTrainer()
            results = trainer.train(max_images=max_images)
            
            trained_model = FaceDetectorTrainer.load_model()
            self.detector = FaceDetector(trained_model)
            self.model_loaded = True
            
            self.message_queue.put(("log", "\n✓ Training complete!"))
            self.message_queue.put(("log", "✓ Model auto-loaded"))
            self.message_queue.put(("status", None))
            
            result_msg = (
                f"Training Complete!\n\n"
                f"Training Accuracy: {results['train_accuracy']*100:.2f}%\n"
                f"Validation Accuracy: {results['val_accuracy']*100:.2f}%\n"
                f"Total Samples: {results['n_samples']}\n"
                f"Feature Dimension: {results['feature_dim']}"
            )
            
            self.message_queue.put(("msgbox", ("Success", result_msg, "info")))
        
        except Exception as e:
            self.message_queue.put(("log", f"\n✗ Error: {str(e)}"))
            self.message_queue.put(("msgbox", ("Error", f"Training error:\n{str(e)}", "error")))
        
        finally:
            self.message_queue.put(("progress_hide", None))
            self.btn_train.config(state=tk.NORMAL, text="🚀 Train Model")
    
    def detect_threaded(self):
        if not self.model_loaded:
            messagebox.showwarning("No Model", "Please train a model first!")
            return
        
        img_path = filedialog.askopenfilename(
            initialdir=Config.TEST_IMAGES_DIR,
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not img_path:
            return
        
        self.btn_detect.config(state=tk.DISABLED, text="⏳ Detecting...")
        threading.Thread(target=self._detect_worker, args=(img_path,), daemon=True).start()
    
    def _detect_worker(self, img_path):
        try:
            self.message_queue.put(("progress_show", "Detecting faces..."))
            self.message_queue.put(("log", f"\n🔍 Processing: {Path(img_path).name}"))
            
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            detections = self.detector.detect(image, verbose=True)
            
            self.message_queue.put(("log", f"\n✓ Found {len(detections)} face(s)"))
            
            for i, det in enumerate(detections, 1):
                bbox = det['bbox']
                score = det['score']
                self.message_queue.put(("log", f"  Face {i}: score={score:.3f}, bbox={bbox}"))
            
            result_image = self.detector.draw_detections(image, detections)
            output_path = os.path.join(Config.OUTPUT_DIR, f"detected_{Path(img_path).name}")
            cv2.imwrite(output_path, result_image)
            
            self.message_queue.put(("log", f"💾 Saved: {Path(output_path).name}\n"))
            self.message_queue.put(("result_comparison", (img_path, output_path, len(detections))))
            self.message_queue.put(("msgbox", ("Success", f"Detection complete!\nFound {len(detections)} face(s)", "info")))
        
        except Exception as e:
            self.message_queue.put(("log", f"✗ Error: {str(e)}\n"))
            self.message_queue.put(("msgbox", ("Error", f"Detection failed:\n{str(e)}", "error")))
        
        finally:
            self.message_queue.put(("progress_hide", None))
            self.btn_detect.config(state=tk.NORMAL, text="🔍 Detect Single Image")
    
    def batch_threaded(self):
        if not self.model_loaded:
            messagebox.showwarning("No Model", "Please train a model first!")
            return
        
        folder = filedialog.askdirectory(
            initialdir=Config.TEST_IMAGES_DIR,
            title="Select Image Folder"
        )
        
        if not folder:
            return
        
        max_images = simpledialog.askinteger(
            "Batch Detection",
            "How many images to process?\n(max 100)",
            initialvalue=10,
            minvalue=1,
            maxvalue=100
        )
        
        if not max_images:
            return
        
        threading.Thread(target=self._batch_worker, args=(folder, max_images), daemon=True).start()
    
    def _batch_worker(self, folder, max_images):
        try:
            self.message_queue.put(("progress_show", f"Processing {max_images} images..."))
            
            images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            images = images[:max_images]
            
            if not images:
                self.message_queue.put(("log", f"\n⚠️ No images found in {folder}\n"))
                self.message_queue.put(("msgbox", ("Warning", "No images found!", "warning")))
                return
            
            self.message_queue.put(("log", f"\n📁 BATCH DETECTION: {len(images)} images\n"))
            
            total_faces = 0
            processed = 0
            
            for i, img_name in enumerate(images, 1):
                self.message_queue.put(("log", f"[{i}/{len(images)}] {img_name}"))
                
                img_path = os.path.join(folder, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    self.message_queue.put(("log", "  ⚠️ Failed to load"))
                    continue
                
                detections = self.detector.detect(image, verbose=False)
                total_faces += len(detections)
                processed += 1
                
                self.message_queue.put(("log", f"  → {len(detections)} face(s)"))
                
                result_image = self.detector.draw_detections(image, detections)
                output_path = os.path.join(Config.OUTPUT_DIR, f"detected_{img_name}")
                cv2.imwrite(output_path, result_image)
            
            self.message_queue.put(("log", f"\n✓ Batch complete: {total_faces} faces in {processed} images\n"))
            self.message_queue.put(("msgbox", ("Success", f"Processed {processed} images\nFound {total_faces} faces total!", "info")))
        
        except Exception as e:
            self.message_queue.put(("log", f"✗ Error: {str(e)}\n"))
            self.message_queue.put(("msgbox", ("Error", f"Batch detection failed:\n{str(e)}", "error")))
        
        finally:
            self.message_queue.put(("progress_hide", None))
    
    def evaluate_threaded(self):
        if not self.model_loaded:
            messagebox.showwarning("No Model", "Please train a model first!")
            return
        
        if not messagebox.askyesno(
            "Evaluate Model",
            "Run full evaluation on test set?\nThis may take several minutes."
        ):
            return
        
        threading.Thread(target=self._evaluate_worker, daemon=True).start()
    
    def _evaluate_worker(self):
        try:
            self.message_queue.put(("progress_show", "Evaluating model..."))
            self.message_queue.put(("log", "\n📈 EVALUATION STARTED\n" + "="*60))
            
            evaluator = ModelEvaluator(self.detector)
            results = evaluator.evaluate()
            
            self.message_queue.put(("log", "\n✓ Evaluation complete!\n"))
            
            result_msg = (
                f"Evaluation Results:\n\n"
                f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n"
                f"Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)\n"
                f"F1-Score: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n\n"
                f"True Positives: {results['true_positives']}\n"
                f"False Positives: {results['false_positives']}\n"
                f"False Negatives: {results['false_negatives']}\n\n"
                f"Results saved in: {Config.OUTPUT_DIR}/"
            )
            
            self.message_queue.put(("msgbox", ("Evaluation Complete", result_msg, "info")))
        
        except Exception as e:
            self.message_queue.put(("log", f"✗ Error: {str(e)}\n"))
            self.message_queue.put(("msgbox", ("Error", f"Evaluation failed:\n{str(e)}", "error")))
        
        finally:
            self.message_queue.put(("progress_hide", None))
    
    def open_output(self):
        import subprocess
        import platform
        
        output_dir = Config.OUTPUT_DIR
        
        if platform.system() == "Windows":
            os.startfile(output_dir)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])
    
    def show_comparison_view(self, original_path, detected_path, num_faces):
        try:
            result_window = tk.Toplevel(self.root)
            result_window.title(f"Detection Results - {num_faces} face(s) found")
            result_window.geometry("1200x700")
            
            main_frame = tk.Frame(result_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Original image
            left_frame = tk.LabelFrame(main_frame, text="Original Image", font=("Arial", 11, "bold"))
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            original_img = Image.open(original_path)
            original_img.thumbnail((550, 550), Image.Resampling.LANCZOS)
            original_photo = ImageTk.PhotoImage(original_img)
            
            original_label = tk.Label(left_frame, image=original_photo)
            original_label.image = original_photo
            original_label.pack(padx=10, pady=10)
            
            original_info = tk.Label(
                left_frame,
                text=f"Original: {Path(original_path).name}",
                font=("Arial", 9)
            )
            original_info.pack(pady=5)
            
            right_frame = tk.LabelFrame(main_frame, text=f"Detected - {num_faces} Face(s)", font=("Arial", 11, "bold"), fg="darkgreen")
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
            
            detected_img = Image.open(detected_path)
            detected_img.thumbnail((550, 550), Image.Resampling.LANCZOS)
            detected_photo = ImageTk.PhotoImage(detected_img)
            
            detected_label = tk.Label(right_frame, image=detected_photo)
            detected_label.image = detected_photo
            detected_label.pack(padx=10, pady=10)
            
            detected_info = tk.Label(
                right_frame,
                text=f"Detected: {Path(detected_path).name}",
                font=("Arial", 9),
                fg="darkgreen"
            )
            detected_info.pack(pady=5)
            
            bottom_frame = tk.Frame(result_window, bg="lightyellow")
            bottom_frame.pack(fill=tk.X, padx=10, pady=5)
            
            info_text = tk.Label(
                bottom_frame,
                text=f"✓ Detection Complete | Found {num_faces} face(s) using HOG+LBP+SVM",
                font=("Arial", 10, "bold"),
                bg="lightyellow",
                fg="darkgreen"
            )
            info_text.pack(pady=10)
            
            button_frame = tk.Frame(result_window)
            button_frame.pack(pady=10)
            
            tk.Button(
                button_frame,
                text="📂 Open Output Folder",
                command=self.open_output,
                width=20,
                height=2,
                bg="lightblue"
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                button_frame,
                text="Close",
                command=result_window.destroy,
                width=20,
                height=2,
                bg="lightgray"
            ).pack(side=tk.LEFT, padx=5)
        
        except Exception as e:
            self.log(f"⚠️ Could not display comparison: {str(e)}")


def main():
    root = tk.Tk()
    app = ModernFaceDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()