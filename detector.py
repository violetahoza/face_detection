import cv2
import numpy as np
from config import Config

class FaceDetector:
   
    def __init__(self, trained_model):
        self.classifier = trained_model.classifier
        self.scaler = trained_model.scaler
        self.feature_extractor = trained_model.feature_extractor
        self.window_size = Config.WINDOW_SIZE
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)
        
        if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
            return 0.0
        
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def non_max_suppression(self, detections, iou_threshold=None):
        if not detections:
            return []
        
        iou_threshold = iou_threshold or Config.NMS_IOU_THRESHOLD
        
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            filtered = []
            for det in detections:
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def detect(self, image, detection_threshold=None, verbose=False):

        detection_threshold = detection_threshold or Config.DETECTION_THRESHOLD

        h, w = image.shape[:2]

        MAX_DIMENSION = getattr(Config, 'MAX_DIMENSION', 640)

        if max(w, h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if verbose:
                print(f"  Resizing {w}x{h} → {new_w}x{new_h} for speed...")
            
            resized_image = cv2.resize(image, (new_w, new_h))
            detections_small = self._detect_internal(resized_image, detection_threshold, verbose)
            
            detections = []
            for det in detections_small:
                x0, y0, x1, y1 = det['bbox']
                detections.append({
                    'bbox': (int(x0 / scale), int(y0 / scale), 
                            int(x1 / scale), int(y1 / scale)),
                    'score': det['score']
                })
            
            return detections
        else:
            return self._detect_internal(image, detection_threshold, verbose)
    
    def _detect_internal(self, image, detection_threshold, verbose):
        
        detections = []
        h, w = image.shape[:2]
        
        if verbose:
            print(f"  Detecting faces in {w}x{h} image...")
        
        window_count = 0
        
        scale = Config.MAX_SCALE
        while scale >= Config.MIN_SCALE:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            
            if scaled_w < self.window_size[0] or scaled_h < self.window_size[1]:
                scale /= Config.SCALE_FACTOR
                continue
            
            scaled_image = cv2.resize(image, (scaled_w, scaled_h))
            
            for y in range(0, scaled_h - self.window_size[1], Config.SLIDE_STEP):
                for x in range(0, scaled_w - self.window_size[0], Config.SLIDE_STEP):
                    window_count += 1
                    
                    window = scaled_image[y:y+self.window_size[1], x:x+self.window_size[0]]
                    
                    features = self.feature_extractor.extract_features(window)
                    features_scaled = self.scaler.transform([features])
                    
                    score = self.classifier.decision_function(features_scaled)[0]
                    
                    if score > detection_threshold:
                        x0 = int(x / scale)
                        y0 = int(y / scale)
                        x1 = int((x + self.window_size[0]) / scale)
                        y1 = int((y + self.window_size[1]) / scale)
                        
                        detections.append({
                            'bbox': (x0, y0, x1, y1),
                            'score': float(score)
                        })
            
            scale /= Config.SCALE_FACTOR
        
        if verbose:
            print(f"    Windows checked: {window_count}")
            print(f"    Candidates found: {len(detections)}")
        
        detections = self.non_max_suppression(detections)
        
        if verbose:
            print(f"    After NMS: {len(detections)} faces")
        
        return detections
    
    def draw_detections(self, image, detections):
        result = image.copy()
        
        for det in detections:
            x0, y0, x1, y1 = det['bbox']
            score = det['score']
            cv2.rectangle(result, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = f"Face {score:.2f}"
            cv2.putText(result, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result

if __name__ == "__main__":
    import os
    from training import FaceDetectorTrainer
    from config import Config
    
    if not Config.model_exists():
        print("❌ No trained model found. Please run training first.")
        exit(1)
    
    print("Loading trained model...")
    trained_model = FaceDetectorTrainer.load_model()
    
    detector = FaceDetector(trained_model)
    
    test_image_dir = Config.TEST_IMAGES_DIR
    test_images = [f for f in os.listdir(test_image_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if test_images:
        test_image_path = os.path.join(test_image_dir, test_images[0])
        print(f"\nTesting on: {test_images[0]}")
        
        import time
        start = time.time()
        
        image = cv2.imread(test_image_path)
        detections = detector.detect(image, verbose=True)
        
        elapsed = time.time() - start
        
        print(f"\n✓ Detected {len(detections)} face(s) in {elapsed:.2f} seconds")
        
        result = detector.draw_detections(image, detections)
        output_path = os.path.join(Config.DETECTIONS_DIR, f"detected_{test_images[0]}")
        cv2.imwrite(output_path, result)
        print(f"💾 Result saved: {output_path}")