import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from csv_parser import AnnotationParser
from feature_extraction import FeatureExtractor
from config import Config

class FaceDetectorTrainer:
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        if Config.SVM_KERNEL == 'rbf':
            self.classifier = SVC(
                kernel='rbf',
                C=Config.SVM_C,
                gamma=Config.SVM_GAMMA, 
                probability=True,
                cache_size=500  
            )
        elif Config.SVM_KERNEL == 'linear':
            self.classifier = SVC(
                kernel='linear',
                C=Config.SVM_C,
                probability=True
            )
        else:
            self.classifier = SVC(
                kernel=Config.SVM_KERNEL,
                C=Config.SVM_C,
                probability=True
            )
    
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
    
    def generate_negative_sample(self, image, face_bboxes, max_attempts=150):
        h, w = image.shape[:2]
        window_w, window_h = Config.WINDOW_SIZE
        
        for _ in range(max_attempts):
            x = np.random.randint(0, max(1, w - window_w))
            y = np.random.randint(0, max(1, h - window_h))
            
            sample_w = window_w
            sample_h = window_h
            
            if x + sample_w > w or y + sample_h > h:
                continue
            
            candidate_box = (x, y, x + sample_w, y + sample_h)
            
            is_valid = True
            for face_box in face_bboxes:
                iou = self.calculate_iou(candidate_box, face_box)
                
                if iou > 0.2: 
                    is_valid = False
                    break
            
            if not is_valid:
                continue
            
            crop = image[y:y+sample_h, x:x+sample_w]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

            std_dev = np.std(crop_gray)
            if std_dev < 3:  
                continue
            
            return crop
        
        return None
    
    def collect_training_samples(self, annotations, max_images=None):

        if max_images:
            annotations = annotations[:max_images]
        
        print(f"\n📦 Collecting samples from {len(annotations)} images...")
        
        positive_features = []
        negative_features = []
        
        for idx, ann in enumerate(annotations, 1):
            if idx % 50 == 0:
                print(f"  Processing {idx}/{len(annotations)}...")
            
            img_path = ann['image_path']
            
            if not os.path.exists(img_path):
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            for bbox in ann['bboxes']:
                x0, y0, x1, y1 = bbox
                face_crop = image[y0:y1, x0:x1]
                
                if face_crop.size == 0:
                    continue
                
                features = self.feature_extractor.extract_features(face_crop)
                positive_features.append(features)
                
                face_flipped = cv2.flip(face_crop, 1)
                features_flip = self.feature_extractor.extract_features(face_flipped)
                positive_features.append(features_flip)
                
            n_negatives = len(ann['bboxes']) * 10
            
            collected = 0
            for _ in range(n_negatives * 2): 
                neg_sample = self.generate_negative_sample(image, ann['bboxes'])
                if neg_sample is not None:
                    features = self.feature_extractor.extract_features(neg_sample)
                    negative_features.append(features)
                    collected += 1
                    if collected >= n_negatives:
                        break
        
        print(f"\n✓ Collected {len(positive_features)} positive samples (faces)")
        print(f"✓ Collected {len(negative_features)} negative samples (non-faces)")
        
        if len(positive_features) == 0 or len(negative_features) == 0:
            raise ValueError("No samples collected! Check your data paths.")
        
        X = np.array(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        return X, y
    
    def train(self, train_annotations_path=None, train_images_dir=None, max_images=None):
        print("🚀 TRAINING FACE DETECTOR")
        print("=" * 60)
        
        train_annotations_path = train_annotations_path or Config.TRAIN_ANNOTATIONS
        train_images_dir = train_images_dir or Config.TRAIN_IMAGES_DIR
        max_images = max_images or Config.MAX_TRAIN_IMAGES
        
        parser = AnnotationParser(train_annotations_path, train_images_dir)
        annotations = parser.parse()
        
        stats = parser.get_stats()
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total faces: {stats['total_faces']}")
        print(f"  Using: {min(max_images, stats['total_images'])} images")
        
        X, y = self.collect_training_samples(annotations, max_images)
        
        print(f"\n📊 Training Data:")
        print(f"  Total samples: {len(X)}")
        print(f"  Positive (faces): {np.sum(y == 1)}")
        print(f"  Negative (non-faces): {np.sum(y == 0)}")
        print(f"  Feature dimension: {X.shape[1]}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🔄 Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"\n🎯 Training SVM classifier...")
        print(f"  Kernel: {Config.SVM_KERNEL}")
        print(f"  C: {Config.SVM_C}")
        
        self.classifier.fit(X_train_scaled, y_train)
        
        train_acc = self.classifier.score(X_train_scaled, y_train)
        val_acc = self.classifier.score(X_val_scaled, y_val)
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"  Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.save_model()
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'n_samples': len(X),
            'feature_dim': X.shape[1]
        }
    
    def save_model(self, path=None):
        path = path or Config.MODEL_PATH
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved: {path}")
    
    @staticmethod
    def load_model(path=None):
        path = path or Config.MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = FaceDetectorTrainer()
        trainer.classifier = model_data['classifier']
        trainer.scaler = model_data['scaler']
        trainer.feature_extractor = model_data['feature_extractor']
        
        return trainer

if __name__ == "__main__":
    trainer = FaceDetectorTrainer()
    trainer.train()