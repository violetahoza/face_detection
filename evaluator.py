import os
import cv2
from csv_parser import AnnotationParser
from detector import FaceDetector
from config import Config

class ModelEvaluator:
    
    def __init__(self, detector):
        self.detector = detector
    
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
    
    def evaluate(self, test_annotations_path=None, test_images_dir=None, 
                 max_images=None, iou_threshold=None, detection_threshold=None):
       
        print("\n📈 EVALUATING MODEL")
        print("=" * 60)
        
        test_annotations_path = test_annotations_path or Config.TEST_ANNOTATIONS
        test_images_dir = test_images_dir or Config.TEST_IMAGES_DIR
        max_images = max_images or Config.MAX_TEST_IMAGES
        iou_threshold = iou_threshold or Config.EVAL_IOU_THRESHOLD
        detection_threshold = detection_threshold or Config.DETECTION_THRESHOLD
        
        parser = AnnotationParser(test_annotations_path, test_images_dir)
        annotations = parser.parse()
        
        if max_images:
            annotations = annotations[:max_images]
        
        stats = parser.get_stats()
        print(f"\n📊 Test Set:")
        print(f"  Images: {len(annotations)}")
        print(f"  Faces: {sum(len(ann['bboxes']) for ann in annotations)}")
        
        print(f"\n🔍 Detection Parameters:")
        print(f"  Detection threshold: {detection_threshold}")
        print(f"  IoU threshold: {iou_threshold}")
        
        print(f"\n🔄 Running detection...")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_gt_faces = 0
        total_det_faces = 0
        
        for idx, ann in enumerate(annotations, 1):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(annotations)}")
            
            img_path = ann['image_path']
            
            if not os.path.exists(img_path):
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            gt_boxes = ann['bboxes']
            total_gt_faces += len(gt_boxes)
            
            detections = self.detector.detect(
                image,
                detection_threshold=detection_threshold,
                verbose=False
            )
            det_boxes = [d['bbox'] for d in detections]
            total_det_faces += len(det_boxes)
            
            matched_gt = set()
            matched_det = set()
            
            for i, det_box in enumerate(det_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    matched_det.add(i)
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 Detection Statistics:")
        print(f"  Ground truth faces: {total_gt_faces}")
        print(f"  Detected faces: {total_det_faces}")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  F1-Score: {f1_score:.4f} ({f1_score * 100:.2f}%)")
        
        print(f"\n💡 Interpretation:")
        print(f"  Precision: {precision*100:.1f}% of detections are correct faces")
        print(f"  Recall: {recall*100:.1f}% of actual faces were detected")
        print(f"  F1-Score: Overall detection quality = {f1_score*100:.1f}%")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_gt_faces': total_gt_faces,
            'total_det_faces': total_det_faces
        }

if __name__ == "__main__":
    from training import FaceDetectorTrainer
    
    if not Config.model_exists():
        print("❌ No trained model found. Please run training first.")
        print("   Run: python trainer.py")
        exit(1)
    
    print("Loading trained model...")
    trained_model = FaceDetectorTrainer.load_model()
    
    detector = FaceDetector(trained_model)
    
    evaluator = ModelEvaluator(detector)
    results = evaluator.evaluate()