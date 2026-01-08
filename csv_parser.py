import csv
import os

class AnnotationParser:
    
    def __init__(self, csv_path, images_dir):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.annotations = []
    
    def parse(self):

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Annotations not found: {self.csv_path}")
        
        image_dict = {}
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                img_name = row['image_name']
                img_path = os.path.join(self.images_dir, img_name)
                
                bbox = (
                    int(row['x0']),
                    int(row['y0']),
                    int(row['x1']),
                    int(row['y1'])
                )
                
                if img_name not in image_dict:
                    image_dict[img_name] = {
                        'image_name': img_name,
                        'image_path': img_path,
                        'width': int(row['width']),
                        'height': int(row['height']),
                        'bboxes': []
                    }
                
                image_dict[img_name]['bboxes'].append(bbox)
        
        self.annotations = list(image_dict.values())
        return self.annotations
    
    def get_stats(self):
        if not self.annotations:
            self.parse()
        
        total_images = len(self.annotations)
        total_faces = sum(len(ann['bboxes']) for ann in self.annotations)
        
        return {
            'total_images': total_images,
            'total_faces': total_faces,
            'avg_faces_per_image': total_faces / total_images if total_images > 0 else 0
        }

if __name__ == "__main__":
    from config import Config
    
    print("Testing Annotation Parser\n")
    
    print("Training set:")
    parser = AnnotationParser(Config.TRAIN_ANNOTATIONS, Config.TRAIN_IMAGES_DIR)
    parser.parse()
    stats = parser.get_stats()
    print(f"  Images: {stats['total_images']}")
    print(f"  Faces: {stats['total_faces']}")
    print(f"  Avg faces/image: {stats['avg_faces_per_image']:.2f}")
    
    print("\nTest set:")
    parser = AnnotationParser(Config.TEST_ANNOTATIONS, Config.TEST_IMAGES_DIR)
    parser.parse()
    stats = parser.get_stats()
    print(f"  Images: {stats['total_images']}")
    print(f"  Faces: {stats['total_faces']}")
    print(f"  Avg faces/image: {stats['avg_faces_per_image']:.2f}")