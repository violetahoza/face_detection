import os

class Config:
    TRAIN_IMAGES_DIR = r'C:\Users\hozas\Desktop\facultate\an4\Sem1\PRS\project\data\faces_dataset\train\images'
    TRAIN_ANNOTATIONS = r'C:\Users\hozas\Desktop\facultate\an4\Sem1\PRS\project\data\faces_dataset\train\annotations.csv'
    
    TEST_IMAGES_DIR = r'C:\Users\hozas\Desktop\facultate\an4\Sem1\PRS\project\data\faces_dataset\test\images'
    TEST_ANNOTATIONS = r'C:\Users\hozas\Desktop\facultate\an4\Sem1\PRS\project\data\faces_dataset\test\annotations.csv'
    
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'face_detector.pkl')
    
    WINDOW_SIZE = (80, 80)
    
    HOG_ORIENTATIONS = 12
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    LBP_POINTS = 8
    LBP_RADIUS = 2
    LBP_METHOD = 'uniform'
    
    NEGATIVE_SAMPLES_PER_IMAGE = 10
    NEGATIVE_IOU_THRESHOLD = 0.3
    
    SVM_KERNEL = 'linear'
    SVM_C = 1.0
    SVM_GAMMA = 'scale'  
    
    MAX_TRAIN_IMAGES = 500  
    MAX_TEST_IMAGES = 100   
    
    SCALE_FACTOR = 1.25
    MIN_SCALE = 0.4   
    MAX_SCALE = 1.0         
    SLIDE_STEP = 28          
    
    DETECTION_THRESHOLD = 2.0
    NMS_IOU_THRESHOLD = 0.5
    EVAL_IOU_THRESHOLD = 0.5
    
    @staticmethod
    def create_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    @staticmethod
    def model_exists():
        return os.path.exists(Config.MODEL_PATH)

Config.create_directories()