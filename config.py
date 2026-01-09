import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'faces_dataset')

    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')
    TRAIN_ANNOTATIONS = os.path.join(DATA_DIR, 'train', 'annotations.csv')

    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test', 'images')
    TEST_ANNOTATIONS = os.path.join(DATA_DIR, 'test', 'annotations.csv')

    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    DETECTIONS_DIR = os.path.join(OUTPUT_DIR, 'detections')  
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'face_detector.pkl')

    WINDOW_SIZE = (64, 64)      

    HOG_ORIENTATIONS = 9         
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)

    LBP_POINTS = 24              
    LBP_RADIUS = 3               
    LBP_METHOD = 'uniform'

    NEGATIVE_SAMPLES_PER_IMAGE = 25    
    NEGATIVE_IOU_THRESHOLD = 0.1        

    SVM_KERNEL = 'rbf'           
    SVM_C = 5.0                 
    SVM_GAMMA = 'scale'

    MAX_TRAIN_IMAGES = 400      
    MAX_TEST_IMAGES = 100
    MAX_DIMENSION = 640

    SCALE_FACTOR = 1.2           
    MIN_SCALE = 0.5
    MAX_SCALE = 1.0
    SLIDE_STEP = 12              

    DETECTION_THRESHOLD = 0.5    
    NMS_IOU_THRESHOLD = 0.3      
    EVAL_IOU_THRESHOLD = 0.35

    HARD_NEGATIVE_MINING = True  
    HARD_NEGATIVE_ROUNDS = 2     
    
    @staticmethod
    def create_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.DETECTIONS_DIR, exist_ok=True)
    
    @staticmethod
    def model_exists():
        return os.path.exists(Config.MODEL_PATH)

Config.create_directories()