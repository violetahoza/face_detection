import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from config import Config

class FeatureExtractor:
   
    def __init__(self):
        self.window_size = Config.WINDOW_SIZE
    
    def preprocess_image(self, image):
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        resized = cv2.resize(gray, self.window_size)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        normalized = clahe.apply(resized)
        denoised = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        return denoised
    
    def extract_hog(self, image):
        
        features = hog(
            image,
            orientations=Config.HOG_ORIENTATIONS,
            pixels_per_cell=Config.HOG_PIXELS_PER_CELL,
            cells_per_block=Config.HOG_CELLS_PER_BLOCK,
            visualize=False,
            feature_vector=True,
            block_norm='L2-Hys'
        )
        return features
    
    def extract_lbp(self, image):
       
        lbp = local_binary_pattern(
            image,
            P=Config.LBP_POINTS,
            R=Config.LBP_RADIUS,
            method=Config.LBP_METHOD
        )
        
        n_bins = Config.LBP_POINTS * (Config.LBP_POINTS - 1) + 3
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins)
        )
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7) 
        
        return hist
    
    def extract_features(self, image):
        preprocessed = self.preprocess_image(image)
        hog_features = self.extract_hog(preprocessed)
        lbp_features = self.extract_lbp(preprocessed)
        features = np.concatenate([hog_features, lbp_features])
        return features
    
    def get_feature_dimension(self):
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        features = self.extract_features(dummy_image)
        return len(features)

if __name__ == "__main__":
    print("Testing Feature Extractor\n")
    
    extractor = FeatureExtractor()
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("Extracting features...")
    features = extractor.extract_features(test_image)
    
    print(f"✓ Feature vector dimension: {len(features)}")
    print(f"✓ Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"✓ Feature mean: {features.mean():.4f}")
    
    preprocessed = extractor.preprocess_image(test_image)
    print(f"\n✓ Preprocessed image shape: {preprocessed.shape}")
    print(f"✓ Expected shape: {Config.WINDOW_SIZE}")