import cv2
import numpy as np
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
       
        orientations = Config.HOG_ORIENTATIONS 
        pixels_per_cell = Config.HOG_PIXELS_PER_CELL  
        cells_per_block = Config.HOG_CELLS_PER_BLOCK  
        
        image = image.astype(np.float64)
        
        # Compute gradients using Sobel operators
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1) 
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)  
        
        # Compute magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi)  
        angle[angle < 0] += 180 
        
        # Divide into cells and compute histograms
        cell_h, cell_w = pixels_per_cell
        n_cells_y = image.shape[0] // cell_h
        n_cells_x = image.shape[1] // cell_w
        
        # Create cell histograms
        cell_histograms = np.zeros((n_cells_y, n_cells_x, orientations))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extract cell
                y_start = i * cell_h
                y_end = y_start + cell_h
                x_start = j * cell_w
                x_end = x_start + cell_w
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ang = angle[y_start:y_end, x_start:x_end]
                
                # Compute histogram with bilinear interpolation
                hist = self._compute_cell_histogram(cell_mag, cell_ang, orientations)
                cell_histograms[i, j, :] = hist
        
        # Normalize histograms over blocks
        block_h, block_w = cells_per_block
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        
        normalized_blocks = []
        
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = cell_histograms[i:i+block_h, j:j+block_w, :]
                block = block.flatten()
                
                norm = np.sqrt(np.sum(block**2) + 1e-5)
                block = block / norm
                
                block = np.clip(block, 0, 0.2)
                
                norm = np.sqrt(np.sum(block**2) + 1e-5)
                block = block / norm
                
                normalized_blocks.append(block)
        
        hog_features = np.concatenate(normalized_blocks)
        
        return hog_features
    
    def _compute_cell_histogram(self, magnitudes, angles, n_bins):
        histogram = np.zeros(n_bins)
        bin_size = 180.0 / n_bins  
        
        for i in range(magnitudes.shape[0]):
            for j in range(magnitudes.shape[1]):
                mag = magnitudes[i, j]
                ang = angles[i, j]
                
                # Find which bins this gradient falls between
                bin_idx = ang / bin_size
                bin_low = int(np.floor(bin_idx)) % n_bins
                bin_high = int(np.ceil(bin_idx)) % n_bins
                
                # Bilinear interpolation weights
                weight_high = bin_idx - np.floor(bin_idx)
                weight_low = 1.0 - weight_high
                
                # Distribute vote between bins
                histogram[bin_low] += mag * weight_low
                histogram[bin_high] += mag * weight_high
        
        return histogram
    
    def extract_lbp(self, image):
        P = Config.LBP_POINTS  
        R = Config.LBP_RADIUS 
        method = Config.LBP_METHOD 
        
        h, w = image.shape
        lbp_image = np.zeros((h, w), dtype=np.float64)
        
        # Precompute neighbor coordinates
        angles = 2 * np.pi * np.arange(P) / P
        neighbor_y = -R * np.sin(angles)
        neighbor_x = R * np.cos(angles)
        
        # Process each pixel (skip border)
        for i in range(R, h - R):
            for j in range(R, w - R):
                center = image[i, j]
                
                # Get neighbors using bilinear interpolation
                binary_pattern = 0
                for p in range(P):
                    ny = i + neighbor_y[p]
                    nx = j + neighbor_x[p]
                    
                    neighbor_value = self._bilinear_interpolate(image, ny, nx)
                    
                    # Compare with center
                    if neighbor_value >= center:
                        binary_pattern |= (1 << p)
                
                # For uniform patterns, count transitions
                if method == 'uniform':
                    lbp_code = self._get_uniform_pattern(binary_pattern, P)
                else:
                    lbp_code = binary_pattern
                
                lbp_image[i, j] = lbp_code
        
        # Create histogram
        if method == 'uniform':
            # Uniform patterns: P*(P-1) + 2 patterns + 1 for non-uniform
            n_bins = P * (P - 1) + 3
        else:
            n_bins = 2 ** P
        
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize histogram
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def _bilinear_interpolate(self, image, y, x):
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1
        
        y0 = max(0, min(y0, image.shape[0] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x0 = max(0, min(x0, image.shape[1] - 1))
        x1 = max(0, min(x1, image.shape[1] - 1))
        
        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        
        # Interpolated value
        value = (wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1])
        
        return value
    
    def _get_uniform_pattern(self, pattern, P):

        # Count transitions (0->1 or 1->0)
        transitions = 0
        for i in range(P):
            bit_current = (pattern >> i) & 1
            bit_next = (pattern >> ((i + 1) % P)) & 1
            if bit_current != bit_next:
                transitions += 1
        
        # Uniform pattern has <= 2 transitions
        if transitions <= 2:
            # Count number of 1s (this is the uniform pattern code)
            return bin(pattern).count('1')
        else:
            # Non-uniform pattern: assign to last bin
            return P * (P - 1) + 2
    
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

