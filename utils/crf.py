import os
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DenseCRF:
    def __init__(self):
        # Default CRF parameters
        self.gauss_sxy = 15
        self.gauss_compat = 30
        self.bilat_sxy = 10
        self.bilat_srgb = 20
        self.bilat_compat = 50
        self.n_infer = 10

    def load_config(self, path):
        """Load CRF parameters from a configuration file."""
        if os.path.exists(path):
            try:
                config = np.load(path)
                if config.shape[0] >= 6:
                    self.gauss_sxy, self.gauss_compat, self.bilat_sxy, self.bilat_srgb, self.bilat_compat, self.n_infer = config[:6]
                    logger.info(f"Loaded CRF config from {path}: gauss_sxy={self.gauss_sxy}, gauss_compat={self.gauss_compat}, "
                                f"bilat_sxy={self.bilat_sxy}, bilat_srgb={self.bilat_srgb}, bilat_compat={self.bilat_compat}, "
                                f"n_infer={self.n_infer}")
                else:
                    logger.warning(f"Invalid CRF config file {path}: expected at least 6 values, got {config.shape[0]}. Using defaults.")
            except Exception as e:
                logger.error(f"Failed to load CRF config from {path}: {str(e)}. Using defaults.")
        else:
            logger.warning(f"CRF config file {path} does not exist - using defaults")

    def process(self, probs, images):
        """
        Apply Dense CRF to refine probability maps.
        
        Args:
            probs (np.ndarray): Probability maps with shape [batch_size, num_classes, height, width]
            images (np.ndarray): Input images with shape [batch_size, height, width, 3]
        
        Returns:
            tuple: (maxconf_crf, crf)
                - maxconf_crf: Argmax of CRF probabilities, shape [batch_size, height, width]
                - crf: Refined probabilities, shape [batch_size, num_classes, height, width]
        """
        # Validate input shapes
        if probs.ndim != 4:
            raise ValueError(f"Expected 4D probs array, got shape {probs.shape}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected 4D images array with 3 channels, got shape {images.shape}")
        
        batch_size, num_classes, height, width = probs.shape
        if images.shape[:3] != (batch_size, height, width):
            raise ValueError(f"Spatial dimensions mismatch: probs {probs.shape[1:]} vs images {images.shape[:3]}")

        logger.debug(f"Processing CRF: batch_size={batch_size}, num_classes={num_classes}, height={height}, width={width}")

        # Initialize output arrays
        crf = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)
        maxconf_crf = np.zeros((batch_size, height, width), dtype=np.int32)

        for i in range(batch_size):
            prob_sample = probs[i]  # Shape: [num_classes, height, width]
            image_sample = images[i]  # Shape: [height, width, 3]

            if prob_sample.shape != (num_classes, height, width):
                logger.error(f"Invalid prob_sample shape for image {i}: expected ({num_classes}, {height}, {width}), got {prob_sample.shape}")
                continue
            if image_sample.shape != (height, width, 3):
                logger.error(f"Invalid image_sample shape for image {i}: expected ({height}, {width}, 3), got {image_sample.shape}")
                continue

            # Check for zero probabilities
            if not np.any(prob_sample):
                logger.warning(f"All probabilities are zero for image {i}. Using original probs.")
                crf[i] = prob_sample
                maxconf_crf[i] = np.argmax(prob_sample, axis=0).clip(0, num_classes - 1)
                continue

            # Ensure image is uint8 and contiguous
            contiguous_image = np.ascontiguousarray(image_sample.astype(np.uint8))

            # Initialize DenseCRF
            d = dcrf.DenseCRF2D(width, height, num_classes)
            U = np.ascontiguousarray(unary_from_softmax(prob_sample))
            if U.shape != (num_classes, height * width):
                logger.error(f"Bad shape for unary energy: expected ({num_classes}, {height * width}), got {U.shape}")
                continue
            d.setUnaryEnergy(U)

            # Add pairwise potentials
            d.addPairwiseGaussian(sxy=self.gauss_sxy, compat=self.gauss_compat)
            d.addPairwiseBilateral(sxy=self.bilat_sxy, srgb=self.bilat_srgb, rgbim=contiguous_image, compat=self.bilat_compat)

            # Perform inference
            try:
                Q = d.inference(self.n_infer)
                crf_probs = np.array(Q).reshape((num_classes, height, width))
                crf[i] = crf_probs
                maxconf_crf[i] = np.argmax(crf_probs, axis=0).clip(0, num_classes - 1)
            except Exception as e:
                logger.error(f"CRF inference failed for image {i}: {str(e)}. Using original probs.")
                crf[i] = prob_sample
                maxconf_crf[i] = np.argmax(prob_sample, axis=0).clip(0, num_classes - 1)

        logger.debug(f"CRF output shapes - maxconf_crf: {maxconf_crf.shape}, crf: {crf.shape}")
        return maxconf_crf, crf
    
    