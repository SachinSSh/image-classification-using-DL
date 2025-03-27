import pandas as pd
import numpy as np
import autogluon.multimodal as agmm
import torch
from sklearn.model_selection import train_test_split
import os
from tqdm.auto import tqdm
import gc
from sklearn.metrics import classification_report
import logging
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ImageFilters:
    """Specialized filters for AI image detection"""
    @staticmethod
    def extract_frequency_features(image):
        """Extract frequency domain features to detect AI patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_transform = fftshift(fft2(gray))
        magnitude_spectrum = np.log(1 + np.abs(f_transform))
        return magnitude_spectrum
    
    @staticmethod
    def detect_texture_consistency(image):
        """Detect unusual texture patterns common in AI images"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = np.zeros((256, 256))
        for i in range(gray.shape[0]-1):
            for j in range(gray.shape[1]-1):
                i_val = gray[i,j]
                j_val = gray[i+1,j+1]
                glcm[i_val, j_val] += 1
        return glcm
    
    @staticmethod
    def analyze_edge_coherence(image):
        """Analyze edge coherence and continuity"""
        edges = cv2.Canny(image, 100, 200)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        edge_coherence = cv2.subtract(dilated, edges)
        return edge_coherence
    
    @staticmethod
    def check_color_distribution(image):
        """Check for unnatural color distributions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        return hist_h, hist_s

class Config:
    """Configuration class with model and training parameters"""
    OUTPUT_PATH = "."
    SEED = 42
    VAL_SIZE = 0.2
    
    # Data paths
    TRAIN_PATH = "/content/ai-vs-human-generated-dataset/train.csv"
    TEST_PATH = "/content/ai-vs-human-generated-dataset/test.csv"
    BASE_PATH = "/content/ai-vs-human-generated-dataset"

    # Enhanced AutoGluon configuration
    HYPERPARAMETERS = {
        "model": {
            "names": ["timm_image"],
            "dropout_rate": 0.5,
            "timm_image": {
                "checkpoint_name": "swin_large_patch4_window7_224",
                "use_pretrained": True,
            },
        },
        "optimization": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "max_epochs": 100,
            "batch_size": 16,
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
        },
    }

class ImageProcessor:
    """Process images and extract AI detection features"""
    def __init__(self):
        self.filters = ImageFilters()
    
    def process_image(self, image_path):
        """Process image and extract features for AI detection"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            freq_features = self.filters.extract_frequency_features(image)
            edge_coherence = self.filters.analyze_edge_coherence(image)
            hist_h, hist_s = self.filters.check_color_distribution(image)
            
            # Normalize features
            freq_features = cv2.resize(freq_features, (32, 32))
            edge_coherence = cv2.resize(edge_coherence, (32, 32))
            
            # Combine features
            combined_features = np.concatenate([
                freq_features.flatten(),
                edge_coherence.flatten(),
                hist_h.flatten(),
                hist_s.flatten()
            ])
            
            return combined_features
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None

def verify_image(image_path: str) -> bool:
    """Verify if an image file is valid"""
    try:
        with Image.open(image_path) as img:
            img.verify()
            if img.size[0] <= 0 or img.size[1] <= 0:
                return False
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return True
    except Exception:
        return False

class ImageClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.seed_everything(config.SEED)
        self.logger = logging.getLogger(__name__)
        self.processor = ImageProcessor()
        self.train = None
        self.test = None
        self.train_data = None
        self.val_data = None
        self.predictor = None
        self.setup_data()

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("training.log"),
            ],
        )
        logging.info("Logging configured successfully.")

    def seed_everything(self, seed: int) -> None:
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def validate_images(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Validate and filter images"""
        self.logger.info(f"Validating {'training' if is_train else 'test'} images...")
        valid_indices = []
        dropped_images = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row["image"]
            full_path = os.path.join(self.config.BASE_PATH, image_path)

            if not os.path.exists(full_path):
                dropped_images.append((image_path, "File not found"))
                continue
            
            if os.path.exists(full_path) and verify_image(full_path):
                valid_indices.append(idx)
                df.at[idx, 'image'] = full_path
            else:
                self.logger.warning(f"Invalid or missing image: {full_path}")
        
        valid_df = df.loc[valid_indices].copy()
        self.logger.info(f"Found {len(valid_df)} valid images out of {len(df)}")
        self.logger.info(f"Dropped {len(dropped_images)} images:")
        for img, reason in dropped_images:
            self.logger.info(f"- {img}: {reason}")
        
        if len(valid_df) == 0:
            raise ValueError(f"No valid images found in {'training' if is_train else 'test'} set")
            
        return valid_df

    def process_batch(self, image_paths: list) -> np.ndarray:
        """Process a batch of images in parallel"""
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(self.processor.process_image, image_paths))
        return np.array([f if f is not None else np.zeros(8960) for f in features])

    def setup_data(self) -> None:
        """Load and prepare data"""
        try:
            # Load CSV files
            self.logger.info("Loading CSV files...")
            train_df = pd.read_csv(self.config.TRAIN_PATH)
            test_df = pd.read_csv(self.config.TEST_PATH)

            # Drop unnamed column if it exists
            if "Unnamed: 0" in train_df.columns:
                train_df = train_df.drop("Unnamed: 0", axis=1)

            # Prepare train DataFrame
            self.train = pd.DataFrame({
                'image': train_df['file_name'],
                'label': train_df['label']
            })

            # Prepare test DataFrame
            self.test = pd.DataFrame({
                'image': test_df['id']
            })

            # Validate images
            self.train = self.validate_images(self.train, is_train=True)
            self.test = self.validate_images(self.test, is_train=False)

            # Create train/val split
            self.train_data, self.val_data = train_test_split(
                self.train,
                test_size=self.config.VAL_SIZE,
                random_state=self.config.SEED,
                stratify=self.train["label"]
            )

            self.logger.info(f"Final data splits - Train: {len(self.train_data)}, "
                           f"Val: {len(self.val_data)}, Test: {len(self.test)}")

        except Exception as e:
            self.logger.error(f"Error in data setup: {str(e)}")
            raise

    def train_model(self) -> None:
        """Train the model"""
        self.logger.info("Training model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            self.predictor = agmm.MultiModalPredictor(
                label="label",
                path=os.path.join(self.config.OUTPUT_PATH, "model_aihi_dketghjbnkndwcector"),
                problem_type="binary",
            )
            
            # Custom evaluation metric
            def custom_metric(y_true, y_pred):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                return (acc + prec + rec + f1) / 4
            
            self.predictor.fit(
                train_data=self.train_data,
                tuning_data=self.val_data,
                hyperparameters=self.config.HYPERPARAMETERS,
                time_limit=7200,  # Increase time limit for more training
            )
            
            # Validate model performance
            val_predictions = self.predictor.predict(self.val_data)
            report = classification_report(
                self.val_data["label"],
                val_predictions,
                target_names=["AI Generated", "Real"],
                output_dict=True,
            )
            self.logger.info(f"\nValidation Results:\n{pd.DataFrame(report).T}")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self) -> None:
        """Generate predictions with confidence scores"""
        self.logger.info("Generating predictions...")
        try:
            predictions = self.predictor.predict(self.test)
            submission = pd.DataFrame({
                "id": self.test["image"].apply(lambda x: os.path.basename(x).split('.')[0]),
                "label": predictions.astype(int)
            })
            missing_ids = set(self.test["image"].apply(lambda x: os.path.basename(x).split('.')[0])) - set(submission["id"])
            if missing_ids:
                self.logger.warning(f"Missing {len(missing_ids)} IDs in submission: {missing_ids}")
            
            # Save predictions
            submission_path = os.path.join(self.config.OUTPUT_PATH, "submissionzx.csv")
            submission.to_csv(submission_path, index=False)
            
            # Save detailed results
            detailed_path = os.path.join(self.config.OUTPUT_PATH, "detailed_results.csv")
            submission.to_csv(detailed_path, index=False)
            
            self.logger.info(f"Predictions saved to {submission_path}")
            self.logger.info(f"Detailed results saved to {detailed_path}")
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def run(self) -> None:
        """Run the complete pipeline"""
        try:
            start_time = pd.Timestamp.now()
            self.logger.info(f"Starting pipeline at {start_time}")
            self.logger.info("GPU Available: " + str(torch.cuda.is_available()))
            if torch.cuda.is_available():
                self.logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            
            if torch.cuda.is_available():
                gpu_memory_start = torch.cuda.memory_allocated(0)
                self.logger.info(f"Initial GPU Memory: {gpu_memory_start/1e9:.2f} GB")
            
            self.train_model()
            self.predict()
            
            if torch.cuda.is_available():
                gpu_memory_end = torch.cuda.memory_allocated(0)
                self.logger.info(f"Final GPU Memory: {gpu_memory_end/1e9:.2f} GB")
            
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            self.logger.info(f"Pipeline completed in {duration}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        classifier = ImageClassifier(Config())
        classifier.run()
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise
