from src.logger import logger
from src.data.validate import validate_data
from src.data.preprocess import preprocess_data
from src.models.train import run_training

def run_pipeline():
    logger.info("===========================================")
    logger.info("🚀 STARTING AUTOMATED ML PIPELINE 🚀")
    logger.info("===========================================")

    try:
        # الخطوة الأولى: فحص البيانات
        logger.info(">>> STEP 1: Data Validation")
        is_valid = validate_data()
        
        if not is_valid:
            logger.error("❌ Pipeline stopped: Data validation failed!")
            return

        # الخطوة الثانية: المعالجة
        logger.info(">>> STEP 2: Data Preprocessing")
        preprocess_data()

        # الخطوة الثالثة: التدريب والتقييم
        logger.info(">>> STEP 3: Model Training & Evaluation")
        run_training()

        logger.info("===========================================")
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY! ✅")
        logger.info("===========================================")

    except Exception as e:
        logger.error(f"❌ Pipeline failed due to an error: {e}")

if __name__ == "__main__":
    run_pipeline()