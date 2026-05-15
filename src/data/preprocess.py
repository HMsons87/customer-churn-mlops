import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import (
    RAW_DATA_PATH, 
    TRAIN_DATA_PATH, 
    TEST_DATA_PATH, 
    PREPROCESSOR_PATH, 
    TARGET_COLUMN, 
    RANDOM_STATE
)
from src.logger import logger

def preprocess_data():
    logger.info("Starting data preprocessing...")
    
    # 1. تحميل البيانات الخام
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. حل مشكلة TotalCharges (تحويل المسافات لـ NaN ثم تعويضها بـ 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. حذف عمود customerID لأنه لا يفيد في التوقع
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # 4. فصل الميزات (Features) عن الهدف (Target)
    X = df.drop(columns=[TARGET_COLUMN])
    # تحويل الهدف من Yes/No إلى 1/0
    y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    
    # 5. تحديد نوع الأعمدة
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"Found {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")
    
    # 6. بناء أنابيب المعالجة (Pipelines)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False مهمة لتجنب مشاكل التوافق مع بعض الخوارزميات
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # دمج الأنابيب في ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # 7. تقسيم البيانات لتدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 8. تنفيذ المعالجة (Fit & Transform)
    logger.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 9. حفظ أداة المعالجة (Preprocessor) لاستخدامها في واجهة المستخدم لاحقاً
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    
    # 10. إرجاع أسماء الأعمدة بعد الـ OneHotEncoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    # 11. دمج البيانات المعالجة مع الهدف وحفظها
    train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    train_df[TARGET_COLUMN] = y_train.values
    
    test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    test_df[TARGET_COLUMN] = y_test.values
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    logger.info(f"Processed Train data saved to {TRAIN_DATA_PATH}")
    logger.info(f"Processed Test data saved to {TEST_DATA_PATH}")
    logger.info("Data Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()