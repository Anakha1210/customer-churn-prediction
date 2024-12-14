import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load dataset from a CSV file."""
    df = pd.read_csv(r"c:\Users\anakh\Documents\churn\customer-churn-prediction\data\telco_churn.csv")
    return df

def clean_data(df):
    """Clean the dataset by handling missing values, encoding, and dropping columns."""
    # Handle missing data
    missing_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    for column in df.select_dtypes(include=['int64', 'float64']):
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object']):
        df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Drop irrelevant columns
    df.drop(columns=['customerID'], inplace=True)
    
    return df

def encode_data(df):
    """Encode categorical columns."""
    # Label encoding for 'gender'
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    
    # One-hot encoding for other categorical columns
    df = pd.get_dummies(df, columns=['contract', 'payment_method'], drop_first=True)
    return df

def split_data(df, target_column):
    """Split the data into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
