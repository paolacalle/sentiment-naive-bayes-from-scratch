from datasets import load_dataset

class IMDBLoader:
    """ 
    A data loader for the IMDB movie reviews dataset.
    This class loads the dataset and provides training and testing splits.
    Methods:
    -------
    load_data():
        Loads the IMDB dataset and returns training and testing data.
    """
    def __init__(self):
        self.train = None
        self.test = None
        self.load_data()

    def load_data(self):
        dataset = load_dataset("imdb")
        self.train = dataset["train"].to_pandas().rename(columns={"text": "review"})
        self.test = dataset["test"].to_pandas().rename(columns={"text": "review"})
        
    def get_sample(self, df, sample_size, random_state=42):
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    
    def get_splits(self):
        train_sample = self.get_sample(self.train)
        test_sample = self.get_sample(self.test)
        
        X_train = train_sample["review"].tolist()
        y_train = train_sample["label"].tolist()
        
        X_test = test_sample["review"].tolist()
        y_test = test_sample["label"].tolist()
        
        return X_train, y_train, X_test, y_test


