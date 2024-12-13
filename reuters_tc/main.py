from dataset import ReutersDatasetLoader

def test_reuters_loader():
    # Initialize the loader
    loader = ReutersDatasetLoader()
    
    # Load and prepare the dataset
    data, encoded_labels = loader.prepare()
    
    # Print some basic information
    print(f"Dataset shape: {data.shape}")
    print(f"Number of documents: {len(data)}")
    print(f"Number of unique labels: {len(loader.label_names)}")
    print(f"Labels encoded shape: {encoded_labels.shape}")
    
    # Print first document example
    print("\nFirst document example:")
    print(f"Document ID: {data['document_id'].iloc[0]}")
    print(f"Content preview: {data['content'].iloc[0][:200]}...")
    print(f"Labels: {data['labels'].iloc[0]}")

if __name__ == "__main__":
    test_reuters_loader()
