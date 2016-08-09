import reuters-tc.representation.core

def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");

    # Documents in a category
    category_docs = reuters.fileids("acq");

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);

    # Raw document
    print(reuters.raw(document_id));

def main():
    train_docs = []
    test_docs = []

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))

    representer = tf_idf(train_docs);

    for doc in test_docs:
        print(feature_values(doc, representer))