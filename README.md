# reuters-tc

Repository to showcase and explore different classification approaches with the Reuters-21578 collection. 
The goal of this repository is to show different Machine Learning ideas when representing and classifying textual data. 

The collection used in this example is Reuters-21578, a traditional and historically very commonly used collection 
for text classification in academia. This collection is very small by any modern standards, but it has a number of 
characteristics that make it great for educational purposes as it is similar to many (albeit small) datasets we might 
encounter in some industrial applications. Namely, it is a clearly unbalanced dataset, with classes ranging from a 
handful of examples to thousands. 


## The collection
Reuters-21578 contains structured information about newswire articles that can be assigned to several classes 
(i.e., multi-label problem). The “ModApte” split is used where we only consider classes with at least one training and 
one test document. As a result, there are 7770 / 3019 documents for training and testing, observing 90 classes 
with a highly skewed distribution over classes.

In order to download the collection we will install nltk (more details on `the NLTK website 
<http://www.nltk.org/data.html>`_) and use the following command:

    python -m nltk.downloader reuters

For easy of use, we also have a downloadCorpus.sh that will perform the same operation.