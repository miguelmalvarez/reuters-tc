# reuters-tc

Repository to perform different classification processes with the Reuters-21578 in python. The goal of this repository is to show some of the power of different Machine Learning tools in python (e.g., scikit-learn) when representing and classifiying textual data. The collection used in this example is Reuters-21578, a traditional and extremely wildly used collection for text classification. This collection is currently very small compared to larger datasets (e.g., RCV-1) but it is big enough to have meaninful examples with real data.

Reuters-21578 contains structured information about newswire articles that can be assigned to several classes (i.e., multi-label problem). The “ModApte” split is used where we only consider classes with at least one training and one test document. As a result, there are 7770/3019 documents for training and testing, observing 90 classes with a highly skewed distribution over classes.

All the examples are presented using jupyter notebooks to increase its reproducibility.

In order to install the requirement please run:

pip install -r requirements.txt

This will download and install NLTK, scikit-learn and jupyter (plus dependencies).

NLTK requires some data to be installed separately (more details on `the NLTK website <http://www.nltk.org/data.html>`_).

From the command line, you can download the required packages::

    python -m nltk.downloader stopwords reuters

Alternatively, from a Python interactive shell::

    >>> import nltk
    >>> nltk.download()

Then use the GUI to select the requires packages (stopwords, reuters).

Finally - run Jupyter::

    jupyter notebook
