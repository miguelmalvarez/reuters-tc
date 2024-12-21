# reuters-tc

Repository to showcase and explore different classification approaches with the Reuters-21578 collection. 

The collection used in this example is Reuters-21578, a historical, and old, collection for text classification commonly seen in academia in the 2010s. This collection is tiny  by any modern standards, but it has a number of characteristics that make it great for educational purposes, as it is similar to many (albeit small) datasets we might 
encounter in some industrial applications. Namely, it is a clearly unbalanced dataset, with classes ranging from a handful of examples to thousands.

Reuters-21578 contains structured information about newswire articles that can be assigned to several classes, making it a multi-label classification problem. The “ModApte” split is used where we only consider classes with at least one training and 
one test document. As a result, there are 7770 / 3019 documents for training and testing, observing 90 classes with a highly skewed distribution over classes.