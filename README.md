# Intelematics Challenge

The project is solving the problem with ...

# Dataset

# Data processing

For this project data processing will be done in `pyspark` and `scikit-learn` this more discussed below.

# Extract

Firstly the data is read from its `CSV` format and stored as a `spark` dataframe.

The dataframe contains well over a million records.

# Transform

The dataframe is then cleaned, in the sense of datatypes, and putting a default value of 
`0.0` to replace `null`

Then the the distinct categorial values are evaluated, currently this is the foundings:

```json
+-----------------+ 
|      activity_id|
+-----------------+
|         standing|
|          running|
|          sitting|
|            lying|
|     rope_jumping|
|          walking|
|          cycling|
|descending_stairs|
| ascending_stairs|
+-----------------+

9 activities

+-------+
|user_id|
+-------+
|      c|
|      g|
|      f|
|      d|
|      a|
|      b|
|      e|
+-------+

7 participants
```

The rest of the features are numerical.

To make the life easier, `user_id` will be dropped, the idea is that it should not matter
which person it is, trying to make a generic regression classification on the data based on the target value, and
therefor being able to send in data without knowing who it is and being able to map that to the original person.
However the main concern here is to figure out which of the target types that occur for the given row.

## Machine learning?

So there is currently 9 categories. In this particular case, I find the `Stochastic Gradient Descent` interesting, let's
have a closer look if it is a good fit due to the large dataset.

When taking a look at the recommendation from `sci-kitlearn`

![alt_text](https://scikit-learn.org/stable/_static/ml_map.png)

The steps will go as followed:

* > `>` 50 samples? (yes)
* > categorical? (yes)
* > labeled? (yes)
* > `<` 100k samples? (no)
* > choose `Stochastic Gradient Descent`

But I feel that it might require some time to get the SGD to work properly, so for the time limit that is for this project,
I will use spark to split the dataset into 12 datasets, and I will then run KNN on
one of these to begin with and if I am successful I will do it on several of them.

The good thing with `KNNRegression` from `scikit-learn` in regards to speed. It is an lazy algorithm and it works
good with multiple targets, and it has a good support in the `scikit-learn` library.

If I had more time, let's say a couple of weeks, I would probably implement a more sofisticated SGD model
to be able to process all data at once into one model.
# KNN

It works with bla bla bla

## datasets

### training

### test

# Requirements

# Setup

# Run

## Manual

## Docker

# Results

# Conclusion