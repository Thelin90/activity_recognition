# Intelematics Challenge

Please se original task under `docs/README.md`.

# Dataset

The dataset contains over `1 million` rows it contains the activity that a person has performed during measurements,
please see original `README` for better context.

# Data processing

## PySpark

PySpark have some core concepts. Such as resilient distributed datasets and dataframes.

This project spefically take advantage of the dataframes.

It has derived from the resilient distributed datasets concept. Dataframes will provide a higher level of abstraction. Which implicitly allow a query language way of typing to manipulate data. The abstraction represents data and a schema. Which enable better interaction with front end applications.

## Scikit-learn

Scikit-learn is a great open source, machine learning library with good and solid support, excellent to use when wanting
to test concepts but also for proper production ready code.

`https://en.wikipedia.org/wiki/Scikit-learn`

# Extract

Firstly the data is read from its `CSV` format and stored as a `spark` dataframe.

# Transform

The original dataset will be splittet up into 12 seperate files to make it easier to work with.

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

# KNN Regressor

For this project, as it has been mentioned before, the original dataset has been splitted into
12 subset files.

Currently a class has been implemented that uses KNN regressor from scikit-learn.
The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.

The thought is that it should be used on all the files, and that you could combine
each model, for each file. But for saving time this has been done on
the first subset to prove that it works, which is the important thing.

The other good thing is that the `KNNRegressor` in scikit-learn, normalises the data for the user, which saves some time
in implementation.

## datasets

The datasets are subsets from the original dataset.

The target value has been re-labled as followed:

```json
+-----------------+ 
| predicted_target|
+-----------------+
|                0|
|                1|
|                2|
|                3|
|                4|
|                5|
|                6|
|                7|
|                8|
+-----------------+
```

This is then translated back to:

```json
+-----------------+ 
| predicted_target|
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
```

### training

The training dataset is a subset of `70%` from the initial subset of the first dataset.

### test

The test dataset is a subset of `30%` from initial subset of the first dataset.



# Results

The results are written to a csv file `data/split_data/end_result`.

The `MSE` is currently `~0.3707049590458891` based on the
first subset and `n_neighbours` set to 5.

# Requirements

* Python setup
* Docker
* Apache Spark

# Setup

## Setup Apache-Spark

Start with downloading Spark (note that depending on your IDE, you need to specify your Spark location):

- https://spark.apache.org/downloads.html

Set your SPARK_HOME in `.bashrc`
```bash
SPARK_HOME='path-to-spark'
```

Then source the file

```bash
source ~/.bashrc
```

## Python and Core project


### Core 

You will need to create the folders for the data, run in project root:

 * `mkdir data`
 * `cd data` --> `mkdir raw_data`, `split_data`

Download the source data: `https://drive.google.com/file/d/1YNG0PPv0lnKKHzDBd3uWq248k7Aj-I8q/view?usp=sharing`

and put this file in the `raw_data` folder. This is because the file is to big to upload to git.

### Python

* PYTHONPATH (Make sure the correct `VENV` is being used)

Example:

```bash
# Python
PYTHONPATH=/usr/bin/python3.6
export PYTHONPATH=$PYTHONPATH:~/activity_recognition
alias python=/usr/bin/python3.6
```

# Run

This section will explain how to run the project.

## Manual

Go inside the project root, run:

`spark-submit main.py`

This will spin up the process and run the application.

## Docker

TODO: 

## Tests

TODO:

# Results

The current result is written as mentioned before to `data/split_data/end_results`.

An example of the finished dataframe:

```json
+----------------+--------------------+-----------------+-----------+
|predicted_target|predicted_target_str|actual_target_str|         id|
+----------------+--------------------+-----------------+-----------+
|             3.0|               lying|            lying|         26|
|             3.0|               lying|            lying|         29|
|             3.0|               lying|            lying|        474|
|             3.0|               lying|            lying|        964|
|             2.0|             sitting|          sitting|       1677|
|             2.0|             sitting|          sitting|       1697|
|             2.0|             sitting|          sitting|       1806|
|             2.0|             sitting|          sitting|       1950|
|             2.0|             sitting|          sitting|       2040|
|             2.0|             sitting|          sitting|       2214|
|             2.0|             sitting|          sitting|       2250|
|             0.0|            standing|         standing|       2453|
|             0.0|            standing|         standing|       2509|
|             0.0|            standing|         standing|       2529|
|             5.4|             walking| ascending_stairs|       2927|
|             5.0|             walking|descending_stairs| 8589934658|
|             8.0|    ascending_stairs| ascending_stairs| 8589934965|
|             7.0|   descending_stairs|descending_stairs| 8589935171|
|             7.0|   descending_stairs|descending_stairs| 8589935183|
|             5.6|             cycling|          walking| 8589935298|
|             5.0|             walking|          walking| 8589935317|
|             5.0|             walking|          walking| 8589935768|
|             5.0|             walking|          walking| 8589935770|
|             6.0|             cycling|          cycling| 8589935936|
|             6.0|             cycling|          cycling| 8589936112|
|             3.0|               lying|            lying| 8589936348|
|             3.0|               lying|            lying| 8589936424|
|             3.0|               lying|            lying| 8589936566|
|             3.0|               lying|            lying| 8589936761|
|             2.0|             sitting|          sitting| 8589936870|
|             2.0|             sitting|          sitting| 8589936972|
|             0.0|            standing|         standing| 8589937263|
|             0.0|            standing|         standing| 8589937582|
|             0.0|            standing|         standing| 8589937853|
|             0.0|            standing|         standing| 8589937874|
|             2.0|             sitting|          sitting| 8589937892|
|             2.0|             sitting|          sitting| 8589937972|
|             2.0|             sitting|          sitting| 8589938024|
|             0.0|            standing|         standing| 8589938116|
|             0.0|            standing|         standing| 8589938148|
|             0.0|            standing|         standing| 8589938178|
|             0.0|            standing|         standing| 8589938345|
|             0.0|            standing|         standing| 8589938353|
|             0.0|            standing|         standing| 8589938411|
|             0.0|            standing|         standing| 8589938560|
|             0.0|            standing|         standing| 8589938663|
|             6.0|             cycling|descending_stairs|17179869487|
|             7.8|    ascending_stairs| ascending_stairs|17179869788|
|             3.4|               lying|          walking|17179870193|
|             5.0|             walking|          walking|17179870343|
|             5.0|             walking|          walking|17179870614|
|             2.4|             sitting|          running|17179870829|
|             3.6|        rope_jumping|          running|17179871026|
|             3.0|               lying|            lying|17179871724|
|             3.0|               lying|            lying|17179871947|
|             3.0|               lying|            lying|17179872052|
|             3.0|               lying|            lying|17179872108|
|             2.0|             sitting|          sitting|17179872248|
|             0.0|            standing|         standing|25769804400|
|             8.0|    ascending_stairs| ascending_stairs|25769804925|
|             6.2|             cycling| ascending_stairs|25769805484|
|             8.0|    ascending_stairs| ascending_stairs|25769805506|
|             8.0|    ascending_stairs| ascending_stairs|25769805608|
|             5.0|             walking|          walking|25769806203|
|             5.0|             walking|          walking|25769806291|
|             5.4|             walking|          walking|25769806339|
|             5.0|             walking|          walking|25769806426|
|             6.0|             cycling|          cycling|25769806957|
|             6.0|             cycling|          cycling|25769807245|
|             1.8|             sitting|          running|25769807430|
|             4.0|        rope_jumping|     rope_jumping|25769807816|
|             3.0|               lying|            lying|34359738398|
|             3.0|               lying|            lying|34359738893|
|             3.0|               lying|            lying|34359738988|
|             2.0|             sitting|          sitting|34359739499|
|             2.0|             sitting|          sitting|34359739552|
|             2.0|             sitting|          sitting|34359739694|
|             2.0|             sitting|          sitting|34359740036|
|             2.0|             sitting|          sitting|34359740216|
|             0.0|            standing|         standing|34359740494|
|             0.0|            standing|         standing|34359740587|
|             0.0|            standing|         standing|34359740629|
|             7.0|   descending_stairs|descending_stairs|34359741238|
|             5.0|             walking|          walking|34359741558|
|             5.0|             walking|          walking|34359742109|
|             5.0|             walking|          walking|34359742178|
|             5.0|             walking|          walking|34359742241|
|             5.0|             walking|          walking|34359742388|
|             5.0|             walking|          walking|34359742413|
|             6.0|             cycling|          cycling|42949673263|
|             6.0|             cycling|          cycling|42949673366|
|             6.0|             cycling|          cycling|42949673505|
|             3.0|               lying|            lying|42949673632|
|             3.0|               lying|            lying|42949673713|
|             2.0|             sitting|          sitting|42949674099|
|             2.0|             sitting|          sitting|42949674345|
|             2.0|             sitting|          sitting|42949674461|
|             0.0|            standing|         standing|42949674641|
|             0.0|            standing|         standing|42949675020|
|             8.0|    ascending_stairs| ascending_stairs|42949675232|
+----------------+--------------------+-----------------+-----------+
```

For the run above the `MSE` was received:

`INFO:root:mse: 0.3843861533663675`
# Conclusion

The main idea with how I solved this solution is that I wanted to make sure what type of data it is, what formats. Also 
I had in mind in regards to the size, how would this in the most efficient and good way be evaluated with the current time frame.

So the idea is that with the current implementation, it should be possible to make 12 different models of the `KNNR` and then
stack them together, to be able to process data of the kind that exists in the original dataset.

`KNNR` is good for datasets `< 100k`, so hence why.

If I had more time I would do some more plotting, and probably also would like to try a `SGD` classifier, but this would probably
take much more time.

So there has been a balance of how can I create a production ready module, and still get good results.