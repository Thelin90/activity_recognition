from src.app.knnRegressor import KNNRegression
from src.app.spark_session import InitSpark
from src.app.etl import ETL


def main():
    """Initialise SparkSession

    """
    init_spark_session = InitSpark("activity_recognition", "spark-warehouse")
    spark = init_spark_session.spark_init()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.debug.maxToStringFields", 100)

    etl = ETL(spark, ',', "true", "true")
    train_x, train_y, test_x, test_y = etl.run()

    knnr = KNNRegression(train_x, train_y, test_x, test_y, 5)

    # currently no cross validation, but would do this would some more time to explore the neighbours
    results = knnr.run()

    etl.load(results)


if __name__ == "__main__":
    main()
