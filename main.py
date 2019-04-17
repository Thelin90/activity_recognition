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
    df = etl.run()

    etl.load(df)

    #print(df.schemaTypes())


if __name__ == "__main__":
    main()
