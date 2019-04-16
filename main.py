from src.modules.env.config import DEF_CSV_PATH
from src.app.spark_session import InitSpark
from src.app.etl import ETL


def main():
    """Initialise SparkSession

    """
    init_spark_session = InitSpark("activity_recognition", "spark-warehouse")
    spark = init_spark_session.spark_init()
    spark.sparkContext.setLogLevel("WARN")

    etl = ETL(spark, ',', True, True, DEF_CSV_PATH)

    df = etl.extract()

    etl.load(df)


if __name__ == "__main__":
    main()
