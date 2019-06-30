package com.sandysnunes.old

import org.apache.spark.sql.{SQLContext, SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

// spark-submit --class GeradorBigTableClientesPecas --master local[*] --executor-memory 5g --driver-memory 14g dados.csv
object GeradorBigTableClientesPecas extends App {

  val separador = ";"

  val conf: SparkConf = new SparkConf()
    .setAppName("GeradorBigTableClientesPecas")
    .setMaster("local[*]")
    .set("spark.driver.memory", "14g")
    .set("spark.executor.memory", "5g")
    .set("spark.sql.pivotMaxValues", "1000000")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = session.sparkContext
  val sqlContext: SQLContext = session.sqlContext

  import org.apache.spark.sql.functions._
  import sqlContext.implicits._

  case class ClientePeca(idCliente: Int, idPeca: Int, qtdeCompras: Double)

  sc.textFile(args(0))
    .map(linha => linha.split(separador))
    .map({ case Array(idCliente, idPeca, qtdeCompras) =>
      ClientePeca(idCliente.toInt, idPeca.toInt, qtdeCompras.toDouble)
    })
    .cache()
    .toDF(colNames = "id_cliente", "id_peca", "qtde_compras")
    .groupBy("id_cliente")
    .pivot("id_peca")
    .agg(coalesce(first("qtde_compras")))
    .coalesce(numPartitions = 1)
    .write
    .mode(SaveMode.Overwrite)
    .format( source = "com.databricks.spark.csv")
    .option("header", "true")
    .option("delimiter", separador)
    .option("emptyValue", "")
    .save(path = "big-table-gerada")


}
