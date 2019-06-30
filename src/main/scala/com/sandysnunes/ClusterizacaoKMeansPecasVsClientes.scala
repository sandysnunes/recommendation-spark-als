package com.sandysnunes

import java.text.NumberFormat
import java.time.LocalDateTime
import java.util.Locale

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.mllib.clustering.{DistanceMeasure, KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SaveMode, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

//spark-submit --class ClusterizacaoKMeansPecasVsClientes --master local[*] --executor-memory 1g --driver-memory 15g --conf "spark.executor.extraJavaOptions=-verbose:gc -XX:+UseSerialGC -XX:+UseCompressedOops -XX:+UseCompressedStrings  -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:PermSize=256M -XX:MaxPermSize=512M" recommendation-spark-als-1.0-SNAPSHOT.jar   2018.csv
object ClusterizacaoKMeansPecasVsClientes extends App {

  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  val log = LogManager.getRootLogger
  val formatter = NumberFormat.getNumberInstance(new Locale("pt", "BR"))
  val separador = ';'

  val conf: SparkConf = new SparkConf()
    .setAppName("ClusterizacaoKMeansPecasVsClientes")
    .set("spark.sql.pivotMaxValues", "1000000")
    .setMaster("local[*]")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    /*.set("spark.driver.memory", "14g")
    .set("spark.executor.memory", "2g")*/

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = session.sparkContext
  val sqlContext: SQLContext = session.sqlContext

  import org.apache.spark.sql.functions._
  import sqlContext.implicits._

  val ZERO = 0.0d

  case class ClientePeca(idCliente: Int, codigoPeca: String, qtdeCompras: Double)

  val rddWithClienteId: RDD[(Int, linalg.Vector)] = sc.textFile(args(0))
    .map(linha => linha.split(separador))
    .map({ case Array(idCliente, codigoPeca, qtdeCompras) => ClientePeca(idCliente.toInt, codigoPeca, qtdeCompras.toDouble)})
    .toDF(colNames = "id_cliente", "id_peca", "qtde_compras")
    .groupBy("id_cliente")
    .pivot("id_peca")
    .agg(coalesce(first("qtde_compras")))
    .rdd
    .map(row => row.toSeq.toArray)
    .map(row => (row(0), row.slice(1, row.length)))
    .map({ case (clienteAgrupadoId, quantidades) => (clienteAgrupadoId.asInstanceOf[Int], quantidades.map(valor => Option(valor).map(value => value.asInstanceOf[Double]).getOrElse(ZERO))) })
    .map({ case (clienteAgrupadoId, quantidades) => (clienteAgrupadoId, Vectors.dense(quantidades).toSparse.asInstanceOf[linalg.Vector]) })
    .persist(StorageLevel.MEMORY_AND_DISK_SER_2)

  val vectors: RDD[linalg.Vector] = rddWithClienteId
      .map({case (_, vetor) => vetor})


  val custos = new ListBuffer[(Int, Double)]

    2.to(45).foreach(k => {
      log.info("")
      log.info(s"[KMeansPecasVsClientes] Iniciando treino com k=$k")
      log.info("")

      val model: KMeansModel = new KMeans().setK(k).setDistanceMeasure(DistanceMeasure.COSINE).run(vectors)

      model.save(sc, s"/home/sandys/modelos/KMeansPecasVsClientes-k-$k")
      custos.append((k, model.trainingCost))

      log.info("")
      log.info(s"[KMeansPecasVsClientes] Finalizando treino com k=$k, custo=${model.trainingCost}")
      log.info("")
    })

    custos.foreach({case (k, custo) => println(s"$k\t${formatter.format(custo)}")})


  List(20, 30, 40).foreach(k => {
    val model: KMeansModel = new KMeans().setK(k).setDistanceMeasure(DistanceMeasure.COSINE).run(vectors)
    model.save(sc, s"/home/sandys/modelos/KMeansPecasVsClientes-${LocalDateTime.now().toString}")

    val clientesClassificados: RDD[(Int, Int)] = rddWithClienteId.map({ case (clienteId, vetor) =>
      val clusterNumber = model.predict(vetor)
      (clusterNumber, clienteId)
    })

    clientesClassificados
      .toDF("cluster_number", "id_cliente_agrupado")
      .coalesce(numPartitions = 1)
      .write
      .mode(SaveMode.Overwrite)
      .format(source = "com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ";")
      .option("emptyValue", "")
      .save(path = s"/home/sandys/tcc-sandys/cliente-classficados-2018-k-${k.toString}")
  })


  /*val contagem = clientesClassificados reduceByKey (_ + _) collect()
  contagem.foreach({case (fold, qtde) => println(s"cluster number: $fold, qtde: $qtde")})*/


}
