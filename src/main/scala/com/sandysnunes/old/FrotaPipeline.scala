package old

import org.apache.log4j.LogManager
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

object FrotaPipeline extends App {


  val log = LogManager.getRootLogger

  val conf: SparkConf = new SparkConf()
    .setAppName("FrotaPipeline")
    .setMaster("local[*]")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = session.sparkContext
  val sqlContext: SQLContext = session.sqlContext

  Class.forName("org.postgresql.Driver")
  val url = "jdbc:postgresql://localhost:7000/dados"
  val usuario = "postgres"
  val senha = "postgres"


  val vectors = sqlContext.read.format("jdbc")
    .option("url", url)
    .option("dbtable",
      """
        |(select some_field from some_table) as temp
      """.stripMargin)
    .option("user", usuario)
    .option("password", senha)
    .load()
    .rdd
    .cache()
    .map(_.toSeq)
    .map(_.toArray)
    .map(_.map(_.asInstanceOf[Long]))
    .map(_.map(_.toDouble))
    .map(Vectors.dense)
    .cache()




  val custos = new ListBuffer[(Int, Double)]

  2 to 20 foreach (k => {
    log.info("")
    log.info(s"[KMeansPecasVsClientes] Iniciando treino com k=$k")
    log.info("")

    val model: KMeansModel = new KMeans().setK(k).run(vectors)
    custos.append((k, model.trainingCost))

    log.info("")
    log.info(s"[KMeansPecasVsClientes] Finalizando treino com k=$k, custo=${model.trainingCost}")
    log.info("")
  })

  custos.foreach({case (k, custo) => println(s"$k\t$custo")})

 /* val model: KMeansModel = KMeans.train(vectors, 3, maxIterations = 20)

  val clientesClassificados: RDD[(Int, Long)] = vectors.map(cliente => {
    val clusterNumber = model.predict(cliente)
    ( clusterNumber,  1)
  })
  val contagem = clientesClassificados reduceByKey (_ + _) collect()
  contagem*/

  sc.stop()

}
