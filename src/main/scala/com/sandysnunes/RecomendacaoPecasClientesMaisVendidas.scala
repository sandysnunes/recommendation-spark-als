package com.sandysnunes

import java.text.NumberFormat
import java.util.Locale
import java.util.Locale.Category

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.List
import scala.collection.mutable.ListBuffer

/**
  * Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008). Large-Scale Parallel Collaborative Filtering for the Netflix Prize BT  - Algorithmic Aspects in Information and Management. Algorithmic Aspects in Information and Management, 337–348. https://doi.org/10.1007/978-3-540-68880-8_32
  *
  * Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30–37. https://doi.org/10.1109/MC.2009.263
  *
  * Hu, Y., Volinsky, C., & Koren, Y. (2008). Collaborative filtering for implicit feedback datasets. Proceedings - IEEE International Conference on Data Mining, ICDM, 263–272. https://doi.org/10.1109/ICDM.2008.22
  */


// spark-submit \
// --class RecomendacaoPecasClientesMaisVendidas \
// --master local[*] \
// --executor-memory 1g \
// --driver-memory 15g \
// --conf "spark.executor.extraJavaOptions=-verbose:gc -XX:+UseSerialGC -XX:+UseCompressedOops -XX:+UseCompressedStrings  -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:PermSize=256M -XX:MaxPermSize=512M" \
// recommendation-spark-als-1.0-SNAPSHOT.jar \
// 2018.csv
object RecomendacaoPecasClientesMaisVendidas extends App {


  val fmtLocale = Locale.getDefault(Category.FORMAT)
  val formatter = NumberFormat.getInstance(fmtLocale)
  formatter.setMaximumFractionDigits(5)
  formatter.setMinimumFractionDigits(5)

  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  Logger.getLogger("com.github").setLevel(Level.OFF)

  val NUMBER_OF_BUCKETS = 10

  val conf: SparkConf = new SparkConf().setAppName("RecomendacaoPecasClientesMaisVendidas")
    .setMaster("local[*]") //comentar linha para spark submit

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = session.sparkContext
  val sqlContext: SQLContext = session.sqlContext

  //import sqlContext.implicits._


  val rawRatings = sc.textFile(args(0))
    .map(_ split ';')
    .map { case Array(idClienteAgrupado, codigoPeca, qtdeCompras) =>
      Rating(idClienteAgrupado.toInt, codigoPeca.hashCode, qtdeCompras.toDouble)
    }
    .cache()

  val vendasPorPeca = rawRatings map { rating => (rating.product, rating.rating) } reduceByKey { _ + _ }

  val qtdePecas = vendasPorPeca count

  val topVendidas = vendasPorPeca
    .sortBy(_._2, ascending = false)
    .map { _._1 }
    .take((qtdePecas * 0.50).asInstanceOf[Int])
    .toSet

  // método de particionamento
  def particionarComprasCliente[A](colecao: Iterable[A], numeroParticoes: Int): Iterator[Iterable[A]] = {
    val (quot, rem) = (colecao.size / numeroParticoes, colecao.size % numeroParticoes)
    val (smaller, bigger) = colecao.splitAt(colecao.size - rem * (quot + 1))
    smaller.grouped(quot) ++ bigger.grouped(quot + 1)
  }


  val clientesBuckets = rawRatings.filter(rating => topVendidas.contains(rating.product)).groupBy(_.user) //agrupa ratings por cliente
    .filter { case (_, clienteRatings) => clienteRatings.size >= NUMBER_OF_BUCKETS } //somente clientes com pelo menos 10 ratings
    .map { case (clienteID, clienteRatings) =>
    (clienteID, particionarComprasCliente(clienteRatings, NUMBER_OF_BUCKETS).toList) //divide lista de ratings do usuário em 10 pedaços
  }.cache

  val scores = new ListBuffer[(Int, (Double, Double))]

  //itera buckets
  0 until NUMBER_OF_BUCKETS foreach { bucketID =>

    //println(s"Bucket ID:           $bucketID")

    //seleciona dados para treino
    val trainingData: RDD[(Int, List[Iterable[Rating]])] = clientesBuckets map { case (clienteID, buckets) => (
      clienteID, buckets.take(bucketID) ++ buckets.drop(bucketID + 1))
    }

    //seleciona dados para teste
    val testData: RDD[(Int, Array[Rating])] = clientesBuckets map { case (clienteID, buckets) =>
      (clienteID, buckets(bucketID).toArray)
    } cache()

    //formata os dados de treino para o formato esperado pelo algoritmo
    val preparedTrainingData: RDD[Rating] = trainingData flatMap { case (_, buckets) => buckets } flatMap { ratings => ratings } cache()

    // efetua o teino e gera o modelo
    val model = ALS.trainImplicit(preparedTrainingData, 10, 10)

    val listTopRecomendacoes = List(1, 5, 10, 15, 20, 25, 30)
    val to30recomendacoes = model.recommendProductsForUsers(listTopRecomendacoes.max).cache()

    listTopRecomendacoes.foreach(topK => {

      val recomendacoes = to30recomendacoes.map({ case (user, recs) => (user, recs.take(topK)) }).flatMap(_._2)
        .map({ case Rating(cliente, produtoRecomendado, _) => (cliente, produtoRecomendado) }).cache()

      val teste = testData.flatMap(_._2)
        .map({ case Rating(cliente, produtoComprado, _) => (cliente, produtoComprado) }).cache()

      val intersectionCount = recomendacoes.intersection(teste).count().toDouble

      //Percentual de itens recomendados que foram comprados
      val precision = intersectionCount / recomendacoes.count().toDouble

      //Percentual de itens de teste que foram recomendados
      val recall = intersectionCount / teste.count().toDouble

      //println(s"[ALS] Precision top-$topK:\t$precision")
      //println(s"[ALS] Recall top-$topK:\t$recall")

      scores.append((topK, (precision, recall)))

    })
    //println()

  }

  scores.groupBy({ case (topk, (_, _)) => topk }).toSeq.sortBy(_._1).foreach({ case (topK, list) =>
    val mediaPrecisao = list.map({ case (_, (precision, _)) => precision }).sum / list.length.toDouble
    val mediaRecall = list.map({ case (_, (_, recall)) => recall }).sum / list.length.toDouble

    println(s"[ALS] top-$topK")
    println(s"[ALS] Média precisão:\t${formatter.format(mediaPrecisao)}")
    println(s"[ALS] Média recall:\t${formatter.format(mediaRecall)}")

    val f1Medida = (mediaPrecisao * mediaRecall * 2.0) / (mediaPrecisao + mediaRecall)
    println(s"[ALS] F-score:\t${formatter.format(f1Medida)}")

    println

  })


  sc.stop()

}
