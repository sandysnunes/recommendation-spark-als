package com.sandysnunes

import org.apache.spark.SparkConf
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SaveMode, SparkSession}

/**
	* <pre>
	* clean package
	* </pre>
	*
	* <p>Exemplo de execução:</p>
	* <pre>
	* spark-submit \
	* --master spark://spark:7077 \
	* --class com.sandysnunes.FPGrowthVendaPecas \
	* recommendation-spark-als-1.0-SNAPSHOT.jar \
	* TCC/2017.txt \
	* 0.01 \
	* 0.07
	* </pre>
	*/
object RegrasGeracaoFPGrowthVendaPecas extends App {

	/*Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
	Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)*/

	val conf = new SparkConf().setAppName("RegrasGeracaoFPGrowthVendaPecas")
		.setMaster("local[*]") //comentar linha para spark submit

	val session = SparkSession.builder().config(conf).getOrCreate()
	val sc = session.sparkContext
	val sqlContext = session.sqlContext

	import sqlContext.implicits._

	val confiancaMinima: Double = 0.2d

	val data: RDD[String] = sc.textFile(args(0))
	val transactions: RDD[Array[String]] = data.map(linha => linha.trim.split('|').distinct).cache()

	// 0.001, 0.002, 0.003, 0.004, 0.005 ... 0.200
	val supportList = 1 to 200 map { support => support.toDouble / 1000.0 }

	val allModelsResults = supportList map(support => {
		val fpGrowth = new FPGrowth().setMinSupport(support).setNumPartitions(10)
		val modelo = fpGrowth.run(transactions)

		val frequencias = modelo.freqItemsets
			.sortBy(itemSet => itemSet.freq, ascending = false)
			.map { itemSet => (support.toString, "FREQUENCIA", itemSet.items.mkString(", "), "", itemSet.freq.toString) }

		val regras = modelo.generateAssociationRules(confiancaMinima)
			.sortBy(rule => rule.confidence, ascending = false)
			.map { rule => (support.toString, "REGRA", rule.antecedent.mkString(", "), rule.consequent.mkString(", "), rule.confidence.toString) }

		frequencias union regras
	})


	sc.union(allModelsResults)
		.toDF( colNames = "SUPPORT", "TYPE", "ANTECEDENT/ITEMSET", "CONSEQUENT", "CONFIANCA/FREQ")
		.coalesce(numPartitions = 1)
		.write
		.mode(SaveMode.Overwrite)
		.format( source = "com.databricks.spark.csv")
		.option("header", "true")
		.option("delimiter", ";")
		.save(path = "regras-geradas")

		sc.stop()
}
