/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.singa.auto.monitor.stream
import com.singa.auto.common.store.Kafka
import com.singa.auto.common.udf.SparkUdfs
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, HasOffsetRanges, KafkaUtils}


abstract class BaseStream extends Serializable {

  def createSparkStreamContext(
                               sparkConf: SparkConf,
                               topics: String,
                               brokers:String,
                               kafkaConsumerGroup:String,
                               batchTime: Long): Unit = {
    val ssc: StreamingContext = new StreamingContext(sparkConf, Seconds(batchTime))

    val topicsSet = topics.split(",").toSet
    val kafkaParams = Kafka.getKafkaParams(brokers, kafkaConsumerGroup)

    // Create kafka stream.
    val messages = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topicsSet, kafkaParams)
    )

    // Create spark session for each instance
    var spark = SparkSession.builder.config(sparkConf).getOrCreate()
    spark = new SparkUdfs(spark).regist()
    spark.sparkContext.setLogLevel("WARN")

    val dStream = messages.map(_.value())

    consumer(spark, dStream)

    // update offset
    messages.foreachRDD {
      rdd =>
        val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
        offsetRanges.foreach { x =>
          println("kafka message ==> partition:" + x.partition + "--from: " + x.fromOffset + "--to: " + x.untilOffset)
        }
        // some time later, after outputs have completed
        messages.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
    }

    ssc.start()
    ssc.awaitTermination()
  }

  /**
   * consumer method
   * @param spark:SparkSession
   * @param dStream: stream message
   */
  def consumer(spark:SparkSession, dStream: DStream[String]): Unit={

  }

}
