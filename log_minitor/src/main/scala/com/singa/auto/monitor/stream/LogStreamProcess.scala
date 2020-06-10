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
import com.singa.auto.Config
import com.singa.auto.common.configuration.SparkCfg
import com.singa.auto.monitor.processer.LogEvents
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.streaming.dstream.DStream

class LogStreamProcess extends  BaseStream {

  override
  def consumer(spark:SparkSession, dStream: DStream[String]): Unit= {
    println("LogToDwStream ---- consumer")
    dStream.print()
    dStream.foreachRDD{
      rdd =>
        if (!rdd.isEmpty()){
          val logs: DataFrame = spark.read.json(rdd)
          logs.show(false)
          LogEvents.stream(spark, logs)
        }
    }
  }
}

object LogStreamProcess {

//  val SPARK_APP_NAME:String = Config.SPARK_APP_NAME
//  val RUNNING_MODEL:String = Config.RUNNING_MODEL

  val TOPIC:String  = Config.TOPIC
  val BROKER:String  = Config.BROKER
  val CONSUME_GROUP:String  = Config.CONSUME_GROUP
  val ES_HOST:String  = Config.ES_HOST
  val ES_PORT:String  = Config.ES_PORT
  val TIME:String  = Config.TIME

  def main(args: Array[String]):Unit ={

    println("In LogStreamProcess")
//    var sparkConf = new SparkConf().setMaster(RUNNING_MODEL).setAppName(SPARK_APP_NAME);
    var sparkConf = new SparkConf()

    sparkConf.set("spark.es.nodes",ES_HOST)
    sparkConf.set("spark.es.port",ES_PORT)

    sparkConf = SparkCfg.setConf(sparkConf)

    // ssc
    val logsStream = new LogStreamProcess
    logsStream.createSparkStreamContext(sparkConf, TOPIC, BROKER, CONSUME_GROUP, Integer.parseInt(TIME));

  }
}
