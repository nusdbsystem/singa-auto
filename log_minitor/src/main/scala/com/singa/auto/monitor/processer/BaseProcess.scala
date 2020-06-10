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

package com.singa.auto.monitor.processer
import com.singa.auto.Config
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.elasticsearch.spark.sql._

abstract class BaseProcess extends Serializable {

  def filter(k: String):Boolean={
    true
  }

  def sparkSQl(dataFrame: DataFrame):String ={
    ""
  }

  def tempView(table: String): String={
    (table.replace(".", "_") + "_VIEW").toUpperCase;

  }

  def process(spark: SparkSession, dataFrame: DataFrame, table: String, data:String):Unit={
    try{

      dataFrame.createOrReplaceTempView(tempView(table));

      val sql = sparkSQl(dataFrame: DataFrame)

      val datas = spark.sql(sql)

      println("-------------Saving Dataframe to ES-------------")
      datas.show(false)
      datas.saveToEs(Config.ES_INDEX)

    }
    catch {
      case ex: Exception =>
        println("\n Exception in processing SparkSQL:" + sparkSQl(dataFrame: DataFrame) + "\n")
        println("\n Exception in processing SparkSQL:" + ex.getMessage + "\n")

    }

  }

}
