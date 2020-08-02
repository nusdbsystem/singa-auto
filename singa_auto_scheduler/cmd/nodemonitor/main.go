//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
package main

import (
	"fmt"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/collection"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/controller"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/crontabjob"
	"github.com/robfig/cron"
	"os"
	"sync"
)

func main() {
	// wg has Add(), Done(), Wait()
	var wg sync.WaitGroup
	fmt.Println("Init Mode, Node from env")
	// Get info from envs
	collection.Node = os.Getenv("NODE_NAME")
	collection.Mode = os.Getenv("MODE")

	// Init the node status
	fmt.Println("Init Node status")
	controller.InitNodeInfo(collection.Mode)
	controller.PrintNodeInfo()
	controller.InitInClusterConfig()

	// Update node gpu info every 1 min
	c := cron.New()
	fmt.Println("Init Crontabjob")
	crontabjob.Update(c, collection.Node)

	// must provide the address of wg, otherwise deadlock
	// // Start the cron scheduler in its own go-routine,
	// or no-op if already started.
	crontabjob.Start(c, &wg)

	//stop the daemonset
	defer controller.CleanNodeLabel(collection.Node)
	defer c.Stop()

	// main process keep waiting, until counter inside wg is 0
	// dont call
	wg.Wait()
}
