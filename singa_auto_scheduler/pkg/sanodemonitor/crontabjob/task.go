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
package crontabjob

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/collection"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/controller"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
	"github.com/robfig/cron"
	"sync"
)

func Update(c *cron.Cron, name string) {

	// every 1 min, update node info and label
	if err := c.AddFunc("*/1 * * * * ?", func() {
		controller.UpdateNodeInfo(collection.Mode)
		controller.UpdateNodeLabel(name)
	}); err != nil {
		log.ErrPrint("Update", err)
	}
}

func Start(c *cron.Cron, w *sync.WaitGroup) {
	// Add 1 to the counter
	w.Add(1)
	c.Start()
}
