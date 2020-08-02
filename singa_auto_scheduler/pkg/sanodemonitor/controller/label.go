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

package controller

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/collection"
	"github.com/naili-xing/singa_auto_scheduler/pkg/sanodemonitor/log"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"strings"
)

func UpdateNodeLabel(name string) {
	// convert node info instance to map
	label := collection.NodeInfoToMap()

	client, err := kubernetes.NewForConfig(Config)
	if err != nil {
		panic(err)
	}
	node, err := client.CoreV1().Nodes().Get(name, v1.GetOptions{})
	if err != nil {
		log.ErrPrint("UpdateNodeLabel", err)
	} else {
		if DoesUpdateNodeLabel(name) {
			log.Print("Label Changed, ready to update.")
			nodeLabel := node.GetLabels()
			for i, v := range label {
				nodeLabel[i] = v
			}
			// update label
			node.SetLabels(nodeLabel)
			if _, err := client.CoreV1().Nodes().Update(node); err != nil {
				log.ErrPrint("UpdateNodeLabel", err)
			}
		}
	}
}

func CleanNodeLabel(name string) {
	client, err := kubernetes.NewForConfig(Config)
	if err != nil {
		panic(err)
	}
	node, err := client.CoreV1().Nodes().Get(name, v1.GetOptions{})
	if err != nil {
		log.ErrPrint("CleanNodeLabel", err)
	} else {
		nodeLabel := node.GetLabels()
		for i := range nodeLabel {
			if strings.Contains(i, "SaMonitor/") {
				delete(nodeLabel, i)
			}
		}
		node.SetLabels(nodeLabel)
		if _, err := client.CoreV1().Nodes().Update(node); err != nil {
			log.ErrPrint("CleanNodeLabel", err)
		}
	}
}

// check if the node label is needed to update
func DoesUpdateNodeLabel(name string) bool {
	client, err := kubernetes.NewForConfig(Config)
	if err != nil {
		panic(err)
	}
	node, err := client.CoreV1().Nodes().Get(name, v1.GetOptions{})
	if err != nil {
		log.ErrPrint("DoesUpdateNodeLabel", err)
	} else {
		nodeLabel := node.GetLabels()
		for k, v := range collection.NodeInfoToMap() {

			if value, ok := nodeLabel[k]; ok {
				// if the current label of node
				// is not the same as latest node info label
				if value != v {
					return true
				}
				// if nodeLabel dont have such key,
			} else {
				return true
			}
		}
	}
	return false
}
