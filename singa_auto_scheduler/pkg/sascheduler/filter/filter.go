package filter

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func NodeHasGPU(node *nodeinfo.NodeInfo) bool {
	if _, ok := node.Node().Labels["samonitor/Gpu"]; ok {
		if node.Node().Labels["samonitor/Gpu"] == "True" {
			return true
		}
	}
	return false
}

func NodeGPUHealth(node *nodeinfo.NodeInfo) bool {
	if node.Node().Labels["samonitor/Health"] == "Healthy" {
		return true
	}
	return false
}

func CheckGPUStatus(node *nodeinfo.NodeInfo) (bool, string) {
	var msg = ""
	if NodeHasGPU(node) {
		if NodeGPUHealth(node) {
			return true, msg
		}
		return false, "GPU Unhealthy"
	}
	return false, "No GPU"
}

func NodeHasGPUNumber(node *nodeinfo.NodeInfo) bool {
	_, ok := node.Node().Labels["samonitor/Number"]
	return ok
}

func NodeHasFreeMemory(node *nodeinfo.NodeInfo) bool {
	_, ok := node.Node().Labels["samonitor/MaxFreeMemory"]
	return ok
}

type PodInfo struct {
	Pod  *v1.Pod
	Node *nodeinfo.NodeInfo
}

// private methods
func (p *PodInfo) podNeedMemory() bool {
	_, ok := (*p).Pod.Labels["samonitor/MaxFreeMemory"]
	return ok
}

// private methods
func (p *PodInfo) podNeedGPUNumber() bool {
	_, ok := (*p).Pod.Labels["samonitor/Number"]
	return ok
}

// public methods
func (p *PodInfo) PodFitsMemory() bool {
	if p.podNeedMemory() {
		if NodeHasFreeMemory((*p).Node) {
			return collection.StrToUInt((*p).Node.Node().Labels["samonitor/MaxFreeMemory"]) >=
				collection.StrToUInt((*p).Pod.Labels["samonitor/MaxFreeMemory"])
		}
		return false
	}
	return true
}

// public methods
func (p *PodInfo) PodFitsNumber() bool {
	if p.podNeedGPUNumber() {
		if NodeHasGPUNumber((*p).Node) {
			return collection.StrToUInt((*p).Node.Node().Labels["samonitor/Number"]) >=
				collection.StrToUInt((*p).Pod.Labels["samonitor/Number"])
		}
		return false
	}
	return true
}
