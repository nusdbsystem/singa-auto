package filter

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"strings"
)

func MatchGpu2Pod(
	pod *v1.Pod,
	nodes []*v1.Node,
	filteredNodesStatuses framework.NodeToStatusMap) {

	defer func() {
		if err := recover(); err != nil {
			klog.V(3).Infof("ErrorMsg in MatchGpu2Pod: %v", err)
		}
	}()

	matchGpu2Pod(pod, nodes, filteredNodesStatuses)
}

func matchGpu2Pod(
	pod *v1.Pod,
	nodes []*v1.Node,
	filteredNodesStatuses framework.NodeToStatusMap) {

	var selectGpu string
	var selectNodeName string
	var maxFreeMemory int64 = 0

	for _, n := range nodes {
		if filteredNodesStatuses[n.GetName()].IsSuccess() {

			// if the node is filtered, get needed node info
			nodeName := n.Labels[collection.NodeSelectorLabel]
			gpuSummary := n.Labels["samonitor/GpuSummary"]

			for _, GpuMemory := range strings.Split(gpuSummary, ".") {
				var gpu string = strings.Split(GpuMemory, "_")[0]
				var freeMemory int64 = collection.StrToInt64(strings.Split(GpuMemory, "_")[1])

				// find the max gpu device number and it's corresponding node
				if freeMemory > maxFreeMemory {
					selectNodeName = nodeName
					selectGpu = gpu
				}
			}
		}
	}

	// assign env to the pod env
	containers := pod.Spec.Containers

	for _, container := range containers {

		envVar := v1.EnvVar{Name: "NVIDIA_VISIBLE_DEVICES", Value: selectGpu}
		container.Env = append(container.Env, envVar)
	}
	// assign node label to the pod env
	pod.Spec.NodeSelector[collection.NodeSelectorLabel] = selectNodeName

}
