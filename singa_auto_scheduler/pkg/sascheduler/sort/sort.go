package sort

import framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"

func Less(podInfo1, podInfo2 *framework.PodInfo) bool {
	return GetPodPriority(podInfo1) > GetPodPriority(podInfo2)
}

func GetPodPriority(podInfo *framework.PodInfo) uint {
	var pod uint = 0
	// if the pod has such label, use them to sort
	if _, ok := podInfo.Pod.Labels["Level"]; ok {
		switch podInfo.Pod.Labels["Level"] {
		case "High":
			pod = 3
		case "Medium":
			pod = 2
		case "Low":
			pod = 1
		}
	}
	return pod
}
