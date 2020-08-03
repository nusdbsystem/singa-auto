package sascheduler

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

func Register() *cobra.Command {
	return app.NewSchedulerCommand(
		app.WithPlugin(collection.AppName, New),
	)
}
