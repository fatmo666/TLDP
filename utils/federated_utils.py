import syft as sy
import torch

def get_federated_dataloaders(train_dataset, batch_size=64):
    hook = sy.TorchHook(torch)
    woker1 = sy.VirtualWorker(hook, id="worker1")
    woker2 = sy.VirtualWorker(hook, id="worker2")
    woker3 = sy.VirtualWorker(hook, id="worker3")

    datasets_split = torch.utils.data.random_split(train_dataset, [20000, 20000, 20000])
    woker1_dataset = datasets_split[0].federate((woker1, ))
    woker2_dataset = datasets_split[1].federate((woker2, ))
    woker3_dataset = datasets_split[2].federate((woker3, ))

    federated_train_loader = sy.FederatedDataLoader(
        woker1_dataset + woker2_dataset + woker3_dataset, batch_size=batch_size, shuffle=True
    )

    return federated_train_loader, worker1, worker2, worker3