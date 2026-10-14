import torch.utils.data as torch_data
import torchvision
import torch
from torchvision.transforms import transforms

from models.simple import SimpleNet
from tasks.task import Task


def _replace_dataset_download_host(dataset_class, base_url):
    """Keep torchvision's filenames/checksums, but use a working download host."""
    if hasattr(dataset_class, 'mirrors'):
        dataset_class.mirrors = [base_url]
    else:
        dataset_class.resources = [
            (base_url + url.rsplit('/', 1)[-1], md5)
            for url, md5 in dataset_class.resources
        ]


# torchvision 0.8.2 points MNIST at an obsolete URL that now returns HTTP 404.
_replace_dataset_download_host(
    torchvision.datasets.MNIST,
    'https://ossci-datasets.s3.amazonaws.com/mnist/'
)
_replace_dataset_download_host(
    torchvision.datasets.FashionMNIST,
    'https://fashion-mnist.s3.eu-central-1.amazonaws.com/'
)


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        self.load_mnist_data()
    
    def load_mnist_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        original_train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        
        clean_size = round(self.params.clean_ratio * len(original_train_dataset))
        train_size = len(original_train_dataset) - clean_size
        self.train_dataset, self.clean_dataset = torch_data.random_split(original_train_dataset,
                                                                         lengths=[train_size, clean_size],
                                                                         generator=torch.Generator().manual_seed(self.params.random_seed))
        if self.params.clean_set_dataset == 'FashionMNIST':
            self.clean_dataset = torchvision.datasets.FashionMNIST(root=self.params.data_path,
                                                                   train=True,
                                                                   download=True, 
                                                                   transform=transform_train)
            tot = len(self.clean_dataset)
            self.clean_dataset, _ = torch_data.random_split(self.clean_dataset,
                                                            lengths=[clean_size, tot-clean_size], 
                                                            generator=torch.Generator().manual_seed(self.params.random_seed))
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                    batch_size=self.params.batch_size,
                    shuffle=True,
                    num_workers=0)
        elif self.params.clean_set_dataset is None and self.params.clean_classes is not None:
            clean_indices = []
            for cls in self.params.clean_classes:
                for i in range(clean_size):
                    if self.clean_dataset[i][1] == cls:
                        clean_indices.append(i)

            sampler = torch_data.SubsetRandomSampler(clean_indices, generator=torch.Generator().manual_seed(self.params.random_seed))
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size,
                                                      sampler=sampler,
                                                      num_workers=0)
        else:
            self.clean_loader = torch_data.DataLoader(self.clean_dataset,
                                                      batch_size=self.params.batch_size,
                                                      shuffle=True,
                                                      num_workers=0)
                                                      
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def build_model(self):
        return SimpleNet(num_classes=len(self.classes))
