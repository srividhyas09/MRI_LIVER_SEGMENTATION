import time
import copy

# torch libs imports
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# imports from local
from Seg_preprocessing import *
import matplotlib.pyplot as plt
from dual_tail_net import *


def train_model(model, dataset, criterion, optimizer, device, num_epochs=3, is_inception=False):
    since = time.time()

    test_loss = []
    val_loss = []
    best_loss = 0
    best_model_wts = None
    batchsize = 1

    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.7),
                                                                   len(dataset) - int(len(dataset) * 0.7)])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloaders = valid_loader

            running_loss = 0.0

            # Iterate over data.
            for batch_index, sample in enumerate(dataloaders):

                inputs = sample['image']
                mask = sample['mask']

                inputs = inputs.type(torch.FloatTensor)
                mask = mask.type(torch.FloatTensor)

                inputs = inputs.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, mask)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                l1_penalty = torch.nn.L1Loss(size_average=False)
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += l1_penalty(param, target=torch.zeros_like(param))
                factor = 1e-4  # lambda
                #loss += factor * reg_loss

                running_loss += loss.item() * inputs.size(0)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch

            epoch_loss = running_loss / len(dataloaders.dataset)
            if phase == 'train':
                test_loss.append(epoch_loss)
            else:

                if len(val_loss) == 1:
                    best_loss = val_loss[-1]

                if val_loss != []:
                    if best_loss >= epoch_loss:
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_loss = epoch_loss
                val_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    plt.plot(test_loss)
    plt.show()
    plt.plot(val_loss)
    plt.show()
    # load best model weights
    # model.load_state_dict()
    return best_model_wts


if __name__ == '__main__':
    data_transform_augment = transforms.Compose([ResizePadCrop(), ToTensor()])

    batch_size = 1
    dataset = Preprocessing('ground_truth/no_liver_df_t2.csv', batch_size,  transform=data_transform_augment)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Dual(n_channels=1, n_classes=1, bilinear=True)
    model = model.to(device)

    learning_rate = 1e-4
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 60
    trained_model = train_model(model, dataset, criterion, optimizer, device, num_epochs=epochs, is_inception=False)

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(trained_model, 'saved_models/UnetBCE_normalized.pt')
