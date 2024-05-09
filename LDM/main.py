import resnet
from generate_watermark import generate_watermark
from torchvision import transforms, datasets
import torch
import torchvision
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='DiffusionShield Training')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--budget', default=0.007, type=float)
    parser.add_argument('--lr_train', default=2e-3, type=float)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--augment_type', default='None', type=str, help='applying corruption to improve robustness, choose from "None", "grey", "blur", "noise", and "jpeg".')
    parser.add_argument('--epoch_num', default=40, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--B', default=2, type=int, help='Transfer to B-nary sequence')
    parser.add_argument('--trainset', default='./train', type=str, help='cifar10_bird_train or specify directory to the training set')
    parser.add_argument('--testset', default='./test', type=str, help='cifar10_bird_train or specify directory to the test set')
    parser.add_argument('--patches_save_path', default='./trained_patches/wm_patches.pt', type=str, help='specify the save path of the trained patches')
    parser.add_argument('--model_save_path', default='./trained_model/classifier.pt', type=str, help='specify the save path of the trained classifier')

    args = parser.parse_args()
    # Initialize the model

    model = resnet.resnet18()

    # Define transformations
    trans = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()])
    
    trainset = datasets.ImageFolder(root=args.trainset, transform=trans)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.ImageFolder(root=args.testset, transform=trans)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize and run the defense mechanism
    defense = generate_watermark(model, 'cuda', epsilon=args.epsilon, lr_train=args.lr_train, epoch_num=args.epoch_num,  patch_size=args.patch_size, budget = args.budget, \
                                 image_size=args.image_size, augment_type=args.augment_type, B=args.B, patches_save_path=args.patches_save_path, model_save_path=args.model_save_path)

    defense.generate_wm(train_loader, test_loader)

if __name__ == "__main__":
    main()
