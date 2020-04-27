import torch
import numpy as np

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

def save_checkpoint(epoch, model, optimizer):
    '''
        Save model checkpoint
    '''
    state = {'epoch': epoch, "model_weights": model, "optimizer": optimizer}
    filename = "model_state.pth.tar"
    torch.save(state, filename)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def eval(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %2f %%' % (100 * correct / total))
    
def detail_eval(model, test_loader):
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for images, labels in test_loader:
            bs, ncrops, c, h, w = np.shape(images)
            images = images.view(-1, c, h, w)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(7):
        print('Accuracy of %5s : %2f (%d / %d) %%' % (
            classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))