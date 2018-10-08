from basenets import net
from basenets import alexnet
from basenets import resnet50
from basenets import mobilenet
__all__ = ['Net', 'AlexNet', 'MobileNet', 'ResNet50', 'utils']

Net = net.Net
AlexNet = alexnet.AlexNet
MobileNet = mobilenet.MobileNet
ResNet50 = resnet50.ResNet50