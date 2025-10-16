import torch.nn as nn

class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=1):
		super(ConvBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)

class ConvNet(nn.Module):
	'''
	Network that composed by layers of ConvBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class Discriminator(nn.Module):
	'''
	Adversary to discriminate the cover image and the encoded image
	'''

	def __init__(self, blocks=4, channels=64):
		super(Discriminator, self).__init__()

		self.layers = nn.Sequential(
			ConvNet(3, channels, blocks),
			nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

		self.linear = nn.Linear(channels, 1)

	def forward(self, image):
		x = self.layers(image)
		x.squeeze_(3).squeeze_(2)
		x = self.linear(x)
		return x
