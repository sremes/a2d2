"""Implements image segmentation models and neural network layers."""
# pylint: disable=too-few-public-methods
# pylint: disable=bad-continuation
import tensorflow as tf


class PositionalNormalization(tf.keras.layers.Layer):
    """Positional normalization removes the mean and standard deviation over spatial dimensions."""

    def __init__(self, epsilon=1e-5, axis=3):
        """Sets parameters of the layer.

        Args:
            epsilon: a small value to stabilize the computations
            axis: the axis along which to compute the moments (the default applies to channels in BxHxWxC)
        """
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis

    def call(self, inputs):
        """

        Args:
            inputs:

        Returns:
            The input with the moments removed, and the computed mean and standard deviation that can be passed
            to a later `MomentShortcut` layer.
        """
        mean, variance = tf.nn.moments(inputs, axes=[self.axis], keepdims=True)
        standard_deviation = tf.sqrt(variance + self.epsilon)
        output = (inputs - mean) / standard_deviation
        return output, mean, standard_deviation


class MomentShortcut(tf.keras.layers.Layer):
    """The moment shortcut applies mean and standard deviation from a previous positional normalization layer."""

    def __init__(self, mean, standard_deviation):
        """Initialize the moment shortcut with given mean and standard deviation.

        Args:
            mean: the mean computed at a previous layer
            standard_deviation: the standard deviation computed at a previous layer
        """
        super().__init__()
        self.mean = mean
        self.standard_deviation = standard_deviation

    def call(self, inputs):
        """Applies the moments to the inputs.

        Args:
            inputs: a tensor with shape matching the mean and standard deviation of the layer

        Returns:
            The input with the moments applied to it
        """
        return inputs * self.standard_deviation + self.mean


class UNetConvolutionalBlock(tf.keras.layers.Layer):
    """Convolutional block used in the encoder part of the UNet semantic segmentation model."""

    def __init__(self, filters, kernel_size=(2, 2), activation="relu", kernel_initializer="he_normal"):
        """Set the parameters of the convolutional layer."""
        super().__init__()
        # basic parameters
        self.activation = activation
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        # layers
        self.convolution = None
        self.positional_normalization = None
        self.pooling = None

    def build(self, input_shape):
        """Initialize the Keras layers."""
        super().build(input_shape)
        self.convolution = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            activation=self.activation,
            padding="same",
        )
        self.positional_normalization = PositionalNormalization()
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs):
        """Apply the convolution and positional normalization with pooling to the given inputs."""
        convolved_output = self.convolution(inputs)
        normalized_output, mean, standard_deviation = self.positional_normalization(convolved_output)
        pooled_output = self.pooling(normalized_output)
        return pooled_output, normalized_output, mean, standard_deviation


class UNetTransposedConvolutionalBlock(tf.keras.layers.Layer):
    """Transposed convolutional block for the decoder part of the UNet semantic segmentation model."""

    # pylint: disable=too-many-arguments
    def __init__(self, filters, kernel_size=(2, 2), activation="relu", strides=(2, 2), kernel_initializer="he_normal"):
        """Set the parameters of the convolutional layer."""
        super().__init__()
        # parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        # layers
        self.transposed_convolution = None
        self.convolution = None

    def build(self, input_shape):
        """Initialize the convolutional layers."""
        super().build(input_shape)
        self.transposed_convolution = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_initializer=self.kernel_initializer,
            padding="same",
        )
        self.convolution = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding="same",
        )

    def call(self, inputs):
        """Apply the convolutional layers to the given inputs.

        Args:
            inputs: a list containing the input, the output from the corresponding decoder block and the mean/std
                    from the corresponding positional normalization layer
        """
        inputs, copied_feature_map, mean, standard_deviation = inputs
        upscaled_x = self.transposed_convolution(inputs)
        # apply the moments from the corresponding block of the encoder
        with_moments = MomentShortcut(mean, standard_deviation)(upscaled_x)
        concatenated = tf.keras.layers.concatenate([copied_feature_map, with_moments])
        convolved = self.convolution(concatenated)
        return convolved


class UNet(tf.keras.models.Model):  # pylint: disable=too-many-instance-attributes
    """UNet-like model with positional normalization and transposed convolutions in the decoder."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_classes=1,
        output_activation="sigmoid",
        kernel_size=(2, 2),
        first_layer_filters=64,
        num_layers=4,
        activation="relu",
        kernel_initializer="he_normal",
    ):
        """Set the parameters for the layers used in the model."""
        super().__init__()
        # Save basic parameters
        self.num_classes = num_classes
        self.output_activation = output_activation
        self.activation = activation
        self.filters = [first_layer_filters * (2 ** i) for i in range(num_layers)]
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        # Initialize structures used to save layers
        self.convolutional_blocks = []
        self.transposed_blocks = []
        self.convolution = None
        self.output_convolution = None

    def build(self, _input_shape):
        """Initialize the convolutional blocks used in the model."""
        for filters in self.filters:
            self.convolutional_blocks.append(
                UNetConvolutionalBlock(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                )
            )
        for filters in reversed(self.filters):
            self.transposed_blocks.append(
                UNetTransposedConvolutionalBlock(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                )
            )
        self.convolution = tf.keras.layers.Conv2D(
            filters=self.filters[-1],
            kernel_size=self.kernel_size,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding="same",
        )
        self.output_convolution = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            padding="same",
        )

    def call(self, inputs):
        """Apply the model on the given input image."""
        # save outputs and moments for the decoder blocks
        means = []
        standard_deviations = []
        outputs = []
        # apply the decoder blocks to the inputs
        output = inputs  # set the inputs for the first iteration
        for block in self.convolutional_blocks:
            output, unpooled_output, mean, standard_deviation = block(output)
            outputs.append(unpooled_output)
            means.append(mean)
            standard_deviations.append(standard_deviation)
        # run a basic convolution before starting the upscaling
        output = self.convolution(output)
        # upscale with the transposed convolutions
        for block, *input_list in zip(
            self.transposed_blocks, reversed(outputs), reversed(means), reversed(standard_deviations)
        ):
            output = block([output, *input_list])
        # create the output
        output = self.output_convolution(output)
        return output
