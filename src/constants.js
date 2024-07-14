export const modelDescriptions = [
  {
    name: '1l_average',
    purpose: 'Calculates the average value of input data.',
    usage: 'Used for obtaining mean values in data processing.',
    howToUse: 'Input data into the model to get the average value as output.'
  },
  {
    name: '1l_powf',
    purpose: 'Computes the power of input values.',
    usage: 'Useful for exponential operations on data.',
    howToUse: 'Provide base and exponent values as input to get the powered output.'
  },
  {
    name: 'accuracy',
    purpose: 'Measures the accuracy of predictions.',
    usage: 'Commonly used in classification tasks to evaluate model performance.',
    howToUse: 'Compare predicted labels with true labels to get the accuracy score.'
  },
  {
    name: 'layernorm',
    purpose: 'Applies layer normalization to input data.',
    usage: 'Used in neural networks to stabilize and speed up training.',
    howToUse: 'Pass input data through the model to get normalized output.'
  },
  {
    name: 'quantize_dequantize',
    purpose: 'Quantizes and then dequantizes the input data.',
    usage: 'Used for model compression and to reduce precision for faster computations.',
    howToUse: 'Input data is quantized to lower precision and then dequantized back.'
  },
  {
    name: '1l_batch_norm',
    purpose: 'Applies batch normalization to input data.',
    usage: 'Improves training speed and stability of neural networks.',
    howToUse: 'Input data is normalized across the batch dimension.'
  },
  {
    name: '1l_prelu',
    purpose: 'Applies Parametric ReLU activation function.',
    usage: 'Used in neural networks to introduce non-linearity.',
    howToUse: 'Pass input data through the model to apply PReLU activation.'
  },
  {
    name: 'arange',
    purpose: 'Generates a sequence of numbers.',
    usage: 'Commonly used to create arrays of numbers for indexing or iteration.',
    howToUse: 'Specify start, stop, and step values to generate the sequence.'
  },
  {
    name: 'lenet_5',
    purpose: 'A convolutional neural network model for digit recognition.',
    usage: 'Used for image classification tasks, particularly on MNIST dataset.',
    howToUse: 'Input image data to get classification predictions.'
  },
  {
    name: 'random_forest',
    purpose: 'An ensemble learning method using multiple decision trees.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data to make predictions on new data.'
  },
  {
    name: '1l_concat',
    purpose: 'Concatenates input data along a specified axis.',
    usage: 'Useful for combining multiple data arrays.',
    howToUse: 'Input data arrays and specify the axis for concatenation.'
  },
  {
    name: '1l_relu',
    purpose: 'Applies the ReLU activation function.',
    usage: 'Introduces non-linearity in neural networks.',
    howToUse: 'Pass input data through the model to apply ReLU activation.'
  },
  {
    name: 'bitshift',
    purpose: 'Performs bitwise shift operations on input data.',
    usage: 'Used for low-level data manipulation and optimization.',
    howToUse: 'Specify the input data and the number of positions to shift.'
  },
  {
    name: 'less',
    purpose: 'Compares two values and returns true if the first is less than the second.',
    usage: 'Used for conditional operations and branching.',
    howToUse: 'Input two values to get a boolean comparison result.'
  },
  {
    name: 'reducel1',
    purpose: 'Computes the L1 norm of input data, reducing along specified axes.',
    usage: 'Commonly used in regularization techniques.',
    howToUse: 'Input data and specify axes to compute the L1 norm.'
  },
  {
    name: '1l_conv',
    purpose: 'Applies a single layer convolution operation.',
    usage: 'Used in convolutional neural networks for feature extraction.',
    howToUse: 'Input data through the convolutional layer to get convolved output.'
  },
  {
    name: '1l_reshape',
    purpose: 'Reshapes the input data to a specified shape.',
    usage: 'Used for data preprocessing and model input adjustments.',
    howToUse: 'Specify the new shape to reshape the input data.'
  },
  {
    name: 'bitwise_ops',
    purpose: 'Performs bitwise operations on input data.',
    usage: 'Used for binary data manipulation and optimization.',
    howToUse: 'Specify the bitwise operation and input data.'
  },
  {
    name: 'lightgbm',
    purpose: 'A gradient boosting framework for machine learning.',
    usage: 'Used for classification, regression, and ranking tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'reducel2',
    purpose: 'Computes the L2 norm of input data, reducing along specified axes.',
    usage: 'Commonly used in regularization techniques.',
    howToUse: 'Input data and specify axes to compute the L2 norm.'
  },
  {
    name: '1l_conv_transpose',
    purpose: 'Applies a single layer transposed convolution operation.',
    usage: 'Used in neural networks for upsampling operations.',
    howToUse: 'Input data through the transposed convolutional layer to get upsampled output.'
  },
  {
    name: '1l_sigmoid',
    purpose: 'Applies the sigmoid activation function.',
    usage: 'Introduces non-linearity in neural networks, often used in binary classification.',
    howToUse: 'Pass input data through the model to apply sigmoid activation.'
  },
  {
    name: 'blackman_window',
    purpose: 'Generates a Blackman window for signal processing.',
    usage: 'Used in spectral analysis and filtering.',
    howToUse: 'Specify the window length to generate the Blackman window.'
  },
  {
    name: 'linear_regression',
    purpose: 'Performs linear regression analysis.',
    usage: 'Used for predicting continuous values based on input features.',
    howToUse: 'Train on labeled data to find the best fit line and make predictions.'
  },
  {
    name: 'remainder',
    purpose: 'Computes the remainder of division between two values.',
    usage: 'Used for modulus operations.',
    howToUse: 'Specify the dividend and divisor to get the remainder.'
  },
  {
    name: '1l_div',
    purpose: 'Performs division operation on input data.',
    usage: 'Basic arithmetic operation for data processing.',
    howToUse: 'Specify numerator and denominator to perform division.'
  },
  {
    name: '1l_slice',
    purpose: 'Slices input data based on specified indices.',
    usage: 'Used for extracting subsets of data.',
    howToUse: 'Specify start, stop, and step indices to slice the data.'
  },
  {
    name: 'boolean',
    purpose: 'Handles boolean operations on input data.',
    usage: 'Used for logical operations and condition evaluations.',
    howToUse: 'Input boolean data to perform logical operations.'
  },
  {
    name: 'linear_svc',
    purpose: 'Implements a linear Support Vector Classifier.',
    usage: 'Used for binary and multi-class classification tasks.',
    howToUse: 'Train on labeled data to classify new input data.'
  },
  {
    name: 'rnn',
    purpose: 'Implements a Recurrent Neural Network.',
    usage: 'Used for sequential data processing, such as time series and text.',
    howToUse: 'Input sequential data to get processed output from the RNN.'
  },
  {
    name: '1l_downsample',
    purpose: 'Reduces the spatial dimensions of input data.',
    usage: 'Used in neural networks for pooling operations.',
    howToUse: 'Specify the downsampling method and input data.'
  },
  {
    name: '1l_softmax',
    purpose: 'Applies the softmax activation function.',
    usage: 'Used in multi-class classification tasks to convert logits to probabilities.',
    howToUse: 'Pass input data through the model to apply softmax activation.'
  },
  {
    name: 'boolean_identity',
    purpose: 'Returns the input boolean data as-is.',
    usage: 'Used as a pass-through for boolean data.',
    howToUse: 'Input boolean data to get the same data as output.'
  },
  {
    name: 'log_softmax',
    purpose: 'Applies the log softmax activation function.',
    usage: 'Used in neural networks for numerical stability in log-probabilities.',
    howToUse: 'Pass input data through the model to apply log softmax activation.'
  },
  {
    name: 'rounding_ops',
    purpose: 'Performs rounding operations on input data.',
    usage: 'Used for rounding numerical data to specified precision.',
    howToUse: 'Specify the rounding method and input data.'
  },
  {
    name: '1l_eltwise_div',
    purpose: 'Performs element-wise division on input data.',
    usage: 'Used for per-element division operations in data processing.',
    howToUse: 'Specify two data arrays to perform element-wise division.'
  },
  {
    name: '1l_sqrt',
    purpose: 'Computes the square root of input data.',
    usage: 'Used for mathematical operations requiring square root calculations.',
    howToUse: 'Input data to get the square root of each element.'
  },
  {
    name: 'celu',
    purpose: 'Applies the CELU (Continuously Differentiable Exponential Linear Unit) activation function.',
    usage: 'Introduces non-linearity in neural networks with smoother transitions.',
    howToUse: 'Pass input data through the model to apply CELU activation.'
  },
  {
    name: 'logsumexp',
    purpose: 'Computes the log of the sum of exponentials of input data.',
    usage: 'Used for numerical stability in computations involving exponentials.',
    howToUse: 'Input data to get the logsumexp result.'
  },
  {
    name: 'scatter_elements',
    purpose: 'Scatters elements of input data based on specified indices.',
    usage: 'Used for data rearrangement and manipulation.',
    howToUse: 'Specify input data and indices for scattering elements.'
  },
  {
    name: '1l_elu',
    purpose: 'Applies the ELU (Exponential Linear Unit) activation function.',
    usage: 'Introduces non-linearity in neural networks with exponential output.',
    howToUse: 'Pass input data through the model to apply ELU activation.'
  },
  {
    name: '1l_tanh',
    purpose: 'Applies the tanh activation function.',
    usage: 'Used in neural networks to introduce non-linearity.',
    howToUse: 'Pass input data through the model to apply tanh activation.'
  },
  {
    name: 'clip',
    purpose: 'Clips input data to specified min and max values.',
    usage: 'Used to limit the range of data values.',
    howToUse: 'Specify the min and max values and input data to clip.'
  },
  {
    name: 'lstm',
    purpose: 'Implements a Long Short-Term Memory network.',
    usage: 'Used for sequential data processing, especially for time series and text.',
    howToUse: 'Input sequential data to get processed output from the LSTM.'
  },
  {
    name: 'scatter_nd',
    purpose: 'Scatters values into a new tensor based on specified indices.',
    usage: 'Used for data rearrangement and manipulation.',
    howToUse: 'Specify input data and indices for scattering into a new tensor.'
  },
  {
    name: 'self_attention',
    purpose: 'Implements self-attention mechanism in neural networks.',
    usage: 'Used in transformer models for capturing dependencies across input sequence.',
    howToUse: 'Input sequence data to get the self-attention output.'
  },
  {
    name: '1l_flatten',
    purpose: 'Flattens the input data to a single dimension.',
    usage: 'Used to convert multi-dimensional data into a flat array.',
    howToUse: 'Input multi-dimensional data to get a flattened output.'
  },
  {
    name: '1l_topk',
    purpose: 'Selects the top K elements from input data.',
    usage: 'Used for selecting top values in ranking and recommendation systems.',
    howToUse: 'Specify the value of K and input data to get top K elements.'
  },
  {
    name: 'doodles',
    purpose: 'Processes doodle drawings for recognition tasks.',
    usage: 'Used in image classification and pattern recognition.',
    howToUse: 'Input doodle data to get recognized output.'
  },
  {
    name: 'lstm_large',
    purpose: 'Implements a large Long Short-Term Memory network.',
    usage: 'Used for complex sequential data processing tasks.',
    howToUse: 'Input large sequential data to get processed output from the LSTM.'
  },
  {
    name: 'selu',
    purpose: 'Applies the SELU (Scaled Exponential Linear Unit) activation function.',
    usage: 'Used in neural networks to introduce self-normalizing properties.',
    howToUse: 'Pass input data through the model to apply SELU activation.'
  },
  {
    name: '1l_gelu_noappx',
    purpose: 'Applies the GELU (Gaussian Error Linear Unit) activation function without approximation.',
    usage: 'Introduces non-linearity in neural networks with Gaussian approximation.',
    howToUse: 'Pass input data through the model to apply GELU activation.'
  },
  {
    name: '1l_upsample',
    purpose: 'Upsamples input data to a higher resolution.',
    usage: 'Used in neural networks for upsampling operations.',
    howToUse: 'Specify the upsampling method and input data.'
  },
  {
    name: 'eye',
    purpose: 'Generates an identity matrix of specified size.',
    usage: 'Used for creating identity matrices in linear algebra.',
    howToUse: 'Specify the size to generate the identity matrix.'
  },
  {
    name: 'ltsf',
    purpose: 'Implements a Long-Term Short-Frequency model.',
    usage: 'Used for time series forecasting with long-term dependencies.',
    howToUse: 'Input time series data to get forecasted output.'
  },
  {
    name: 'sklearn_mlp',
    purpose: 'Implements a Multi-Layer Perceptron using scikit-learn.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: '1l_gelu_tanh_appx',
    purpose: 'Applies the GELU (Gaussian Error Linear Unit) activation function with tanh approximation.',
    usage: 'Introduces non-linearity in neural networks with tanh approximation.',
    howToUse: 'Pass input data through the model to apply GELU activation.'
  },
  {
    name: '1l_var',
    purpose: 'Computes the variance of input data.',
    usage: 'Used for statistical analysis of data variability.',
    howToUse: 'Input data to get the variance value.'
  },
  {
    name: 'gather_elements',
    purpose: 'Gathers elements from input data based on specified indices.',
    usage: 'Used for data rearrangement and selection.',
    howToUse: 'Specify input data and indices for gathering elements.'
  },
  {
    name: 'max',
    purpose: 'Computes the maximum value among input data.',
    usage: 'Used to find the maximum value in a dataset.',
    howToUse: 'Input data to get the maximum value.'
  },
  {
    name: 'softplus',
    purpose: 'Applies the softplus activation function.',
    usage: 'Introduces non-linearity in neural networks with smooth approximation.',
    howToUse: 'Pass input data through the model to apply softplus activation.'
  },
  {
    name: '1l_identity',
    purpose: 'Returns the input data as-is.',
    usage: 'Used as a pass-through for data.',
    howToUse: 'Input data to get the same data as output.'
  },
  {
    name: '1l_where',
    purpose: 'Selects elements from input data based on a condition.',
    usage: 'Used for conditional selection of data elements.',
    howToUse: 'Specify the condition and input data for selection.'
  },
  {
    name: 'gather_nd',
    purpose: 'Gathers values from a tensor based on specified indices.',
    usage: 'Used for advanced indexing and selection.',
    howToUse: 'Specify input tensor and indices for gathering values.'
  },
  {
    name: 'min',
    purpose: 'Computes the minimum value among input data.',
    usage: 'Used to find the minimum value in a dataset.',
    howToUse: 'Input data to get the minimum value.'
  },
  {
    name: 'softsign',
    purpose: 'Applies the softsign activation function.',
    usage: 'Introduces non-linearity in neural networks with a smoother curve.',
    howToUse: 'Pass input data through the model to apply softsign activation.'
  },
  {
    name: '1l_instance_norm',
    purpose: 'Applies instance normalization to input data.',
    usage: 'Used in neural networks to normalize individual data instances.',
    howToUse: 'Pass input data through the model to apply instance normalization.'
  },
  {
    name: '2l_relu_fc',
    purpose: 'Implements a two-layer fully connected network with ReLU activation.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'gradient_boosted_trees',
    purpose: 'Implements gradient boosted trees for machine learning.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: '1l_leakyrelu',
    purpose: 'Applies the Leaky ReLU activation function.',
    usage: 'Introduces non-linearity in neural networks with a small slope for negative values.',
    howToUse: 'Pass input data through the model to apply Leaky ReLU activation.'
  },
  {
    name: '2l_relu_sigmoid',
    purpose: 'Implements a two-layer network with ReLU and sigmoid activation functions.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'gru',
    purpose: 'Implements a Gated Recurrent Unit network.',
    usage: 'Used for sequential data processing, such as time series and text.',
    howToUse: 'Input sequential data to get processed output from the GRU.'
  },
  {
    name: 'mnist_gan',
    purpose: 'Generates images of handwritten digits using a GAN.',
    usage: 'Used for image generation tasks, particularly on MNIST dataset.',
    howToUse: 'Train on MNIST dataset to generate new digit images.'
  },
  {
    name: '1l_linear',
    purpose: 'Implements a single layer linear network.',
    usage: 'Used for linear transformations in neural networks.',
    howToUse: 'Input data to get linearly transformed output.'
  },
  {
    name: '2l_relu_sigmoid_conv',
    purpose: 'Implements a two-layer network with ReLU and sigmoid activation functions and convolutional layers.',
    usage: 'Used for image classification and other tasks requiring convolutional layers.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'hard_max',
    purpose: 'Applies the hardmax activation function.',
    usage: 'Used for classification tasks to convert logits to one-hot encoded labels.',
    howToUse: 'Pass input data through the model to apply hardmax activation.'
  },
  {
    name: 'mobilenet',
    purpose: 'Implements a MobileNet architecture for efficient image classification.',
    usage: 'Used for image classification tasks, especially on mobile and embedded devices.',
    howToUse: 'Train on labeled image data to classify new images.'
  },
  {
    name: 'trig',
    purpose: 'Performs trigonometric operations on input data.',
    usage: 'Used for mathematical operations involving trigonometric functions.',
    howToUse: 'Specify the trigonometric function and input data.'
  },
  {
    name: '1l_lppool',
    purpose: 'Applies Lp pooling to input data.',
    usage: 'Used in neural networks for pooling operations.',
    howToUse: 'Specify the pooling method and input data.'
  },
  {
    name: '2l_relu_sigmoid_small',
    purpose: 'Implements a two-layer network with ReLU and sigmoid activation functions for small datasets.',
    usage: 'Used for classification and regression tasks on small datasets.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'hard_sigmoid',
    purpose: 'Applies the hard sigmoid activation function.',
    usage: 'Introduces non-linearity in neural networks with a hard threshold.',
    howToUse: 'Pass input data through the model to apply hard sigmoid activation.'
  },
  {
    name: 'mobilenet_large',
    purpose: 'Implements a larger MobileNet architecture for image classification.',
    usage: 'Used for image classification tasks, especially on high-performance devices.',
    howToUse: 'Train on labeled image data to classify new images.'
  },
  {
    name: '1l_max_pool',
    purpose: 'Applies max pooling to input data.',
    usage: 'Used in neural networks for downsampling operations.',
    howToUse: 'Specify the pooling size and input data.'
  },
  {
    name: '2l_relu_small',
    purpose: 'Implements a two-layer network with ReLU activation for small datasets.',
    usage: 'Used for classification and regression tasks on small datasets.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'hard_swish',
    purpose: 'Applies the hard swish activation function.',
    usage: 'Introduces non-linearity in neural networks with a swish-like behavior.',
    howToUse: 'Pass input data through the model to apply hard swish activation.'
  },
  {
    name: 'multihead_attention',
    purpose: 'Implements multi-head attention mechanism.',
    usage: 'Used in transformer models for capturing dependencies across input sequences.',
    howToUse: 'Input sequence data to get multi-head attention output.'
  },
  {
    name: '1l_mean',
    purpose: 'Computes the mean value of input data.',
    usage: 'Used for obtaining average values in data processing.',
    howToUse: 'Input data into the model to get the mean value as output.'
  },
  {
    name: '2l_sigmoid_small',
    purpose: 'Implements a two-layer network with sigmoid activation for small datasets.',
    usage: 'Used for classification and regression tasks on small datasets.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'hummingbird_decision_tree',
    purpose: 'Implements a decision tree using Hummingbird.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'nanoGPT',
    purpose: 'Implements a small-scale GPT model for text generation.',
    usage: 'Used for generating text based on input prompts.',
    howToUse: 'Input text prompt to generate continuation or new text.'
  },
  {
    name: 'xgboost',
    purpose: 'Implements the XGBoost algorithm for machine learning.',
    usage: 'Used for classification, regression, and ranking tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: '1l_mlp',
    purpose: 'Implements a single layer multi-layer perceptron.',
    usage: 'Used for classification and regression tasks.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: '3l_relu_conv_fc',
    purpose: 'Implements a three-layer network with ReLU, convolution, and fully connected layers.',
    usage: 'Used for image classification and other tasks requiring convolutional layers.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'idolmodel',
    purpose: 'A specialized model for recognizing idol images.',
    usage: 'Used for image recognition tasks in the idol industry.',
    howToUse: 'Input idol image data to get recognition results.'
  },
  {
    name: 'oh_decision_tree',
    purpose: 'Implements a decision tree with one-hot encoding.',
    usage: 'Used for classification tasks with categorical data.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'xgboost_reg',
    purpose: 'Implements XGBoost for regression tasks.',
    usage: 'Used for predicting continuous values based on input features.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: '1l_pad',
    purpose: 'Pads the input data with specified values.',
    usage: 'Used for adding padding to data arrays.',
    howToUse: 'Specify padding values and input data.'
  },
  {
    name: '4l_relu_conv_fc',
    purpose: 'Implements a four-layer network with ReLU, convolution, and fully connected layers.',
    usage: 'Used for image classification and other tasks requiring convolutional layers.',
    howToUse: 'Train on labeled data and use the model for predictions.'
  },
  {
    name: 'large_op_graph',
    purpose: 'Implements a large operation graph for complex computations.',
    usage: 'Used for handling complex computational tasks in neural networks.',
    howToUse: 'Input data to get processed output based on the operation graph.'
  },
  {
    name: 'prelu_gmm',
    purpose: 'Applies Parametric ReLU activation in a Gaussian Mixture Model.',
    usage: 'Used for clustering and density estimation tasks.',
    howToUse: 'Train on input data to perform clustering or density estimation.'
  }
];
