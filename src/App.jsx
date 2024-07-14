import { useState } from 'react';
import './App.css';
import 'tailwindcss/tailwind.css';
import classNames from "classnames";
import { FaCogs, FaKeyboard, FaCheckCircle } from 'react-icons/fa';

const models = [
  '1l_average', '1l_powf', 'accuracy', 'layernorm', 'quantize_dequantize',
  '1l_batch_norm', '1l_prelu', 'arange', 'lenet_5', 'random_forest',
  '1l_concat', '1l_relu', 'bitshift', 'less', 'reducel1',
  '1l_conv', '1l_reshape', 'bitwise_ops', 'lightgbm', 'reducel2',
  '1l_conv_transpose', '1l_sigmoid', 'blackman_window', 'linear_regression', 'remainder',
  '1l_div', '1l_slice', 'boolean', 'linear_svc', 'rnn',
  '1l_downsample', '1l_softmax', 'boolean_identity', 'log_softmax', 'rounding_ops',
  '1l_eltwise_div', '1l_sqrt', 'celu', 'logsumexp', 'scatter_elements',
  '1l_elu', '1l_tanh', 'clip', 'lstm', 'scatter_nd',
  '1l_erf', '1l_tiny_div', 'decision_tree', 'lstm_large', 'self_attention',
  '1l_flatten', '1l_topk', 'doodles', 'lstm_medium', 'selu',
  '1l_gelu_noappx', '1l_upsample', 'eye', 'ltsf', 'sklearn_mlp',
  '1l_gelu_tanh_appx', '1l_var', 'gather_elements', 'max', 'softplus',
  '1l_identity', '1l_where', 'gather_nd', 'min', 'softsign',
  '1l_instance_norm', '2l_relu_fc', 'gradient_boosted_trees', 'mish', 'trig',
  '1l_leakyrelu', '2l_relu_sigmoid', 'gru', 'mnist_gan', 'tril',
  '1l_linear', '2l_relu_sigmoid_conv', 'hard_max', 'mobilenet', 'triu',
  '1l_lppool', '2l_relu_sigmoid_small', 'hard_sigmoid', 'mobilenet_large', 'tutorial',
  '1l_max_pool', '2l_relu_small', 'hard_swish', 'multihead_attention', 'variable_cnn',
  '1l_mean', '2l_sigmoid_small', 'hummingbird_decision_tree', 'nanoGPT', 'xgboost',
  '1l_mlp', '3l_relu_conv_fc', 'idolmodel', 'oh_decision_tree', 'xgboost_reg',
  '1l_pad', '4l_relu_conv_fc', 'large_op_graph', 'prelu_gmm'
];

function App() {
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [inputData, setInputData] = useState('');
  const [loading, setLoading] = useState(false);
  const [verify, setVerify] = useState(false);
  const [proof, setProof] = useState('');
  const [verifyDisabled, setVerifyDisabled] = useState(true);
  const [verifyLoading, setVerifyLoading] = useState(false);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleInputChange = (event) => {
    setInputData(event.target.value);
  };

  const handleSubmit = () => {
    setLoading(true);
    setVerify(false);
    setProof('');
    setVerifyDisabled(true);
    setTimeout(() => {
      setLoading(false);
      setVerifyDisabled(false);
      console.log(`Model: ${selectedModel}, Input: ${inputData}`);
    }, 5000);
  };

  const handleVerify = () => {
    setVerifyLoading(true);
    const randomDelay = Math.floor(Math.random() * 4000) + 1000;
    setTimeout(() => {
      setVerifyLoading(false);
      setVerify(true);
      setProof('Proof verified successfully!');
    }, randomDelay);
  };

  return (
      <div
          className="App min-h-screen bg-cover bg-center flex flex-col items-center justify-center p-4"
          style={ { backgroundImage: "url('https://source.unsplash.com/1600x900/?ai,technology')" } }
      >
        <h1 className="text-4xl font-bold text-white mb-6">Select Model and Input Data</h1>
        <div className="mb-4 w-full max-w-md">
          <label
              htmlFor="model-selector"
              className="block text-lg font-medium text-white mb-2"
          >Select Model:</label>
          <select
              id="model-selector"
              value={ selectedModel }
              onChange={ handleModelChange }
              className="block w-full p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          >
            { models.map((model) => (
                <option
                    key={ model }
                    value={ model }
                >
                  { model }
                </option>
            )) }
          </select>
        </div>
        <div className="mb-4 w-full max-w-md">
          <label
              htmlFor="model-input"
              className="block text-lg font-medium text-white mb-2"
          >Input Data:</label>
          <input
              id="model-input"
              type="text"
              value={ inputData }
              onChange={ handleInputChange }
              className="block w-full p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        <button
            onClick={ handleSubmit }
            className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-md shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            disabled={ loading }
        >
          { loading ? (
              <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
              >
                <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                ></circle>
                <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                ></path>
              </svg>
          ) : (
              'Submit'
          ) }
        </button>
        <button
            onClick={ handleVerify }
            className={ classNames(
                "mt-4 px-6 py-3 font-semibold rounded-md shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2",
                {
                  "bg-green-600 text-white hover:bg-green-700 focus:ring-green-500": !verifyDisabled,
                  "bg-gray-400 text-gray-700 cursor-not-allowed": verifyDisabled,
                }
            ) }
            disabled={ verifyDisabled || verifyLoading }
        >
          { verifyLoading ? (
              <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
              >
                <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                ></circle>
                <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                ></path>
              </svg>
          ) : (
              'Verify Proof'
          ) }
        </button>
        <div className="mt-6 text-white flex flex-col items-center space-y-4">
          <div className="flex items-center space-x-2">
            <FaCogs className="text-2xl" />
            <h2 className="text-2xl font-semibold">Selected Model: { selectedModel }</h2>
          </div>
          <div className="flex items-center space-x-2">
            <FaKeyboard className="text-2xl" />
            <h2 className="text-2xl font-semibold">Input Data: { inputData }</h2>
          </div>
          { verify && (
              <div className="flex items-center space-x-2">
                <FaCheckCircle className="text-2xl" />
                <h2 className="text-2xl font-semibold">Proof: { proof }</h2>
              </div>
          ) }
        </div>
      </div>
  );
}

export default App;

