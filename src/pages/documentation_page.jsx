import React from "react";
import { Flex, VStack, Text, Divider } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import MarkdownRenderer from "../components/MarkdownRenderer";




const classic_dataset = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryClassicDataset( </p>
<p style="margin-left: 60px"> store_id: str, - <span style="color: green;"> Store ID (STORE_1) </span> </p>
<p style="margin-left: 60px"> shop_sales: pd.DataFrame, - <span style="color: green;"> Data on the number of products sold per store </span> </p>
<p style="margin-left: 60px"> shop_sales_dates: pd.DataFrame, - <span style="color: green;"> Data on holidays </span> </p>
<p style="margin-left: 60px"> shop_sales_prices: pd.DataFrame - <span style="color: green;"> Product price data </span> </p>
<p style="margin-left: 60px"> plots: bool - <span style="color: green;"> Plotting a graph </span> </p>
<p style="margin-left: 60px"> save_plots: bool - <span style="color: green;"> Saving a graph </span> </p>
<p style="margin-left: 60px"> save_path_plots: str - <span style="color: green;"> Path to graph directory </span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;"> Input model data </span> </p>
<p style="margin-left: 20px"> ) -> dict[str, pd.DataFrame]: - <span style="color: green;"> Dictionary of 15 datasets, 1 for each time series </span> </p> </br>

<p style="margin-left: 20px"> <span style="color: green;"> Loads and prepares dataframes for each product </span> </p> </br>

1.<span style="color: blue;"> dataset</span> method: Initialization method. </br>
2.<span style="color: blue;"> fetch_data</span> method: Loads csv files and creates tables using duckdb. </br>
3.<span style="color: blue;"> merge_data</span> method: Initializes SQL query to merge three tables shop_sales, shop_sales_dates, shop_sales_prices into one. </br>
4.<span style="color: blue;"> split_merge</span> method: Splits the general table into 15 tables, 1 for each time series. (item) </br>
5.<span style="color: blue;"> merge</span> method: Returns the merged dataset with all items. </br>
6.<span style="color: blue;"> dictidx</span> method: Returns a dictionary with param2idx and idx2param. </br>
7.<span style="color: blue;"> dictmerge</span> method: Returns a dictionary of datasets, one for each time series. </br>
8.<span style="color: blue;"> visualise</span> method: Plots graphs for each time series with the number of sales and the change in price over time on a normal scale. </br>


`;



const classic_proccess = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryClassicProccess( </p>
<p style="margin-left: 60px"> dictmerge: dict[str, pd.DataFrame], - <span style="color: green;"> Dictionary of merged datasets for each time series </span> </p>
<p style="margin-left: 60px"> dictdecompose: dict[str, int], - <span style="color: green;"> Parameters for decomposition (seasonal periods) </span> </p> 
<p style="margin-left: 60px"> removebound: dict[str, float], - <span style="color: green;"> Lower and upper boundary factors for removing outliers </span> </p>
<p style="margin-left: 60px"> plots: bool, - <span style="color: green;"> Create time series plots </span> </p>
<p style="margin-left: 60px"> saveplots: bool, - <span style="color: green;"> Save time series plots </span> </p>
<p style="margin-left: 60px"> savepathplots: str - <span style="color: green;"> Path to save plots </span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;"> Input parameters for the process </span> </p>
<p style="margin-left: 20px"> ) -> dict[str, dict[str, pd.DataFrame]]: - <span style="color: green;"> Dictionary with decomposition results for time series </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Performs preprocessing of time series, outlier removal, decomposition, and visualization </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">proccess</span> method: The main processing method. Performs decomposition and visualization for each time series. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">remove_outliers</span> method: Removes outliers from the time series based on the interquartile range. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">decompose</span> method: Decomposes the time series into trend, seasonality, and residuals using three methods (additive, multiplicative). </p>
<p style="margin-left: 20px"> 4. <span style="color: blue;">visualise</span> method: Plots and saves graphs for the original and processed data, as well as their autocorrelations. </p>

`;



const classic_models = `

<p style="margin-left: 20px"> <span style="color: green;">class</span> <span style="color: blue;">ClassicModel</span>(<span style="color: green;">ABC</span>): </p>

<p style="margin-left: 40px"> name_model: str = None - <span style="color: green;"> Name of the model, unique for each subclass </span> </p>
<p style="margin-left: 40px"> _registered_models: dict = {} - <span style="color: green;"> Registry of available models </span> </p>

<p style="margin-left: 40px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 60px"> train: pd.DataFrame, - <span style="color: green;"> Data for training the model </span> </p>
<p style="margin-left: 60px"> test: pd.DataFrame, - <span style="color: green;"> Data for testing the model </span> </p>
<p style="margin-left: 60px"> exogenous - <span style="color: green;"> Exogenous data for the model </span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;"> Base class constructor </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">fit_pred</span> method: Trains the model and returns predictions and an accuracy metric. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">score</span> method: Returns the model's evaluation score on the test dataset. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">param</span> method: Returns the model parameters. </p>
<p style="margin-left: 20px"> 4. <span style="color: blue;">pred</span> method: Makes predictions for the test dataset. </p>
<p style="margin-left: 20px"> 5. <span style="color: blue;">save</span> method: Saves the trained model and results to disk. </p>
<p style="margin-left: 20px"> 6. <span style="color: blue;">register_model</span> method: Registers the model in the _registered_models dictionary. </p>
<p style="margin-left: 20px"> 7. <span style="color: blue;">get_model_classes</span> method: Returns registered model classes. </p>
<p style="margin-left: 20px"> 8. <span style="color: blue;">create_model</span> method: Creates a model object by its name. </p>
<p style="margin-left: 20px"> 9. <span style="color: blue;">from_pretrained</span> method: Loads a pretrained model from disk. </p>

<p style="margin-left: 20px"> Registered models: </p>
<p style="margin-left: 40px"> 1. AUTOARIMA </p>
<p style="margin-left: 40px"> 2. AUTOREG </p>
<p style="margin-left: 40px"> 3. AUTOETS </p>
<p style="margin-left: 40px"> 4. PROPHET </p>
<p style="margin-left: 40px"> 5. TBATS </p>

`;



const classic_graduate = `

<span style="color: blue;">ClassicGraduate</span>:</br>

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryClassicGraduate( </p>
<p style="margin-left: 60px"> dictidx: dict[str, pd.Series], - <span style="color: green;"> Dictionary from the ClassicDataset.dictidx method </span> </p>
<p style="margin-left: 60px"> dictmerge: dict[str, pd.DataFrame], - <span style="color: green;"> Dictionary from the ClassicDataset.dictmerge method </span> </p>
<p style="margin-left: 60px"> dictseasonal: dict[str, int], - <span style="color: green;"> Ranges of predictions for each period </span> </p>
<p style="margin-left: 60px"> modelsparams: dict[str, tuple], - <span style="color: green;"> Parameters for all models </span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;"> Model input data </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Initializes key variables and execution results </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">graduate</span> method: The main method for calculation and selection of the best models. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">train_model</span> method: Selects optimal parameters, a model, and a time window for each time series. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">calc_optimum</span> method: Calculates the model with the given parameters, evaluates predictions, and returns metrics. </p>

`;




const classic_inference = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryClassicInference( </p>
<p style="margin-left: 60px"> dictidx: dict[str, pd.Series], - <span style="color: green;"> Dictionary from the ClassicDataset.dictidx method </span> </p>
<p style="margin-left: 60px"> dictmerge: dict[str, pd.DataFrame], - <span style="color: green;"> Dictionary from the ClassicDataset.dictmerge method </span> </p>
<p style="margin-left: 60px"> dictseasonal: dict[str, int], - <span style="color: green;"> Prediction ranges for each period (week, month, quarter) </span> </p>
<p style="margin-left: 60px"> plots: bool, - <span style="color: green;"> Flag to display result graphs </span> </p>
<p style="margin-left: 60px"> save_plots: bool, - <span style="color: green;"> Flag to save graphs to a directory </span> </p>
<p style="margin-left: 60px"> save_path_plots: str, - <span style="color: green;"> Path to save graphs </span> </p>
<p style="margin-left: 60px"> save_path_weights: str, - <span style="color: green;"> Path to save and load model weights </span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;"> Model input data </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Initializes key variables and execution results </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">inference</span> method: Performs time series predictions, evaluates results, and visualizes data. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">load_models</span> method: Loads pretrained models and their parameters from the file system. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">visualise</span> method: Plots actual and predicted data for each time series. </p>
<p style="margin-left: 20px"> 4. <span style="color: blue;">evaluate</span> method: Assesses the quality of models based on predictions and saves metrics. </p>
<p style="margin-left: 20px"> 5. <span style="color: blue;">calc_feature</span> method: Performs predictions, checks data conformity, and returns model metrics. </p>

`;



const neiro_dataset = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> dictidx: dict, - <span style="color: green;"> Dictionary with param2idx and idx2param </span> </p>
<p style="margin-left: 40px"> metadata: list[tuple(pd.DataFrame, pd.DataFrame)], - <span style="color: green;"> List of DataFrame slices for training and testing </span> </p>
<p style="margin-left: 40px"> ) -> torch[Dataset]: - <span style="color: green;"> Torch Dataset </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Loads and prepares the data </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">proccess_data</span> method: General method for processing train and test data. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">__len__</span> method: Returns the number of slices in the main dataset. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">__getitem__</span> method: Iterates over the data. </p>

<span style="color: blue;">get_datasets</span>:</br>

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> dictidx: dict, - <span style="color: green;"> Dictionary with param2idx and idx2param </span> </p>
<p style="margin-left: 40px"> dictmerge: dict, - <span style="color: green;"> List of DataFrame slices for training and testing </span> </p>
<p style="margin-left: 40px"> item_id: str, - <span style="color: green;"> Identifier of the item from the dataset </span> </p>
<p style="margin-left: 40px"> test_size: dict, - <span style="color: green;"> Portion of test data unseen by the model </span> </p>
<p style="margin-left: 40px"> period: dict, - <span style="color: green;"> Prediction range </span> </p>
<p style="margin-left: 40px"> seq_len: dict, - <span style="color: green;"> Sequence length (lookback) </span> </p>
<p style="margin-left: 40px"> step_length: dict, - <span style="color: green;"> Step size for slicing </span> </p>
<p style="margin-left: 40px"> seed: dict, - <span style="color: green;"> Seed value </span> </p>
<p style="margin-left: 40px"> future_or_estimate_or_train: dict, - <span style="color: green;"> Rule for splitting the dataset </span> </p>
<p style="margin-left: 40px"> ) -> torch[Dataset]: - <span style="color: green;"> Torch Dataset </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Splits data for a specific item based on the prediction period </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">Function</span>: Function with three conditions: estimate/future/train. </p>

`;



const neiro_graduate = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryNeiroGraduate( </p>
<p style="margin-left: 60px"> dictidx: dict, - <span style="color: green;"> Dictionary obtained via ClassicDataset.dictidx </span> </p>
<p style="margin-left: 60px"> dictmerge: dict[str, pd.DataFrame], - <span style="color: green;"> Dictionary obtained via ClassicDataset.dictmerge </span> </p>
<p style="margin-left: 60px"> dictseasonal: dict[str, int], - <span style="color: green;"> Dictionary of prediction ranges (e.g., week, month, quarter) </span> </p>
<p style="margin-left: 60px"> dictmodels: dict[str, dict], - <span style="color: green;"> Dictionary with model parameters </span> </p>
<p style="margin-left: 60px"> seq_len: int, - <span style="color: green;"> Sequence length (lookback) </span> </p>
<p style="margin-left: 60px"> test_size: float, - <span style="color: green;"> Proportion of the test set unseen by the model </span> </p>
<p style="margin-left: 60px"> step_length: int, - <span style="color: green;"> Steps between data slices </span> </p>
<p style="margin-left: 60px"> path_to_weights: str, - <span style="color: green;"> Path for saving model weights </span> </p>
<p style="margin-left: 60px"> use_device: str, - <span style="color: green;"> Device to use (cpu/cuda) </span> </p>
<p style="margin-left: 60px"> start_learning_rate: float, - <span style="color: green;"> Initial learning rate </span> </p>
<p style="margin-left: 60px"> batch_size: int, - <span style="color: green;"> Batch size during training </span> </p>
<p style="margin-left: 60px"> num_workers: int, - <span style="color: green;"> Number of workers for data loading </span> </p>
<p style="margin-left: 60px"> pin_memory: bool, - <span style="color: green;"> If True, accelerates GPU data loading (False for CPU) </span> </p>
<p style="margin-left: 60px"> num_epochs: int, - <span style="color: green;"> Number of training epochs </span> </p>
<p style="margin-left: 60px"> name_optimizer: str, - <span style="color: green;"> Optimizer name </span> </p>
<p style="margin-left: 60px"> seed: int, - <span style="color: green;"> Seed for reproducibility </span> </p>
<p style="margin-left: 60px"> ) - <span style="color: green;"> Data input model </span> </p>
<p style="margin-left: 40px"> ) -> None: - <span style="color: green;"> Constructor for configuring NeiroGraduate parameters </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> NeiroGraduate trains, tests models, selects the best one, and saves weights. </span></p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">graduate</span> method: Initialization method. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">__str__</span> method: Outputs certain parameters. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">get_loaders</span> method: Initializes torch_loader. </p>
<p style="margin-left: 20px"> 4. <span style="color: blue;">get_models</span> method: Initializes models. </p>
<p style="margin-left: 20px"> 5. <span style="color: blue;">get_opt_crit_sh</span> method: Initializes optimizer and loss function. </p>
<p style="margin-left: 20px"> 6. <span style="color: blue;">load_checkpoint</span> method: Loads model checkpoints. </p>
<p style="margin-left: 20px"> 7. <span style="color: blue;">train_models</span> method: Trains models. </p>
<p style="margin-left: 20px"> 8. <span style="color: blue;">evaluate_models</span> method: Tests models. </p>

<span style="color: blue;">collate_fn</span>:</br>

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> batch: dict[str, dict], - <span style="color: green;"> Multi-level batch containing training data </span> </p>
<p style="margin-left: 40px"> minmax_resid: MinMaxScaler, - <span style="color: green;"> MinMaxScaler for residuals </span> </p>
<p style="margin-left: 40px"> minmax_trend: MinMaxScaler, - <span style="color: green;"> MinMaxScaler for trends </span> </p>
<p style="margin-left: 40px"> minmax_season: MinMaxScaler, - <span style="color: green;"> MinMaxScaler for seasonality </span> </p>
<p style="margin-left: 40px"> minmax_sellprice: MinMaxScaler, - <span style="color: green;"> MinMaxScaler for item prices </span> </p>
<p style="margin-left: 40px"> minmax_series: MinMaxScaler, - <span style="color: green;"> MinMaxScaler for time series </span> </p>
<p style="margin-left: 40px"> pdata: bool, - <span style="color: green;"> If True, returns pd.Series instead of torch tensors </span> </p>
<p style="margin-left: 40px"> without_test: bool, - <span style="color: green;"> If True, excludes the test set </span> </p>
<p style="margin-left: 40px"> ) -> dict[str, dict]: - <span style="color: green;"> Multi-level batch after processing </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> Collate_fn preprocesses each batch during iteration in torch_loader </span> </p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">Function</span>: Function with conditions. </p>

<span style="color: blue;">CustomLoss</span>:</br>

<p style="margin-left: 20px"> <span style="color: green;">class</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> beta: float = 1.0, - <span style="color: green;"> Parameter for SmoothL1Loss </span> </p>
<p style="margin-left: 40px"> delta: float = 0.5, - <span style="color: green;"> Weight for penalty on negative values </span> </p>
<p style="margin-left: 40px"> gamma: float = 0.1, - <span style="color: green;"> Weight for CosineEmbeddingLoss </span> </p>
<p style="margin-left: 40px"> cosine_margin: float = 0.0, - <span style="color: green;"> Margin for CosineEmbeddingLoss </span> </p>
<p style="margin-left: 40px"> special_penalty: float = 1.0, - <span style="color: green;"> Penalty for second and subsequent components </span> </p>
<p style="margin-left: 40px"> ) -> None: - <span style="color: green;"> Constructor for configuring custom loss function </span> </p>

<p style="margin-left: 20px"> <span style="color: green;"> CustomLoss combines SmoothL1Loss, penalties for negative values, </br>
CosineEmbeddingLoss, and special penalties for trend to calculate final loss function. </span></p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">forward</span> method: </p>
<p style="margin-left: 40px"> <span style="color: green;">def</span> <span style="color: blue;">forward</span>( </p>
<p style="margin-left: 60px"> pred: torch.Tensor, - <span style="color: green;"> Model predictions (batch_size, seq_len, num_components) </span> </p>
<p style="margin-left: 60px"> target: torch.Tensor, - <span style="color: green;"> Target values (batch_size, seq_len, num_components) </span> </p>
<p style="margin-left: 60px"> aux_input: Optional[torch.Tensor] = None, - <span style="color: green;"> Auxiliary input for CosineEmbeddingLoss </span> </p>
<p style="margin-left: 60px"> ) -> torch.Tensor: - <span style="color: green;"> Calculates final loss </span> </p>

<p style="margin-left: 20px"> 2. Final loss function: </p>
<p style="margin-left: 40px"> <span style="color: green;"> Summed losses for each component, weighted and adjusted by additional penalties </span> </p>

`;



const neiro_inference = `

<p style="margin-left: 20px"> <span style="color: green;">def</span> <span style="color: blue;">__init__</span>( </p>
<p style="margin-left: 40px"> entry: EntryNeiroInference( </p>
<p style="margin-left: 60px"> dictidx: dict, - <span style="color: green;">Dictionary from the ClassicDataset.dictidx method</span> </p>
<p style="margin-left: 60px"> dictmerge: dict[str, pd.DataFrame], - <span style=color: green;">*Dictionary from the ClassicDataset.dictmerge method</span> </p>
<p style="margin-left: 60px"> dictseasonal: dict[str, int], - <span style="color: green;">Prediction ranges for each period (week, month, quarter, year)</span> </p>
<p style="margin-left: 60px"> dictmodels: dict[str, dict], - <span style="color: green;">Dictionary with model parameters</span> </p>
<p style="margin-left: 60px"> future_or_estimate: str, - <span style="color: green;">Inference mode: 'estimate' (evaluates the last chunk of data) or 'future' (makes a future prediction)</span> </p>
<p style="margin-left: 60px"> seq_len: int, - <span style="color: green;">Sequence length (lookback)</span> </p>
<p style="margin-left: 60px"> path_to_weights: str, - <span style="color: green;">Path to save/load weights, default './weights_neiro'</span> </p>
<p style="margin-left: 60px"> plots: bool, - <span style="color: green;">Flag to plot time series graphs? Yes/No</span> </p>
<p style="margin-left: 60px"> save_plots: bool, - <span style="color: green;">Flag to save time series graphs? Yes/No</span> </p>
<p style="margin-left: 60px"> save_path_plots: str, - <span style="color: green;">If saving graphs, path to save (default './plots')</span> </p>
<p style="margin-left: 60px"> use_device: str, - <span style="color: green;">Device for computation: cpu/cuda*</span> </p>
<p style="margin-left: 60px"> num_workers: int, - <span style="color: green;">Number of threads for loading data (0 means 1 thread)</span> </p>
<p style="margin-left: 60px"> pin_memory: bool, - <span style="color: green;">If True, accelerates data loading on GPU, always False for CPU</span> </p>
<p style="margin-left: 40px"> ) - <span style="color: green;">Model input data</span> </p>

<p style="margin-left: 20px"> <span style="color: green;">NeiroInference is used to prepare data and perform model inference on time series with the option to customize graphs and save results.</span></p>

<p style="margin-left: 20px"> 1. <span style="color: blue;">inference</span> method: Performs time series predictions, evaluates results, and visualizes data. </p>
<p style="margin-left: 20px"> 2. <span style="color: blue;">load_models</span> method: Loads pretrained models and their parameters from the file system. </p>
<p style="margin-left: 20px"> 3. <span style="color: blue;">visualise</span> method: Plots actual and predicted data for each time series. </p>
<p style="margin-left: 20px"> 4. <span style="color: blue;">evaluate</span> method: Assesses the quality of models based on predictions and saves metrics. </p>
<p style="margin-left: 20px"> 5. <span style="color: blue;">calc_feature</span> method: Performs predictions, checks data conformity, and returns model metrics. </p>

`;


const season_analytic_query = `

<p style="margin-left: 20px"> { </p>
<p style="margin-left: 40px"> <span style="color: green;">"dataset"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"store_id"</span>: <span style="color: red;">"STORE_1"</span> </p>
<p style="margin-left: 40px"> }, </p>
<p style="margin-left: 40px"> <span style="color: green;">"process"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictdecompose"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"week"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"month"</span>: <span style="color: red;">30</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"quarter"</span>: <span style="color: red;">90</span> </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"remove_bound"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"lower_bound_factor"</span>: <span style="color: red;">5</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"upper_bound_factor"</span>: <span style="color: red;">5</span> </p>
<p style="margin-left: 40px"> } </p>
<p style="margin-left: 20px"> } </p>

`;




const classic_graduate_query = `

<p style="margin-left: 20px"> { </p>
<p style="margin-left: 40px"> <span style="color: green;">"dataset"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"store_id"</span>: <span style="color: red;">"STORE_1"</span> </p>
<p style="margin-left: 40px"> }, </p>
<p style="margin-left: 40px"> <span style="color: green;">"graduate"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictseasonal"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"week"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"month"</span>: <span style="color: red;">30</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"quater"</span>: <span style="color: red;">90</span> </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"models_params"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"AUTOREG"</span>: [ </p>
<p style="margin-left: 100px"> <span style="color: red;">7</span>, </p>
<p style="margin-left: 100px"> <span style="color: red;">"week"</span> </p>
<p style="margin-left: 80px"> ] </p>
<p style="margin-left: 40px"> } </p>
<p style="margin-left: 20px"> } </p>

`;



const classic_inference_query = `

<p style="margin-left: 20px"> { </p>
<p style="margin-left: 40px"> <span style="color: green;">"dataset"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"store_id"</span>: <span style="color: red;">"STORE_1"</span> </p>
<p style="margin-left: 40px"> }, </p>
<p style="margin-left: 40px"> <span style="color: green;">"inference"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictseasonal"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"week"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"month"</span>: <span style="color: red;">30</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"quater"</span>: <span style="color: red;">90</span> </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"future_or_estimate"</span>: <span style="color: red;">"estimate"</span> </p>
<p style="margin-left: 40px"> } </p>
<p style="margin-left: 20px"> } </p>

`;




const neiro_graduate_query = `

<p style="margin-left: 20px"> { </p>
<p style="margin-left: 40px"> <span style="color: green;">"dataset"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"store_id"</span>: <span style="color: red;">"STORE_1"</span> </p>
<p style="margin-left: 40px"> }, </p>
<p style="margin-left: 40px"> <span style="color: green;">"graduate"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictseasonal"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"month"</span>: <span style="color: red;">30</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"quater"</span>: <span style="color: red;">90</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"week"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"year"</span>: <span style="color: red;">365</span> </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictmodels"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"IF"</span>: { </p>
<p style="margin-left: 100px"> <span style="color: blue;">"depth"</span>: <span style="color: red;">6</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"dim"</span>: <span style="color: red;">512</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"dim_head"</span>: <span style="color: red;">64</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"heads"</span>: <span style="color: red;">8</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"num_tokens_per_variate"</span>: <span style="color: red;">1</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"num_variates"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"use_reversible_instance_norm"</span>: <span style="color: red;">true</span> </p>
<p style="margin-left: 80px"> } </p>
<p style="margin-left: 60px"> } </p>
<p style="margin-left: 40px"> } </p>
<p style="margin-left: 20px"> } </p>

`;



const neiro_inference_query = `

<p style="margin-left: 20px"> { </p>
<p style="margin-left: 40px"> <span style="color: green;">"dataset"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"store_id"</span>: <span style="color: red;">"STORE_1"</span> </p>
<p style="margin-left: 40px"> }, </p>
<p style="margin-left: 40px"> <span style="color: green;">"inference"</span>: { </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictseasonal"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"week"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"month"</span>: <span style="color: red;">30</span>, </p>
<p style="margin-left: 80px"> <span style="color: blue;">"quater"</span>: <span style="color: red;">90</span> </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"dictmodels"</span>: { </p>
<p style="margin-left: 80px"> <span style="color: blue;">"IFFT"</span>: { </p>
<p style="margin-left: 100px"> <span style="color: blue;">"depth"</span>: <span style="color: red;">6</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"dim"</span>: <span style="color: red;">256</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"dim_head"</span>: <span style="color: red;">64</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"heads"</span>: <span style="color: red;">8</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"num_tokens_per_variate"</span>: <span style="color: red;">1</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"num_variates"</span>: <span style="color: red;">7</span>, </p>
<p style="margin-left: 100px"> <span style="color: blue;">"use_reversible_instance_norm"</span>: <span style="color: red;">true</span> </p>
<p style="margin-left: 80px"> } </p>
<p style="margin-left: 60px"> }, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"future_or_estimate"</span>: <span style="color: red;">"estimate"</span>, </p>
<p style="margin-left: 60px"> <span style="color: blue;">"use_device"</span>: <span style="color: red;">"cuda"</span> </p>
<p style="margin-left: 40px"> } </p>
<p style="margin-left: 20px"> } </p>

`;






const DocumentPage = () => {
    const { width } = useWindowDimensions();

    return (
        <Flex
            direction="column"
            bg="transparent"
            padding={25}
            spacing="20px"
            flexGrow={1}
            align="center"
            justify="flex-start" // Изменяем на "flex-start", чтобы контент не центрировался
            width={width}
            height="100%"
            overflowX="hidden" // Отключение горизонтального скролла
            overflowY="auto" // Включаем вертикальную прокрутку
            paddingTop="150px" // Устанавливаем отступ сверху равный высоте header
        >
            <VStack
                spacing={4}
                align="stretch"
                width="100%"
                maxW="1200px"
                height="100%"
            >


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    SeasonAnalyticPipelineExampleQuery
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={season_analytic_query} />
                </Flex>
                <Divider />



                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicGraduatePipelineExampleQuery
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_graduate_query} />
                </Flex>
                <Divider />



                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicInferencePipelineExampleQuery
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_inference_query} />
                </Flex>
                <Divider />



                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    NeiroGraduatePipelineExampleQuery
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={neiro_graduate_query} />
                </Flex>
                <Divider />



                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    NeiroInferencePipelineExampleQuery
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={neiro_inference_query} />
                </Flex>
                <Divider />
                
                

                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicDataset
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_dataset} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicProccess
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_proccess} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicModels
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_models} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicGraduate
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_graduate} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    ClassicInference
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={classic_inference} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    NeiroDataset + get_datasets
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={neiro_dataset} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="408px"
                >
                    NeiroGraduate + collate_fn + CustomLoss
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={neiro_graduate} />
                </Flex>
                <Divider />


                <Text
                    color="#FFFFFF"
                    fontFamily="Inter"
                    fontSize="42px"
                    lineHeight="44px"
                    fontWeight="0"
                    width="308px"
                >
                    NeiroInference
                </Text>
                <Divider />
                <Flex
                    direction="column"
                    width="100%"
                    height="calc(100vh - 500px)"
                    overflowY="auto"
                    padding="16px"
                    bg="gray.100"
                >
                    <MarkdownRenderer markdownText={neiro_inference} />
                </Flex>
                <Divider />


            </VStack>
        </Flex>
    );
};

export default DocumentPage;