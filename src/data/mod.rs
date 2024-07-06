use crate::data::loading::{
    inference_data_iterator_from_python, train_data_generator_from_files, BatchLimitType,
    BatchedIterator, BufferedIterator, ItemSize, PipelineIterator, Tensorize, TensorizedIterator,
    TextIterationStrategy, TextIterator,
};
use crate::data::preprocessing::{preprocessing, PreprocessingFnConfig};
use crate::tokenization::{
    padding_mask, token_groups_to_sparse_coo_matrix, tokenizer, TensorizedTokenizationInfo,
    Tokenization, TokenizationInfo, TokenizerConfig,
};
use crate::unicode::CS;
use crate::utils::{py_invalid_type_error, py_required_key_error};
use crate::windows::{windows, WindowConfig};
use anyhow::anyhow;
use itertools::Itertools;
use numpy::ndarray::prelude::*;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::iter::repeat;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use self::postprocessing::{postprocessing, PostprocessingFn, PostprocessingFnConfig};
use self::preprocessing::PreprocessingFn;
use self::task::{train_task, TrainTaskConfig};

pub mod loading;
pub mod postprocessing;
pub mod preprocessing;
pub mod task;
mod utils;

#[derive(Default, Clone, Debug)]
pub struct TextDataInfo {
    pub seed: u64,
    pub file_idx: usize,
    pub marks: HashMap<String, String>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct TrainData {
    #[pyo3(get)]
    input: String,
    #[pyo3(get)]
    target: String,
}

impl TrainData {
    pub fn new(input: String, target: Option<String>) -> Self {
        let target = target.unwrap_or_else(|| input.clone());
        TrainData { input, target }
    }
}

#[derive(Clone, Debug)]
pub enum TrainTaskInput {
    Classification {
        token_ids: Vec<u32>,
        pad_token_id: u32,
        label: i32,
    },
    SequenceClassification {
        token_ids: Vec<u32>,
        pad_token_id: u32,
        labels: Vec<i32>,
    },
    Generation {
        token_ids: Vec<u32>,
        pad_token_id: u32,
        labels: Vec<i32>,
    },
    ConditionalGeneration {
        token_ids: Vec<u32>,
        pad_token_id: u32,
        target_token_ids: Vec<u32>,
        target_pad_token_id: u32,
        labels: Vec<i32>,
    },
}

impl TrainTaskInput {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            TrainTaskInput::Classification { token_ids, .. } => token_ids.len(),
            TrainTaskInput::SequenceClassification { token_ids, .. } => token_ids.len(),
            TrainTaskInput::Generation { token_ids, .. } => token_ids.len(),
            TrainTaskInput::ConditionalGeneration {
                token_ids,
                target_token_ids,
                ..
            } => token_ids.len() + target_token_ids.len(),
        }
    }
}

impl IntoPy<PyObject> for TrainTaskInput {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new_bound(py);
        let label_type = match &self {
            TrainTaskInput::Classification {
                token_ids,
                pad_token_id,
                label,
            } => {
                d.set_item("token_ids", token_ids).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                d.set_item("label", label).unwrap();
                "classification"
            }
            TrainTaskInput::SequenceClassification {
                token_ids,
                pad_token_id,
                labels,
            } => {
                d.set_item("token_ids", token_ids).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                d.set_item("labels", labels).unwrap();
                "sequence_classification"
            }
            TrainTaskInput::Generation {
                token_ids,
                pad_token_id,
                labels,
            } => {
                d.set_item("token_ids", token_ids).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                d.set_item("labels", labels).unwrap();
                "generation"
            }
            TrainTaskInput::ConditionalGeneration {
                token_ids,
                pad_token_id,
                target_token_ids,
                target_pad_token_id,
                labels,
            } => {
                d.set_item("token_ids", token_ids).unwrap();
                d.set_item("target_token_ids", target_token_ids).unwrap();
                d.set_item("labels", labels).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                d.set_item("target_pad_token_id", target_pad_token_id)
                    .unwrap();
                "conditional_generation"
            }
        };
        d.set_item("type", label_type).unwrap();
        d.into()
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct TrainItem {
    #[pyo3(get)]
    pub data: TrainData,
    #[pyo3(get)]
    pub input: TrainTaskInput,
}

impl ItemSize for TrainItem {
    fn size(&self) -> usize {
        self.input.len()
    }
}

impl TrainItem {
    pub fn new(data: TrainData, label: TrainTaskInput) -> Self {
        TrainItem { data, input: label }
    }
}

#[pymethods]
impl TrainItem {
    fn __len__(&self) -> usize {
        self.size()
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct InferenceData {
    #[pyo3(get, set)]
    text: String,
    #[pyo3(get, set)]
    info: Option<PyObject>,
}

#[pymethods]
impl InferenceData {
    #[new]
    #[pyo3(signature = (text, info = None))]
    pub fn new(text: String, info: Option<PyObject>) -> Self {
        Self { text, info }
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct InferenceItem {
    #[pyo3(get)]
    pub data: InferenceData,
    #[pyo3(get)]
    pub tokenization: Tokenization,
    #[pyo3(get)]
    pub item_idx: usize,
    #[pyo3(get)]
    pub window_idx: usize,
    #[pyo3(get)]
    pub window: (usize, usize, usize, usize),
    #[pyo3(get)]
    pub byte_window: (usize, usize, usize, usize),
}

impl InferenceItem {
    pub fn new(
        data: InferenceData,
        tokenization: Tokenization,
        item_idx: usize,
        window_idx: usize,
        window: (usize, usize, usize, usize),
        byte_window: (usize, usize, usize, usize),
    ) -> Self {
        InferenceItem {
            data,
            tokenization,
            item_idx,
            window_idx,
            window,
            byte_window,
        }
    }
}

impl ItemSize for InferenceItem {
    fn size(&self) -> usize {
        self.tokenization.token_ids.len()
    }
}

#[pymethods]
impl InferenceItem {
    fn __len__(&self) -> usize {
        self.size()
    }

    fn window_bytes(&self) -> usize {
        self.byte_window.2 - self.byte_window.1
    }

    fn context_bytes(&self) -> usize {
        self.byte_window.3 - self.byte_window.0
    }

    fn window_str(&self) -> String {
        self.data.text[self.byte_window.1..self.byte_window.2].to_string()
    }

    fn context_str(&self) -> String {
        self.data.text[self.byte_window.0..self.byte_window.3].to_string()
    }

    fn window_chars(&self) -> Vec<String> {
        let cs = CS::new(
            &self.data.text[self.byte_window.1..self.byte_window.2],
            true,
        );
        cs.chars().map(|c| c.str.to_string()).collect()
    }

    fn context_chars(&self) -> Vec<String> {
        let cs = CS::new(
            &self.data.text[self.byte_window.0..self.byte_window.3],
            true,
        );
        cs.chars().map(|c| c.str.to_string()).collect()
    }
}

pub type Batch<T> = Vec<T>;

#[pyclass]
pub struct TrainBatch {
    len: usize,
    batch: Option<Batch<TrainItem>>,
    tensorized: Option<<Batch<TrainItem> as Tensorize>::Output>,
}

#[pymethods]
impl TrainBatch {
    fn __len__(&self) -> usize {
        self.len
    }

    fn sizes(&self) -> anyhow::Result<Vec<usize>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get sizes before getting items, because they are moved")
            })
            .map(|batch| batch.iter().map(|item| item.size()).collect())
    }

    fn items(&mut self) -> anyhow::Result<Batch<TrainItem>> {
        self.batch
            .take()
            .ok_or_else(|| anyhow!("can only get items once, because their data is moved"))
    }

    fn tensors(&mut self, py: Python<'_>) -> anyhow::Result<PyObject> {
        let tensorized = self
            .tensorized
            .take()
            .ok_or_else(|| anyhow!("can only get tensors once, because their data is moved"))?;
        tensorized.into_py(py)
    }
}

#[allow(dead_code)]
#[inline]
fn prepare_info(tokenizations: &[&Tokenization], lengths: &[usize]) -> TensorizedTokenizationInfo {
    match tokenizations[0].info {
        TokenizationInfo::Empty => TensorizedTokenizationInfo::Empty,
        TokenizationInfo::TokenGroups(_) => {
            let mut info = HashMap::new();
            let mut all_groupings: HashMap<_, Vec<_>> = HashMap::new();
            for &tokenization in tokenizations {
                if let TokenizationInfo::TokenGroups(token_groups) = &tokenization.info {
                    for (key, grouping) in token_groups {
                        all_groupings.entry(key).or_default().push(grouping);
                    }
                } else {
                    panic!("should not happen");
                }
            }
            for (name, groupings) in all_groupings {
                let sparse_mat = token_groups_to_sparse_coo_matrix(&groupings, lengths)
                    .expect("should not fail");
                let pad_mask = padding_mask(&sparse_mat.group_lengths);
                info.insert(name.clone(), (sparse_mat, pad_mask));
            }
            TensorizedTokenizationInfo::TokenGroups(info)
        }
        TokenizationInfo::Info(_) => {
            unimplemented!()
        }
    }
}

#[inline]
fn pad_ids<T: num::PrimInt>(ids: &[impl AsRef<[T]>], pad_id: T) -> (Array2<T>, Array1<usize>) {
    let batch_size = ids.len();
    let max_len = ids
        .iter()
        .map(|t| t.as_ref().len())
        .max()
        .unwrap_or_default();
    let mut padded_ids = Vec::with_capacity(max_len * batch_size);
    let mut lengths = Vec::with_capacity(batch_size);
    for id in ids {
        padded_ids.extend(id.as_ref().iter().cloned());
        padded_ids.extend(repeat(pad_id).take(max_len - id.as_ref().len()));
        lengths.push(id.as_ref().len());
    }
    (
        Array2::from_shape_vec((batch_size, max_len), padded_ids).expect("should not happen"),
        Array1::from_vec(lengths),
    )
}

pub enum TensorizedTrainTaskInput {
    Classification(Array2<u32>, Array1<usize>, Array1<i32>),
    SequenceClassification(Array2<u32>, Array1<usize>, Array2<i32>),
    Generation(Array2<u32>, Array1<usize>, Array2<i32>),
    ConditionalGeneration(
        Array2<u32>,
        Array1<usize>,
        Array2<u32>,
        Array1<usize>,
        Array2<i32>,
    ),
}

impl TensorizedTrainTaskInput {
    pub fn into_py(self, py: Python<'_>) -> anyhow::Result<PyObject> {
        let d = PyDict::new_bound(py);
        let input_type = match self {
            TensorizedTrainTaskInput::Classification(token_ids, lengths, labels) => {
                d.set_item("token_ids", token_ids.into_pyarray_bound(py))?;
                d.set_item("lengths", lengths.into_pyarray_bound(py))?;
                d.set_item("labels", labels.into_pyarray_bound(py))?;
                "classification"
            }
            TensorizedTrainTaskInput::SequenceClassification(token_ids, lengths, labels) => {
                d.set_item("token_ids", token_ids.into_pyarray_bound(py))?;
                d.set_item("lengths", lengths.into_pyarray_bound(py))?;
                d.set_item("labels", labels.into_pyarray_bound(py))?;
                "sequence_classification"
            }
            TensorizedTrainTaskInput::Generation(token_ids, lengths, labels) => {
                d.set_item("token_ids", token_ids.into_pyarray_bound(py))?;
                d.set_item("lengths", lengths.into_pyarray_bound(py))?;
                d.set_item("labels", labels.into_pyarray_bound(py))?;
                "generation"
            }
            TensorizedTrainTaskInput::ConditionalGeneration(
                token_ids,
                lengths,
                target_token_ids,
                target_lengths,
                labels,
            ) => {
                d.set_item("token_ids", token_ids.into_pyarray_bound(py))?;
                d.set_item("lengths", lengths.into_pyarray_bound(py))?;
                d.set_item("labels", labels.into_pyarray_bound(py))?;
                d.set_item("target_token_ids", target_token_ids.into_pyarray_bound(py))?;
                d.set_item("target_lengths", target_lengths.into_pyarray_bound(py))?;
                "conditional_generation"
            }
        };
        d.set_item("type", input_type)?;
        Ok(d.into())
    }
}

impl Tensorize for Batch<TrainItem> {
    type Output = TensorizedTrainTaskInput;

    fn tensorize(&self) -> Self::Output {
        match self.first().expect("empty batch should not happen").input {
            TrainTaskInput::Classification { pad_token_id, .. } => {
                let (token_ids, labels): (Vec<_>, _) = self
                    .iter()
                    .filter_map(|item| {
                        if let TrainTaskInput::Classification {
                            token_ids, label, ..
                        } = &item.input
                        {
                            Some((token_ids, *label))
                        } else {
                            None
                        }
                    })
                    .unzip();
                let (token_ids, lengths) = pad_ids(&token_ids, pad_token_id);
                TensorizedTrainTaskInput::Classification(
                    token_ids,
                    lengths,
                    Array1::from_vec(labels),
                )
            }
            TrainTaskInput::SequenceClassification { pad_token_id, .. } => {
                let (token_ids, labels): (Vec<_>, Vec<_>) = self
                    .iter()
                    .filter_map(|item| {
                        if let TrainTaskInput::SequenceClassification {
                            token_ids, labels, ..
                        } = &item.input
                        {
                            Some((token_ids, labels))
                        } else {
                            None
                        }
                    })
                    .unzip();
                let (token_ids, lengths) = pad_ids(&token_ids, pad_token_id);
                let (labels, _) = pad_ids(&labels, -1);
                TensorizedTrainTaskInput::SequenceClassification(token_ids, lengths, labels)
            }
            TrainTaskInput::Generation { pad_token_id, .. } => {
                let (token_ids, labels): (Vec<_>, Vec<_>) = self
                    .iter()
                    .filter_map(|item| {
                        if let TrainTaskInput::Generation {
                            token_ids, labels, ..
                        } = &item.input
                        {
                            Some((token_ids, labels))
                        } else {
                            None
                        }
                    })
                    .unzip();
                let (token_ids, lengths) = pad_ids(&token_ids, pad_token_id);
                let (labels, _) = pad_ids(&labels, -1);
                TensorizedTrainTaskInput::Generation(token_ids, lengths, labels)
            }
            TrainTaskInput::ConditionalGeneration {
                pad_token_id,
                target_pad_token_id,
                ..
            } => {
                let (token_ids, target_token_ids, labels): (Vec<_>, Vec<_>, Vec<_>) = self
                    .iter()
                    .filter_map(|item| {
                        if let TrainTaskInput::ConditionalGeneration {
                            token_ids,
                            target_token_ids,
                            labels,
                            ..
                        } = &item.input
                        {
                            Some((token_ids, target_token_ids, labels))
                        } else {
                            None
                        }
                    })
                    .multiunzip();
                let (token_ids, lengths) = pad_ids(&token_ids, pad_token_id);
                let (target_token_ids, target_lengths) =
                    pad_ids(&target_token_ids, target_pad_token_id);
                let (labels, _) = pad_ids(&labels, -1);
                TensorizedTrainTaskInput::ConditionalGeneration(
                    token_ids,
                    lengths,
                    target_token_ids,
                    target_lengths,
                    labels,
                )
            }
        }
    }
}

#[pyclass]
pub struct InferenceBatch {
    len: usize,
    batch: Option<Batch<InferenceItem>>,
}

#[pymethods]
impl InferenceBatch {
    fn __len__(&self) -> usize {
        self.len
    }

    fn sizes(&self) -> anyhow::Result<Vec<usize>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get sizes before getting items, because they are moved")
            })
            .map(|batch| batch.iter().map(|item| item.size()).collect())
    }

    fn infos(&self) -> anyhow::Result<Batch<Option<PyObject>>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get infos before getting items, because they are moved")
            })
            .map(|batch| batch.iter().map(|item| item.data.info.clone()).collect())
    }

    fn token_ids(&self) -> anyhow::Result<Batch<Vec<u32>>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get token ids before getting items, because they are moved")
            })
            .map(|batch| {
                batch
                    .iter()
                    .map(|item| item.tokenization.token_ids.clone())
                    .collect()
            })
    }

    fn items(&mut self) -> anyhow::Result<Batch<InferenceItem>> {
        self.batch
            .take()
            .ok_or_else(|| anyhow!("can only get items once, because their data is moved"))
    }
}

#[derive(Debug, Clone)]
pub enum PreprocessingConfig {
    Global(PreprocessingFnConfig),
    PerSource(Vec<PreprocessingFnConfig>),
}

impl<'a> FromPyObject<'a> for PreprocessingConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = obj.extract()?;
        let Some(preprocessing_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "preprocessing config"));
        };
        let preprocessing_type: String = preprocessing_type.extract()?;
        let preprocessing_cfg = match preprocessing_type.as_str() {
            "global" => {
                let Some(preprocessing) = d.get_item("fn")? else {
                    return Err(py_required_key_error("fn", "preprocessing config"));
                };
                PreprocessingConfig::Global(preprocessing.extract()?)
            }
            "per_source" => {
                let Some(preprocessings) = d.get_item("fns")? else {
                    return Err(py_required_key_error("fns", "preprocessing config"));
                };
                PreprocessingConfig::PerSource(preprocessings.extract()?)
            }
            k => return Err(py_invalid_type_error(k, "preprocessing config")),
        };
        Ok(preprocessing_cfg)
    }
}

#[derive(Debug, Clone)]
pub enum PostprocessingConfig {
    Global(PostprocessingFnConfig),
    PerSource(Vec<PostprocessingFnConfig>),
}

impl<'a> FromPyObject<'a> for PostprocessingConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = obj.extract()?;
        let Some(postprocessing_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "postprocessing config"));
        };
        let postprocessing_type: String = postprocessing_type.extract()?;
        let postprocessing_cfg = match postprocessing_type.as_str() {
            "global" => {
                let Some(postprocessing) = d.get_item("fn")? else {
                    return Err(py_required_key_error("fn", "postprocessing config"));
                };
                PostprocessingConfig::Global(postprocessing.extract()?)
            }
            "per_source" => {
                let Some(postprocessings) = d.get_item("fns")? else {
                    return Err(py_required_key_error("fns", "postprocessing config"));
                };
                PostprocessingConfig::PerSource(postprocessings.extract()?)
            }
            k => return Err(py_invalid_type_error(k, "postprocessing config")),
        };
        Ok(postprocessing_cfg)
    }
}

#[derive(Debug, Clone)]
pub struct TrainPipelineConfig {
    pub preprocessing: PreprocessingConfig,
    pub task: TrainTaskConfig,
    pub postprocessing: PostprocessingConfig,
}

impl<'a> FromPyObject<'a> for TrainPipelineConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = obj.extract()?;
        let Some(preprocessing) = d.get_item("preprocessing")? else {
            return Err(py_required_key_error("preprocessing", "pipeline config"));
        };
        let Some(task) = d.get_item("task")? else {
            return Err(py_required_key_error("task", "pipeline config"));
        };
        let Some(postprocessing) = d.get_item("postprocessing")? else {
            return Err(py_required_key_error("postprocessing", "pipeline config"));
        };
        Ok(TrainPipelineConfig {
            preprocessing: preprocessing.extract()?,
            task: task.extract()?,
            postprocessing: postprocessing.extract()?,
        })
    }
}

pub enum InferenceTaskConfig {}

pub struct InferencePipelineConfig {
    pub preprocessing: PreprocessingFnConfig,
    pub task: InferenceTaskConfig,
}

// a pipeline is a function mapping an input to an output,
// and it also sharable across threads
pub type Pipeline<I, O> = Arc<dyn Send + Sync + 'static + Fn(I) -> O>;

pub type TrainPipeline = Pipeline<(TrainData, TextDataInfo), anyhow::Result<TrainItem>>;
pub fn train_pipeline(
    pipeline_cfg: TrainPipelineConfig,
    max_length: usize,
) -> anyhow::Result<(TrainPipeline, Arc<AtomicUsize>)> {
    let max_length = Arc::new(AtomicUsize::new(max_length));
    let preprocess_fn: Box<PreprocessingFn> = match pipeline_cfg.preprocessing {
        PreprocessingConfig::Global(cfg) => {
            let preprocessing = preprocessing(cfg);
            Box::new(preprocessing)
        }
        PreprocessingConfig::PerSource(cfgs) => {
            let preprocessings: Vec<_> = cfgs.into_iter().map(preprocessing).collect();
            Box::new(move |data, info| {
                preprocessings.get(info.file_idx).unwrap_or_else(|| {
                    panic!("could not find preprocessing for file {}", info.file_idx)
                })(data, info)
            })
        }
    };
    let task_fn = train_task(pipeline_cfg.task);
    let postprocess_fn: Box<PostprocessingFn> = match pipeline_cfg.postprocessing {
        PostprocessingConfig::Global(cfg) => {
            let postprocessing = postprocessing(cfg, max_length.clone());
            Box::new(postprocessing)
        }
        PostprocessingConfig::PerSource(cfgs) => {
            let postprocessings: Vec<_> = cfgs
                .into_iter()
                .map(|cfg| postprocessing(cfg, max_length.clone()))
                .collect();
            Box::new(move |item, info| {
                postprocessings.get(info.file_idx).unwrap_or_else(|| {
                    panic!("could not find postprocessing for file {}", info.file_idx)
                })(item, info)
            })
        }
    };
    Ok((
        Arc::new(move |(data, info)| -> anyhow::Result<TrainItem> {
            let (data, info) = preprocess_fn(data, info)?;
            let item = TrainItem {
                input: task_fn(&data)?,
                data,
            };
            let (item, _) = postprocess_fn(item, info)?;
            Ok(item)
        }),
        max_length,
    ))
}

pub struct InferenceDataInfo {
    pub item_idx: usize,
}
pub type InferencePipeline =
    Pipeline<(InferenceData, InferenceDataInfo), anyhow::Result<Vec<InferenceItem>>>;
pub fn inference_pipeline(
    window_cfg: WindowConfig,
    tokenizer_cfg: TokenizerConfig,
    ignore_special_tokens: bool,
) -> anyhow::Result<InferencePipeline> {
    let tok = tokenizer(tokenizer_cfg)?;
    Ok(Arc::new(move |(data, info)| {
        windows(&data.text, &window_cfg)?
            .iter()
            .enumerate()
            .map(|(w_idx, w)| {
                let tokenization = tok.tokenize(w.str, ignore_special_tokens)?;
                Ok(InferenceItem::new(
                    data.clone(),
                    tokenization,
                    info.item_idx,
                    w_idx,
                    w.boundaries(),
                    w.byte_boundaries(),
                ))
            })
            .collect()
    }))
}

type InferenceDataIter = dyn Iterator<Item = Batch<InferenceItem>> + Send;
#[pyclass]
struct InferenceLoader {
    iter: Box<InferenceDataIter>,
    iter_err: Arc<Mutex<Option<anyhow::Error>>>,
}

impl InferenceLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        iter: impl Iterator<Item = anyhow::Result<InferenceData>> + Send + 'static,
        tokenizer: TokenizerConfig,
        ignore_special_tokens: bool,
        window: WindowConfig,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        let pipeline = inference_pipeline(window, tokenizer, ignore_special_tokens)?;
        let prefetch_factor = prefetch_factor.max(1);
        let iter_err = Arc::new(Mutex::new(None));
        let iter = iter
            .scan(iter_err.clone(), move |err, item| match item {
                Ok(item) => Some(item),
                Err(e) => {
                    if let Ok(mut lock) = err.lock() {
                        *lock = Some(e);
                    };
                    None
                }
            })
            .enumerate()
            .map(|(item_idx, data)| (data, InferenceDataInfo { item_idx }))
            .pipe(pipeline, num_threads)
            .scan(iter_err.clone(), move |err, item| match item {
                Ok(item) => Some(item),
                Err(e) => {
                    if let Ok(mut lock) = err.lock() {
                        *lock = Some(e);
                    };
                    None
                }
            })
            .flatten()
            .batched(
                sort,
                false,
                prefetch_factor,
                batch_limit,
                batch_limit_type,
                None,
            )
            .buffered(buffer_size);
        Ok(InferenceLoader {
            iter: Box::new(iter),
            iter_err,
        })
    }
}

#[pymethods]
impl InferenceLoader {
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature=(
        iterator,
        tokenizer,
        window,
        ignore_special_tokens = false,
        num_threads = num_cpus::get() as u8,
        buffer_size = 128,
        batch_limit = 16,
        batch_limit_type = BatchLimitType::BatchSize,
        prefetch_factor = 1,
        sort = false
    ))]
    pub fn from_iterator(
        iterator: PyObject,
        tokenizer: TokenizerConfig,
        window: WindowConfig,
        ignore_special_tokens: bool,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        let data: Vec<_> = inference_data_iterator_from_python(iterator).collect();
        Self::new(
            data.into_iter(),
            tokenizer,
            ignore_special_tokens,
            window,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            prefetch_factor,
            sort,
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> anyhow::Result<Option<Py<InferenceBatch>>> {
        if let Some(batch) = slf.iter.next() {
            Ok(Some(Py::new(
                slf.py(),
                InferenceBatch {
                    len: batch.len(),
                    batch: Some(batch),
                },
            )?))
        } else {
            // check if batch is None because iterator is stopped,
            // or because an error was encountered
            match slf.iter_err.lock().unwrap().as_ref() {
                Some(e) => Err(anyhow!("error in inference iterator: {e}")),
                None => Ok(None),
            }
        }
    }
}

type TrainIter =
    dyn Iterator<Item = (Batch<TrainItem>, <Batch<TrainItem> as Tensorize>::Output)> + Send;
#[pyclass]
struct TrainLoader {
    pipeline: TrainPipeline,
    files: Vec<(String, Option<String>)>,
    strategy: TextIterationStrategy,
    num_threads: u8,
    buffer_size: usize,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    max_length: Arc<AtomicUsize>,
    epoch: usize,
    fast_forward: usize,
    limit: usize,
    skip: usize,
    rank: usize,
    world_size: usize,
    seed: Option<u64>,
    shuffle: bool,
    prefetch_factor: usize,
    sort: bool,
    // the next to values will be set after each __iter__ call
    #[pyo3(get)]
    min_items: Option<usize>,
    iter: Option<Box<TrainIter>>,
}

impl TrainLoader {
    #[allow(clippy::too_many_arguments)]
    fn new(
        files: Vec<(String, Option<String>)>,
        pipeline: TrainPipelineConfig,
        strategy: TextIterationStrategy,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        max_length: usize,
        shuffle: bool,
        prefetch_factor: usize,
        sort: bool,
        seed: Option<u64>,
        skip: usize,
        limit: Option<usize>,
        distributed: Option<(usize, usize)>,
    ) -> anyhow::Result<Self> {
        if shuffle && seed.is_none() {
            return Err(anyhow!("seed cannot be None if shuffle is true",));
        }
        let prefetch_factor = prefetch_factor.max(1);
        let (pipeline, max_length) = train_pipeline(pipeline, max_length)?;
        // handle distributed arguments
        let (rank, world_size) = distributed.unwrap_or((0, 1));
        assert!(
            rank < world_size,
            "rank {rank} is invalid given world size {world_size}"
        );
        let limit = limit.unwrap_or(usize::MAX);
        Ok(TrainLoader {
            pipeline,
            files,
            strategy,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            max_length,
            iter: None,
            min_items: None,
            epoch: 0,
            fast_forward: 0,
            limit,
            skip,
            rank,
            world_size,
            seed,
            shuffle,
            prefetch_factor,
            sort,
        })
    }

    fn init_iter(&mut self) -> anyhow::Result<()> {
        let seed = self.seed.unwrap_or(0) + self.epoch as u64;
        let mut generators = vec![];
        for (input_file, target_file) in self.files.iter() {
            let generator = train_data_generator_from_files(input_file, target_file.as_ref())?;
            generators.push(generator);
        }

        let text_iter = TextIterator::new(generators, self.strategy, Some(seed))?;
        self.min_items = Some(
            text_iter
                .min_len()
                .min(self.limit)
                .saturating_sub(self.skip),
        );
        let batch_iter = text_iter
            .enumerate()
            .take(self.limit)
            .skip(self.skip + self.fast_forward + self.rank)
            .step_by(self.world_size)
            .filter_map(move |(item_idx, (d, file_idx))| {
                if let Ok(d) = d {
                    Some((
                        d,
                        TextDataInfo {
                            file_idx,
                            seed: seed + item_idx as u64,
                            ..Default::default()
                        },
                    ))
                } else {
                    None
                }
            })
            .pipe(self.pipeline.clone(), self.num_threads)
            .filter_map(|i| i.ok())
            .batched(
                self.sort,
                self.shuffle,
                self.prefetch_factor,
                self.batch_limit,
                self.batch_limit_type,
                Some(seed),
            )
            .tensorized()
            .buffered(self.buffer_size);
        self.iter = Some(Box::new(batch_iter));
        Ok(())
    }
}

#[pymethods]
impl TrainLoader {
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (
        files,
        pipeline,
        strategy = TextIterationStrategy::Sequential,
        num_threads = num_cpus::get() as u8,
        buffer_size = 128,
        batch_limit = 16,
        batch_limit_type = BatchLimitType::BatchSize,
        max_length = 512,
        shuffle = false,
        prefetch_factor = 4,
        sort = false,
        seed = None,
        skip = 0,
        limit = None,
        distributed = None,
    ))]
    pub fn from_files(
        files: Vec<(String, Option<String>)>,
        pipeline: TrainPipelineConfig,
        strategy: TextIterationStrategy,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        max_length: usize,
        shuffle: bool,
        prefetch_factor: usize,
        sort: bool,
        seed: Option<u64>,
        skip: usize,
        limit: Option<usize>,
        distributed: Option<(usize, usize)>,
    ) -> anyhow::Result<Self> {
        if files.is_empty() {
            return Err(anyhow!("no files specified"));
        }
        Self::new(
            files,
            pipeline,
            strategy,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            max_length,
            shuffle,
            prefetch_factor,
            sort,
            seed,
            skip,
            limit,
            distributed,
        )
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> anyhow::Result<PyRefMut<'_, Self>> {
        slf.init_iter()?;
        Ok(slf)
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> anyhow::Result<Option<Py<TrainBatch>>> {
        if slf.iter.is_none() {
            slf.init_iter()?;
        }
        let next = if let Some((batch, tensorized)) = slf.iter.as_mut().unwrap().next() {
            Some(
                Py::new(
                    slf.py(),
                    TrainBatch {
                        len: batch.len(),
                        batch: Some(batch),
                        tensorized: Some(tensorized),
                    },
                )
                .expect("should not fail"),
            )
        } else {
            None
        };
        Ok(next)
    }

    #[getter]
    fn max_length(&self) -> usize {
        self.max_length.load(Ordering::Relaxed)
    }

    fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    fn set_fast_forward(&mut self, num_items: usize) {
        self.fast_forward = num_items
    }

    fn set_max_length(&mut self, max_length: usize) {
        self.max_length.swap(max_length, Ordering::SeqCst);
    }
}

/// A submodule containing functionality for text data loading.
/// Currently supported:
/// - loading text files
/// - several loading strategies (sequential, interleaved, weighted)
/// - single or multi-threaded preprocessing
/// - batched loading (limited by a max batch size or a max number of tokens)
/// - distributed loading (distribute work across multiple processes or machines)
pub(super) fn add_submodule(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "data")?;
    m.add_class::<TrainLoader>()?;
    m.add_class::<TrainData>()?;
    m.add_class::<TrainItem>()?;
    m.add_class::<TrainBatch>()?;
    m.add_class::<InferenceLoader>()?;
    m.add_class::<InferenceData>()?;
    m.add_class::<InferenceItem>()?;
    m.add_class::<InferenceBatch>()?;
    parent_module.add_submodule(&m)?;

    Ok(())
}
