use crate::data::loading::{
    inference_data_generator_from_file, inference_data_generator_from_python,
    text_data_generator_from_files, BatchLimitType, BatchedIterator, BufferedIterator, DataGen,
    ItemSize, PipelineIterator, Tensorize, TensorizedIterator, TextIterationStrategy, TextIterator,
};
use crate::data::preprocessing::{preprocessing, PreprocessingFnConfig};
use crate::text::clean;
use crate::tokenization::{
    padding_mask, token_groups_to_sparse_coo_matrix, tokenizer, PaddingMask,
    TensorizedTokenizationInfo, Tokenization, TokenizationInfo, Tokenizer, TokenizerConfig,
};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::py_required_key_error;
use crate::windows::{windows, WindowConfig};
use anyhow::anyhow;
use numpy::ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use self::labeling::{labeling, LabelingConfig};
use self::postprocessing::{postprocessing, PostprocessingFn, PostprocessingFnConfig};
use self::preprocessing::PreprocessingFn;

pub mod labeling;
pub mod loading;
pub mod postprocessing;
pub mod preprocessing;
mod utils;

#[derive(Default, Clone, Debug)]
pub struct TextDataInfo {
    pub seed: u64,
    pub file_idx: usize,
    pub marks: HashMap<String, String>,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct TextData {
    #[pyo3(get)]
    input: String,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    language: Option<String>,
}

impl TextData {
    pub fn new(input: String, target: Option<String>, language: Option<String>) -> Self {
        let target = target.unwrap_or_else(|| input.clone());
        TextData {
            input,
            target,
            language,
        }
    }
}

pub enum TensorizedLabelInfo {
    Empty,
    Generation(Array2<i32>, PaddingMask, Vec<usize>, Option<Vec<usize>>),
}

impl IntoPy<PyObject> for TensorizedLabelInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        if let TensorizedLabelInfo::Generation(token_ids, padding_mask, lengths, prefix_lengths) =
            self
        {
            let token_ids: Py<PyArray2<i32>> = token_ids.into_pyarray(py).into_py(py);
            d.set_item("token_ids", token_ids).unwrap();
            d.set_item("padding_mask", padding_mask.into_py(py))
                .unwrap();
            d.set_item("lengths", lengths).unwrap();
            if let Some(prefix_lengths) = prefix_lengths {
                d.set_item("prefix_lengths", prefix_lengths).unwrap();
            }
        };
        d.into_py(py)
    }
}

#[derive(Clone, Debug)]
pub enum Label {
    Classification(i32),
    SequenceClassification(Vec<i32>),
    Generation(Vec<i32>, u32, Option<usize>),
    Empty,
}

impl IntoPy<PyObject> for Label {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let label_type = match &self {
            Label::Classification(label) => {
                d.set_item("label", label).unwrap();
                "classification"
            }
            Label::SequenceClassification(labels) => {
                d.set_item("labels", labels).unwrap();
                "sequence_classification"
            }
            Label::Generation(labels, pad_token_id, prefix_length) => {
                d.set_item("labels", labels).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                if let Some(prefix_length) = prefix_length {
                    d.set_item("prefix_length", prefix_length).unwrap();
                }
                "generation"
            }
            Label::Empty => "empty",
        };
        d.set_item("type", label_type).unwrap();
        d.into()
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Item {
    #[pyo3(get)]
    pub data: TextData,
    #[pyo3(get)]
    pub tokenization: Tokenization,
    #[pyo3(get)]
    pub label: Label,
}

impl ItemSize for Item {
    fn size(&self) -> usize {
        self.tokenization.token_ids.len()
    }
}

impl Item {
    pub fn new(data: TextData, tokenization: Tokenization, label: Label) -> Self {
        Item {
            data,
            tokenization,
            label,
        }
    }
}

#[pymethods]
impl Item {
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
pub struct DataBatch {
    len: usize,
    batch: Option<Batch<Item>>,
    tensorized: Option<<Batch<Item> as Tensorize>::Output>,
}

type PyDataTensors = (
    Py<PyArray2<i32>>,
    PyObject,
    Vec<usize>,
    PyObject,
    Py<PyArrayDyn<i32>>,
    PyObject,
);

#[pymethods]
impl DataBatch {
    fn __len__(&self) -> usize {
        self.len
    }

    fn items(&mut self) -> anyhow::Result<Batch<Item>> {
        if self.batch.is_none() {
            return Err(anyhow!(
                "can only get items once, because their data is moved"
            ));
        }
        Ok(self.batch.take().unwrap())
    }

    fn tensors(&mut self, py: Python<'_>) -> anyhow::Result<PyDataTensors> {
        if self.tensorized.is_none() {
            return Err(anyhow!(
                "can only get tensors once, because their data is moved"
            ));
        }
        let tensorized = self.tensorized.take().unwrap();
        Ok((
            tensorized.0.into_pyarray(py).into_py(py),
            tensorized.1.into_py(py),
            tensorized.2,
            tensorized.3.into_py(py),
            tensorized.4.into_pyarray(py).into_py(py),
            tensorized.5.into_py(py),
        ))
    }
}

#[inline]
fn join<T>(vectors: Vec<Vec<T>>) -> Vec<T> {
    let mut joined = vec![];
    for mut v in vectors {
        joined.append(&mut v);
    }
    joined
}

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
                let pad_mask = padding_mask(&sparse_mat.group_lengths).expect("should not fail");
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
fn prepare_tokenization(
    tokenizations: &[&Tokenization],
    pad_token_id: u32,
) -> (
    Array2<i32>,
    PaddingMask,
    Vec<usize>,
    TensorizedTokenizationInfo,
) {
    let batch_size = tokenizations.len();
    let max_token_ids = tokenizations
        .iter()
        .map(|t| t.token_ids.len())
        .max()
        .unwrap_or(0);
    let mut token_ids = Vec::with_capacity(max_token_ids * tokenizations.len());
    let mut lengths = Vec::with_capacity(tokenizations.len());
    for &tokenization in tokenizations {
        let num_token_ids = tokenization.token_ids.len();
        token_ids.append(&mut join(vec![
            tokenization
                .token_ids
                .clone()
                .into_iter()
                .map(|t| t as i32)
                .collect(),
            vec![pad_token_id as i32; max_token_ids - num_token_ids],
        ]));
        lengths.push(num_token_ids);
    }
    let token_id_arr = Array2::from_shape_vec((batch_size, max_token_ids), token_ids).unwrap();
    let info = prepare_info(tokenizations, &lengths);
    let pad_mask = padding_mask(&lengths).unwrap();
    (token_id_arr, pad_mask, lengths, info)
}

#[inline]
fn label_lengths(labels: &[&Label]) -> Vec<usize> {
    labels
        .iter()
        .map(|&label| match label {
            Label::Classification(_) => 1,
            Label::SequenceClassification(label, ..) => label.len(),
            Label::Generation(label, ..) => {
                assert!(
                    label.len() > 1,
                    "generation label must be at least two tokens long"
                );
                label.len() - 1
            }
            Label::Empty => 0,
        })
        .collect()
}

#[inline]
fn prepare_label_info(labels: &[&Label]) -> (TensorizedLabelInfo, Vec<usize>, usize) {
    let label_lengths = label_lengths(labels);
    let max_label_length = label_lengths.iter().max().copied().unwrap_or(0);
    let label_info = match labels[0] {
        Label::Generation(..) => {
            let mut label_vec =
                Vec::with_capacity(labels.len() * max_label_length.saturating_sub(1));
            let mut prefix_lengths = Vec::with_capacity(labels.len());
            for (idx, label) in labels.iter().enumerate() {
                if let Label::Generation(label, pad_token_id, prefix_length) = label {
                    label_vec.extend(label.iter().cloned().take(label.len() - 1).chain(vec![
                                *pad_token_id as i32;
                                max_label_length - label_lengths[idx]
                            ]));
                    if let Some(prefix_length) = prefix_length {
                        prefix_lengths.push(*prefix_length);
                    }
                } else {
                    unreachable!()
                }
            }
            let label_arr = Array2::from_shape_vec((labels.len(), max_label_length), label_vec)
                .expect("should not fail");
            TensorizedLabelInfo::Generation(
                label_arr,
                padding_mask(label_lengths.as_slice()).expect("failed to create padding mask"),
                label_lengths.clone(),
                if prefix_lengths.is_empty() {
                    None
                } else {
                    Some(prefix_lengths)
                },
            )
        }
        _ => TensorizedLabelInfo::Empty,
    };
    (label_info, label_lengths, max_label_length)
}

impl Tensorize for Batch<Item> {
    type Output = (
        Array2<i32>,                // token ids
        PaddingMask,                // padding mask
        Vec<usize>,                 // lengths
        TensorizedTokenizationInfo, // additional info
        ArrayD<i32>,                // labels
        TensorizedLabelInfo,        // additional label info
    );

    fn tensorize(&self, tokenizer: &Tokenizer) -> Self::Output {
        assert!(!self.is_empty());
        let (token_id_arr, padding_mask, lengths, info) = prepare_tokenization(
            &self.iter().map(|i| &i.tokenization).collect::<Vec<_>>(),
            tokenizer.pad_token_id(),
        );

        let (label_info, label_lengths, max_label_length) =
            prepare_label_info(&self.iter().map(|i| &i.label).collect::<Vec<_>>());
        let batch_size = self.len();
        let mut labels = Vec::with_capacity(batch_size * max_label_length);
        for (idx, item) in self.iter().enumerate() {
            labels.append(&mut match &item.label {
                Label::Classification(label) => vec![*label],
                Label::SequenceClassification(labels) => labels
                    .iter()
                    .cloned()
                    .chain(vec![-1; max_label_length - label_lengths[idx]])
                    .collect(),
                Label::Generation(labels, ..) => labels
                    .iter()
                    .cloned()
                    .skip(1)
                    .chain(vec![-1; max_label_length - label_lengths[idx]])
                    .collect(),
                Label::Empty => vec![],
            });
        }
        let label_arr = match labels.len() {
            n if n == batch_size => Array1::from_vec(labels).into_dyn(),
            n => Array2::from_shape_vec((batch_size, n / batch_size), labels)
                .unwrap()
                .into_dyn(),
        };
        (
            token_id_arr,
            padding_mask,
            lengths,
            info,
            label_arr,
            label_info,
        )
    }
}

#[pyclass]
pub struct InferenceBatch {
    len: usize,
    batch: Option<Batch<InferenceItem>>,
    tensorized: Option<<Batch<InferenceItem> as Tensorize>::Output>,
}

type PyInferenceTensors = (Py<PyArray2<i32>>, PyObject, Vec<usize>, PyObject);
#[pymethods]
impl InferenceBatch {
    fn __len__(&self) -> usize {
        self.len
    }

    fn data(&mut self) -> anyhow::Result<Batch<InferenceData>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get data before getting items, because they are moved")
            })
            .map(|batch| batch.iter().map(|item| item.data.clone()).collect())
    }

    fn infos(&mut self) -> anyhow::Result<Batch<Option<PyObject>>> {
        self.batch
            .as_ref()
            .ok_or_else(|| {
                anyhow!("can only get infos before getting items, because they are moved")
            })
            .map(|batch| batch.iter().map(|item| item.data.info.clone()).collect())
    }

    fn items(&mut self) -> anyhow::Result<Batch<InferenceItem>> {
        self.batch
            .take()
            .ok_or_else(|| anyhow!("can only get items once, because their data is moved"))
    }

    fn tensors(&mut self, py: Python<'_>) -> anyhow::Result<PyInferenceTensors> {
        let tensorized = self
            .tensorized
            .take()
            .ok_or_else(|| anyhow!("can only get tensors once, because their data is moved"))?;
        Ok((
            tensorized.0.into_pyarray(py).into_py(py),
            tensorized.1.into_py(py),
            tensorized.2,
            tensorized.3.into_py(py),
        ))
    }
}

impl Tensorize for Batch<InferenceItem> {
    type Output = (
        Array2<i32>,
        PaddingMask,
        Vec<usize>,
        TensorizedTokenizationInfo,
    );
    fn tensorize(&self, tokenizer: &Tokenizer) -> Self::Output {
        prepare_tokenization(
            &self.iter().map(|i| &i.tokenization).collect::<Vec<_>>(),
            tokenizer.pad_token_id(),
        )
    }
}

#[derive(Debug, Clone)]
pub enum PreprocessingConfig {
    Single(PreprocessingFnConfig),
    PerFile(Vec<PreprocessingFnConfig>),
}

impl<'a> FromPyObject<'a> for PreprocessingConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let mut errs = vec![];
        match obj.extract::<PreprocessingFnConfig>() {
            Ok(value) => return Ok(PreprocessingConfig::Single(value)),
            Err(e) => errs.push(e),
        };
        match obj.extract::<Vec<PreprocessingFnConfig>>() {
            Ok(value) => return Ok(PreprocessingConfig::PerFile(value)),
            Err(e) => errs.push(e),
        };
        Err(PyTypeError::new_err(format!(
            "failed to extract preprocessing config with the following errors: {errs:#?}"
        )))
    }
}

#[derive(Debug, Clone)]
pub enum PostprocessingConfig {
    Single(PostprocessingFnConfig),
    PerFile(Vec<PostprocessingFnConfig>),
}

impl<'a> FromPyObject<'a> for PostprocessingConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        if let Ok(value) = obj.extract::<PostprocessingFnConfig>() {
            Ok(PostprocessingConfig::Single(value))
        } else if let Ok(value) = obj.extract::<Vec<PostprocessingFnConfig>>() {
            Ok(PostprocessingConfig::PerFile(value))
        } else {
            Err(PyTypeError::new_err(
                "postprocessing config must be a single postprocessing function or a list of postprocessing functions",
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextDataPipelineConfig {
    pub preprocessing: PreprocessingConfig,
    pub labeling: LabelingConfig,
    pub postprocessing: PostprocessingConfig,
}

impl<'a> FromPyObject<'a> for TextDataPipelineConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = obj.extract()?;
        let Some(preprocessing) = d.get_item("preprocessing")? else {
            return Err(py_required_key_error(
                "preprocessing",
                "preprocessing pipeline config",
            ));
        };
        let Some(labeling) = d.get_item("labeling")? else {
            return Err(py_required_key_error(
                "labeling",
                "preprocessing pipeline config",
            ));
        };
        let Some(postprocessing) = d.get_item("postprocessing")? else {
            return Err(py_required_key_error(
                "postprocessing",
                "preprocessing pipeline config",
            ));
        };
        Ok(TextDataPipelineConfig {
            preprocessing: preprocessing.extract()?,
            labeling: labeling.extract()?,
            postprocessing: postprocessing.extract()?,
        })
    }
}

// a pipeline is a function mapping an input to an output,
// and it also sharable across threads
pub type Pipeline<I, O> = Arc<dyn Send + Sync + 'static + Fn(I) -> O>;

pub type TextDataPipeline = Pipeline<(TextData, TextDataInfo), anyhow::Result<Item>>;
pub fn text_data_pipeline_with_tokenizer(
    pipeline_cfg: TextDataPipelineConfig,
    tokenizer_cfg: TokenizerConfig,
    max_length: usize,
) -> anyhow::Result<(TextDataPipeline, Arc<AtomicUsize>)> {
    let tokenizer = tokenizer(tokenizer_cfg)?;
    let max_length = Arc::new(AtomicUsize::new(max_length));
    let preprocess_fn: Box<PreprocessingFn> = match pipeline_cfg.preprocessing {
        PreprocessingConfig::Single(cfg) => {
            let preprocessing = preprocessing(cfg);
            Box::new(preprocessing)
        }
        PreprocessingConfig::PerFile(cfgs) => {
            let preprocessings: Vec<_> = cfgs.into_iter().map(preprocessing).collect();
            Box::new(move |data, info| {
                preprocessings.get(info.file_idx).unwrap_or_else(|| {
                    panic!("could not find preprocessing for file {}", info.file_idx)
                })(data, info)
            })
        }
    };
    let label_fn = labeling(pipeline_cfg.labeling);
    let postprocess_fn: Box<PostprocessingFn> = match pipeline_cfg.postprocessing {
        PostprocessingConfig::Single(cfg) => {
            let postprocessing = postprocessing(cfg, &tokenizer, max_length.clone());
            Box::new(postprocessing)
        }
        PostprocessingConfig::PerFile(cfgs) => {
            let postprocessings: Vec<_> = cfgs
                .into_iter()
                .map(|cfg| postprocessing(cfg, &tokenizer, max_length.clone()))
                .collect();
            Box::new(move |item, info| {
                postprocessings.get(info.file_idx).unwrap_or_else(|| {
                    panic!("could not find postprocessing for file {}", info.file_idx)
                })(item, info)
            })
        }
    };
    Ok((
        Arc::new(move |(data, info)| -> anyhow::Result<Item> {
            let (data, info) = preprocess_fn(data, info)?;
            let item = Item {
                tokenization: tokenizer.tokenize(&data.input, None, None, false)?,
                label: label_fn(&data)?,
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
pub fn inference_pipeline_with_windows(
    tokenizer_cfg: TokenizerConfig,
    window_cfg: WindowConfig,
    normalization: Option<Normalization>,
    clean_text: bool,
    use_graphemes: bool,
) -> anyhow::Result<InferencePipeline> {
    let tok = tokenizer(tokenizer_cfg)?;
    Ok(Arc::new(move |(mut data, info)| {
        if clean_text {
            data.text = clean(&data.text, use_graphemes)
        }
        if let Some(normalization) = normalization {
            data.text = normalize(&data.text, normalization, use_graphemes);
        }
        windows(&data.text, &window_cfg)?
            .iter()
            .enumerate()
            .map(|(w_idx, w)| {
                let tokenization = tok.tokenize(w.str, None, None, true)?;
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

type InferenceDataIter = dyn Iterator<
        Item = (
            Batch<InferenceItem>,
            <Batch<InferenceItem> as Tensorize>::Output,
        ),
    > + Send;
#[pyclass]
struct InferenceLoader {
    iter: Box<InferenceDataIter>,
    iter_err: Arc<Mutex<Option<anyhow::Error>>>,
    #[pyo3(get)]
    min_items: usize,
    #[pyo3(get)]
    splits: Vec<usize>,
}

impl InferenceLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        generators: Vec<Box<dyn DataGen<Item = anyhow::Result<InferenceData>>>>,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        normalization: Option<Normalization>,
        clean_text: bool,
        use_graphemes: bool,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        let pipeline = inference_pipeline_with_windows(
            tokenizer_config.clone(),
            window_config,
            normalization,
            clean_text,
            use_graphemes,
        )?;
        let splits: Vec<usize> = generators.iter().map(|g| g.min_len()).collect();
        let min_items = splits.iter().sum();
        let prefetch_factor = prefetch_factor.max(1);
        let text_iter = TextIterator::new(generators, TextIterationStrategy::Sequential, None)?;
        let iter_err = Arc::new(Mutex::new(None));
        let text_iter_err = iter_err.clone();
        let pipe_iter_err = iter_err.clone();
        let iter = text_iter
            .scan((), move |_, (data, _)| {
                if let Err(e) = data {
                    *text_iter_err.lock().unwrap() = Some(e);
                    None
                } else {
                    data.ok()
                }
            })
            .enumerate()
            .map(|(item_idx, data)| (data, InferenceDataInfo { item_idx }))
            .pipe(pipeline, num_threads)
            .scan((), move |_, item| {
                if let Err(e) = item {
                    *pipe_iter_err.lock().unwrap() = Some(e);
                    None
                } else {
                    item.ok()
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
            .tensorized(tokenizer_config)?
            .buffered(buffer_size);
        Ok(InferenceLoader {
            iter: Box::new(iter),
            iter_err,
            min_items,
            splits,
        })
    }
}

#[pymethods]
impl InferenceLoader {
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature=(
        iterator,
        tokenizer_config,
        window_config,
        normalization = None,
        clean_text = false,
        use_graphemes = true,
        num_threads = num_cpus::get() as u8,
        buffer_size = 128,
        batch_limit = 16,
        batch_limit_type = BatchLimitType::BatchSize,
        prefetch_factor = 1,
        sort = false
    ))]
    pub fn from_iterator(
        iterator: PyObject,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        normalization: Option<Normalization>,
        clean_text: bool,
        use_graphemes: bool,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        // we have to convert the full python iterator
        // here already, because of some weird issues with pyo3 and threading
        let data: Vec<anyhow::Result<_>> = inference_data_generator_from_python(iterator).collect();
        Self::new(
            vec![Box::new(data.into_iter())],
            tokenizer_config,
            window_config,
            normalization,
            clean_text,
            use_graphemes,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            prefetch_factor,
            sort,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature=(
        files,
        tokenizer_config,
        window_config,
        normalization = None,
        clean_text = false,
        use_graphemes = true,
        num_threads = num_cpus::get() as u8,
        buffer_size = 128,
        batch_limit = 16,
        batch_limit_type = BatchLimitType::BatchSize,
        prefetch_factor = 1,
        sort = false
    ))]
    pub fn from_files(
        files: Vec<String>,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        normalization: Option<Normalization>,
        clean_text: bool,
        use_graphemes: bool,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        if files.is_empty() {
            return Err(anyhow!("files is empty"));
        }
        let mut generators = vec![];
        for file in &files {
            let generator = inference_data_generator_from_file(Path::new(file))?;
            generators.push(generator);
        }
        Self::new(
            generators,
            tokenizer_config,
            window_config,
            normalization,
            clean_text,
            use_graphemes,
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
        if let Some((batch, tensorized)) = slf.iter.next() {
            Ok(Some(
                Py::new(
                    slf.py(),
                    InferenceBatch {
                        len: batch.len(),
                        batch: Some(batch),
                        tensorized: Some(tensorized),
                    },
                )
                .expect("should not fail"),
            ))
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

type DataIter = dyn Iterator<Item = (Batch<Item>, <Batch<Item> as Tensorize>::Output)> + Send;
#[pyclass]
struct DataLoader {
    pipeline: TextDataPipeline,
    files: Vec<(String, Option<String>)>,
    languages: Option<Vec<String>>,
    strategy: TextIterationStrategy,
    tokenizer_config: TokenizerConfig,
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
    iter: Option<Box<DataIter>>,
}

impl DataLoader {
    #[allow(clippy::too_many_arguments)]
    fn new(
        files: Vec<(String, Option<String>)>,
        languages: Option<Vec<String>>,
        pipeline_config: TextDataPipelineConfig,
        tokenizer_config: TokenizerConfig,
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
        let (pipeline, max_length) = text_data_pipeline_with_tokenizer(
            pipeline_config,
            tokenizer_config.clone(),
            max_length,
        )?;
        // handle distributed arguments
        let (rank, world_size) = distributed.unwrap_or((0, 1));
        assert!(
            rank < world_size,
            "rank {rank} is invalid given world size {world_size}"
        );
        let limit = limit.unwrap_or(usize::MAX);
        Ok(DataLoader {
            pipeline,
            files,
            languages,
            strategy,
            tokenizer_config,
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
        for (idx, (input_file, target_file)) in self.files.iter().enumerate() {
            let lang = if self.languages.is_some() {
                Some(self.languages.as_ref().unwrap()[idx].clone())
            } else {
                None
            };
            let generator = text_data_generator_from_files(input_file, target_file.as_ref(), lang)?;
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
            .tensorized(self.tokenizer_config.clone())?
            .buffered(self.buffer_size);
        self.iter = Some(Box::new(batch_iter));
        Ok(())
    }
}

#[pymethods]
impl DataLoader {
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (
        files,
        pipeline_config,
        tokenizer_config,
        languages = None,
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
        pipeline_config: TextDataPipelineConfig,
        tokenizer_config: TokenizerConfig,
        languages: Option<Vec<String>>,
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
        if languages.is_some() && files.len() != languages.as_ref().unwrap().len() {
            return Err(anyhow!(
                "there must be one language for every file if specified, but \
                    got {} files and {} languages",
                files.len(),
                languages.as_ref().unwrap().len()
            ));
        }
        Self::new(
            files,
            languages,
            pipeline_config,
            tokenizer_config,
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

    fn __next__(mut slf: PyRefMut<'_, Self>) -> anyhow::Result<Option<Py<DataBatch>>> {
        if slf.iter.is_none() {
            slf.init_iter()?;
        }
        let next = if let Some((batch, tensorized)) = slf.iter.as_mut().unwrap().next() {
            Some(
                Py::new(
                    slf.py(),
                    DataBatch {
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
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "data")?;
    m.add_class::<DataLoader>()?;
    m.add_class::<InferenceLoader>()?;
    m.add_class::<TextData>()?;
    m.add_class::<InferenceData>()?;
    m.add_class::<Item>()?;
    m.add_class::<InferenceItem>()?;
    m.add_class::<DataBatch>()?;
    m.add_class::<InferenceBatch>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
