use crate::data::loading::{
    inference_data_generator_from_file, inference_data_generator_from_python,
    text_data_generator_from_files, BatchLimitType, BatchedIterator, BufferedIterator, DataGen,
    ItemSize, PipelineIterator, Tensorize, TensorizedIterator, TextIterationStrategy, TextIterator,
};
use crate::data::preprocessing::{labeling, preprocessing, LabelingConfig, PreprocessingFnConfig};
use crate::text::clean;
use crate::tokenization::{
    padding_mask, token_groups_to_sparse_coo_matrix, tokenizer, PaddingMask,
    TensorizedTokenizationInfo, Tokenization, TokenizationInfo, Tokenizer, TokenizerConfig,
    LANG_UNK,
};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::{py_invalid_type_error, py_required_key_error};
use crate::windows::{windows, WindowConfig};
use anyhow::{anyhow, Context};
use numpy::ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub mod loading;
pub mod preprocessing;

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[pyclass]
pub struct TextData {
    #[pyo3(get)]
    original: String,
    #[pyo3(get)]
    processed: String,
    #[pyo3(get)]
    language: Option<String>,
}

impl TextData {
    pub fn new(original: String, processed: Option<String>, language: Option<String>) -> Self {
        let processed = processed.unwrap_or_else(|| original.clone());
        TextData {
            original,
            processed,
            language,
        }
    }
}

#[pymethods]
impl TextData {
    #[new]
    #[pyo3(signature = (original, processed = None, language = None))]
    fn new_py(
        original: String,
        processed: Option<String>,
        language: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self::new(original, processed, language))
    }

    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        op.matches(self.cmp(other))
    }
}

pub enum TensorizedLabelInfo {
    Empty,
    SequenceGeneration((Array2<i32>, PaddingMask, Vec<usize>)),
}

impl IntoPy<PyObject> for TensorizedLabelInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        if let TensorizedLabelInfo::SequenceGeneration((token_ids, padding_mask, lengths)) = self {
            let token_ids: Py<PyArray2<i32>> = token_ids.into_pyarray(py).into_py(py);
            d.set_item("token_ids", token_ids).unwrap();
            d.set_item("padding_mask", padding_mask.into_py(py))
                .unwrap();
            d.set_item("lengths", lengths).unwrap();
        };
        d.into_py(py)
    }
}

#[derive(Clone, Debug)]
pub enum Label {
    Classification(i32),
    SequenceClassification(Vec<i32>),
    SequenceGeneration(Vec<i32>, u32),
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
            Label::SequenceGeneration(labels, pad_token_id) => {
                d.set_item("labels", labels).unwrap();
                d.set_item("pad_token_id", pad_token_id).unwrap();
                "sequence_generation"
            }
        };
        d.set_item("type", label_type).unwrap();
        d.into()
    }
}

impl<'a> FromPyObject<'a> for Label {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(label_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "label"));
        };
        let label_type: String = label_type.extract()?;
        let label = match label_type.as_str() {
            "classification" => {
                let Some(label) = d.get_item("label") else {
                    return Err(py_required_key_error(
                        "label",
                        "classification label"));
                };
                Label::Classification(label.extract()?)
            }
            "sequence_classification" => {
                let Some(labels) = d.get_item("labels") else {
                    return Err(py_required_key_error(
                        "labels",
                        "sequence classification label"));
                };
                Label::SequenceClassification(labels.extract()?)
            }
            "sequence_generation" => {
                let Some(labels) = d.get_item("labels") else {
                    return Err(py_required_key_error(
                        "labels",
                        "sequence generation label"));
                };
                let Some(pad_token_id) = d.get_item("pad_token_id") else {
                    return Err(py_required_key_error(
                        "pad_token_id",
                        "sequence generation label"));
                };
                Label::SequenceGeneration(labels.extract()?, pad_token_id.extract()?)
            }
            k => {
                return Err(py_invalid_type_error(k, "label"));
            }
        };
        Ok(label)
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Item {
    #[pyo3(get)]
    data: TextData,
    #[pyo3(get)]
    tokenization: Tokenization,
    #[pyo3(get)]
    label: Label,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceDataFormat {
    Text,
    TextPlusDetections,
    TextPlusLanguage,
    TextPlusDetectionsPlusLanguage,
}

impl<'a> FromPyObject<'a> for InferenceDataFormat {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let format = match s.as_str() {
            "text" => InferenceDataFormat::Text,
            "text_detections" => InferenceDataFormat::TextPlusDetections,
            "text_language" => InferenceDataFormat::TextPlusLanguage,
            "text_detections_language" => InferenceDataFormat::TextPlusDetectionsPlusLanguage,
            k => return Err(py_invalid_type_error(k, "inference data format")),
        };
        Ok(format)
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[pyclass]
pub struct InferenceData {
    #[pyo3(get, set)]
    text: String,
    #[pyo3(get, set)]
    detections: Option<Vec<bool>>,
    #[pyo3(get, set)]
    language: Option<String>,
}

impl InferenceData {
    #[inline]
    fn parse_detections(str: &str) -> anyhow::Result<Vec<bool>> {
        str.split(char::is_whitespace)
            .map(|s| Ok(str::parse::<u8>(s.trim())? != 0))
            .collect::<anyhow::Result<Vec<bool>>>()
            .with_context(|| format!("failed to parse '{str}' to detections"))
    }

    #[inline]
    fn detection_str(&self) -> anyhow::Result<String> {
        if let Some(detections) = &self.detections {
            Ok(detections
                .iter()
                .map(|d| (*d as u8).to_string())
                .collect::<Vec<String>>()
                .join(" "))
        } else {
            Err(anyhow!("expected detections to be set, but got none"))
        }
    }

    pub fn from_str(s: &str, format: InferenceDataFormat) -> anyhow::Result<Self> {
        let splits: Vec<&str> = s.split('\t').collect();
        let (text, detections, language) = match format {
            InferenceDataFormat::Text => (s, None, None),
            InferenceDataFormat::TextPlusDetections => {
                if splits.len() < 2 {
                    return Err(anyhow!(
                        "expected at least 2 tab separated values in '{}', got {}",
                        s,
                        splits.len()
                    ));
                }
                (
                    &s[0..splits.len() - 2
                        + splits
                            .iter()
                            .take(splits.len() - 1)
                            .map(|s| s.len())
                            .sum::<usize>()],
                    Some(Self::parse_detections(splits[splits.len() - 1])?),
                    None,
                )
            }
            InferenceDataFormat::TextPlusLanguage => {
                let splits: Vec<&str> = s.split('\t').collect();
                if splits.len() < 2 {
                    return Err(anyhow!(
                        "expected at least 2 tab separated values in '{}', got {}",
                        s,
                        splits.len()
                    ));
                }
                (
                    &s[0..splits.len() - 2
                        + splits
                            .iter()
                            .take(splits.len() - 1)
                            .map(|s| s.len())
                            .sum::<usize>()],
                    None,
                    Some(splits[splits.len() - 1].trim().to_string()),
                )
            }
            InferenceDataFormat::TextPlusDetectionsPlusLanguage => {
                let splits: Vec<&str> = s.split('\t').collect();
                if splits.len() < 3 {
                    return Err(anyhow!(
                        "expected at least 3 tab separated values in '{}', got {}",
                        s,
                        splits.len()
                    ));
                }
                (
                    &s[0..splits.len() - 3
                        + splits
                            .iter()
                            .take(splits.len() - 2)
                            .map(|s| s.len())
                            .sum::<usize>()],
                    Some(Self::parse_detections(splits[splits.len() - 2])?),
                    Some(splits[splits.len() - 1].trim().to_string()),
                )
            }
        };
        Ok(Self::new(text.trim().to_string(), detections, language))
    }
}

#[pymethods]
impl InferenceData {
    #[new]
    pub fn new(s: String, detections: Option<Vec<bool>>, language: Option<String>) -> Self {
        Self {
            text: s,
            detections,
            language,
        }
    }
    #[staticmethod]
    #[pyo3(name = "from_str", signature = (s, format = InferenceDataFormat::Text))]
    pub fn from_str_py(s: String, format: InferenceDataFormat) -> anyhow::Result<Self> {
        Self::from_str(&s, format)
    }

    #[pyo3(signature = (format = InferenceDataFormat::Text))]
    pub fn to_str(&self, format: InferenceDataFormat) -> anyhow::Result<String> {
        let mut s = self.text.clone();
        if format == InferenceDataFormat::Text {
            return Ok(s);
        }
        if format == InferenceDataFormat::TextPlusDetections
            || format == InferenceDataFormat::TextPlusDetectionsPlusLanguage
        {
            s.push('\t');
            s.push_str(&self.detection_str()?);
        }
        s.push('\t');
        s.push_str(self.language.as_deref().unwrap_or(LANG_UNK));
        Ok(s)
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct InferenceItem {
    #[pyo3(get)]
    data: InferenceData,
    #[pyo3(get)]
    tokenization: Tokenization,
    #[pyo3(get)]
    item_idx: usize,
    #[pyo3(get)]
    window_idx: usize,
    #[pyo3(get)]
    window: (usize, usize, usize, usize),
    #[pyo3(get)]
    byte_window: (usize, usize, usize, usize),
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

    #[getter]
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
            Label::SequenceGeneration(label, ..) => {
                assert!(
                    label.len() > 1,
                    "sequence generation label must be at least two tokens long"
                );
                label.len() - 1
            }
        })
        .collect()
}

#[inline]
fn prepare_label_info(labels: &[&Label]) -> (TensorizedLabelInfo, Vec<usize>, usize) {
    let label_lengths = label_lengths(labels);
    let max_label_length = label_lengths.iter().max().copied().unwrap_or(0);
    let label_info = match labels[0] {
        Label::SequenceGeneration(..) => {
            let mut label_vec = Vec::new();
            for (idx, label) in labels.iter().enumerate() {
                if let Label::SequenceGeneration(label, pad_token_id) = label {
                    label_vec.extend(label.iter().cloned().take(label.len() - 1).chain(vec![
                                *pad_token_id as i32;
                                max_label_length - label_lengths[idx]
                            ]));
                } else {
                    unreachable!()
                }
            }
            let label_arr = Array2::from_shape_vec((labels.len(), max_label_length), label_vec)
                .expect("should not fail");
            TensorizedLabelInfo::SequenceGeneration((
                label_arr,
                padding_mask(label_lengths.as_slice()).expect("failed to create padding mask"),
                label_lengths.clone(),
            ))
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
                Label::SequenceGeneration(labels, ..) => labels
                    .iter()
                    .cloned()
                    .skip(1)
                    .chain(vec![-1; max_label_length - label_lengths[idx]])
                    .collect(),
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

    #[getter]
    fn items(&mut self) -> anyhow::Result<Batch<InferenceItem>> {
        if self.batch.is_none() {
            return Err(anyhow!(
                "can only get items once, because their data is moved"
            ));
        }
        Ok(self.batch.take().unwrap())
    }

    fn tensors(&mut self, py: Python<'_>) -> anyhow::Result<PyInferenceTensors> {
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
    Single(Vec<PreprocessingFnConfig>),
    PerFile(Vec<Vec<PreprocessingFnConfig>>),
}

impl<'a> FromPyObject<'a> for PreprocessingConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        if let Ok(value) = obj.extract::<Vec<PreprocessingFnConfig>>() {
            Ok(PreprocessingConfig::Single(value))
        } else if let Ok(value) = obj.extract::<Vec<Vec<PreprocessingFnConfig>>>() {
            Ok(PreprocessingConfig::PerFile(value))
        } else {
            Err(PyTypeError::new_err(
                "preprocessing config must be a list of preprocessing functions or a list of lists of preprocessing functions",
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessingPipelineConfig {
    pub preprocessing: PreprocessingConfig,
    pub labeling: LabelingConfig,
}

impl<'a> FromPyObject<'a> for PreprocessingPipelineConfig {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = obj.extract()?;
        let Some(preprocessing) = d.get_item("preprocessing") else {
            return Err(py_required_key_error("preprocessing", "preprocessing pipeline config"));
        };
        let Some(labeling) = d.get_item("labeling") else {
            return Err(py_required_key_error("labeling", "preprocessing pipeline config"));
        };
        Ok(PreprocessingPipelineConfig {
            preprocessing: preprocessing.extract()?,
            labeling: labeling.extract()?,
        })
    }
}

// a pipeline is a function mapping an input to an output,
// and it also sharable across threads
pub type Pipeline<I, O> = Arc<dyn Send + Sync + 'static + Fn(I) -> O>;

type TextDataFn = dyn Fn(TextData, usize, u64) -> anyhow::Result<TextData> + Send + Sync + 'static;
pub struct TextDataInfo {
    pub seed: u64,
    pub file_idx: usize,
}
pub type TextDataPipeline = Pipeline<(TextData, TextDataInfo), anyhow::Result<Item>>;
pub fn text_data_pipeline_with_tokenizer(
    pipeline_cfg: PreprocessingPipelineConfig,
    tokenizer_cfg: TokenizerConfig,
) -> anyhow::Result<TextDataPipeline> {
    let tok = tokenizer(tokenizer_cfg)?;
    let preprocess_fn: Box<TextDataFn> =
        match pipeline_cfg.preprocessing {
            PreprocessingConfig::Single(cfg) => {
                let preprocessing = preprocessing(cfg);
                Box::new(move |data: TextData, _: usize, seed: u64| preprocessing(data, Some(seed)))
            }
            PreprocessingConfig::PerFile(cfgs) => {
                let preprocessings: HashMap<_, _> = HashMap::from_iter(
                    cfgs.into_iter()
                        .enumerate()
                        .map(|(idx, cfg)| (idx, preprocessing(cfg))),
                );
                Box::new(move |data, file_idx, seed| {
                    preprocessings.get(&file_idx).unwrap_or_else(|| {
                        panic!("could not find preprocessing for file {file_idx}")
                    })(data, Some(seed))
                })
            }
        };
    let label_fn = labeling(pipeline_cfg.labeling);
    Ok(Arc::new(move |(data, info)| -> anyhow::Result<Item> {
        let data = preprocess_fn(data, info.file_idx, info.seed)?;
        Ok(Item {
            tokenization: tok.tokenize(
                &data.processed,
                data.language.as_deref(),
                None,
                None,
                false,
            )?,
            label: label_fn(&data)?,
            data,
        })
    }))
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
    use_graphemes: bool,
) -> anyhow::Result<InferencePipeline> {
    let tok = tokenizer(tokenizer_cfg)?;
    Ok(Arc::new(move |(data, info)| {
        let mut data = InferenceData {
            text: clean(&data.text, use_graphemes),
            ..data
        };
        if let Some(normalization) = normalization {
            data.text = normalize(&data.text, normalization, use_graphemes);
        }
        windows(&data.text, &window_cfg)?
            .iter()
            .enumerate()
            .map(|(w_idx, w)| {
                let tokenization =
                    tok.tokenize(w.str, data.language.as_deref(), None, None, true)?;
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
        normalization = Normalization::NFKC,
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
        let gen = inference_data_generator_from_python(iterator);
        let mut values = vec![];
        for value in gen {
            values.push(Ok(value?));
        }
        Self::new(
            vec![Box::new(values.into_iter())],
            tokenizer_config,
            window_config,
            normalization,
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
        file_format = InferenceDataFormat::Text,
        normalization = Normalization::NFKC,
        use_graphemes = true,
        languages = None,
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
        file_format: InferenceDataFormat,
        normalization: Option<Normalization>,
        use_graphemes: bool,
        languages: Option<Vec<String>>,
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
        if languages.is_some() && files.len() != languages.as_ref().unwrap().len() {
            return Err(anyhow!(
                "there must be one language for every file if specified, but \
                    got {} files and {} languages",
                files.len(),
                languages.as_ref().unwrap().len()
            ));
        }
        let mut generators = vec![];
        for (idx, file) in files.iter().enumerate() {
            let lang = if languages.is_some() {
                Some(languages.as_ref().unwrap()[idx].clone())
            } else {
                None
            };
            let generator = inference_data_generator_from_file(Path::new(file), file_format, lang)?;
            generators.push(generator);
        }
        Self::new(
            generators,
            tokenizer_config,
            window_config,
            normalization,
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
        pipeline_config: PreprocessingPipelineConfig,
        tokenizer_config: TokenizerConfig,
        strategy: TextIterationStrategy,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
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
        let pipeline =
            text_data_pipeline_with_tokenizer(pipeline_config, tokenizer_config.clone())?;
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
        for (idx, (original_file, processed_file)) in self.files.iter().enumerate() {
            let lang = if self.languages.is_some() {
                Some(self.languages.as_ref().unwrap()[idx].clone())
            } else {
                None
            };
            let generator =
                text_data_generator_from_files(original_file, processed_file.as_ref(), lang)?;
            generators.push(generator);
        }

        let text_iter = TextIterator::new(generators, self.strategy, Some(seed))?;
        self.min_items = Some(
            text_iter
                .min_len()
                .min(self.limit)
                .saturating_sub(self.skip)
                / self.world_size,
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
        shuffle = false,
        prefetch_factor = 4,
        sort = false,
        seed = None,
        skip = 0,
        limit = None,
        distributed = None
    ))]
    pub fn from_files(
        files: Vec<(String, Option<String>)>,
        pipeline_config: PreprocessingPipelineConfig,
        tokenizer_config: TokenizerConfig,
        languages: Option<Vec<String>>,
        strategy: TextIterationStrategy,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
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

    fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    fn set_fast_forward(&mut self, num_items: usize) {
        self.fast_forward = num_items
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
