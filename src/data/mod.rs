use crate::data::loading::{
    BatchLimitType, BatchedIterator, DataGen, PipelineIterator, TextIterationStrategy,
};
use crate::data::preprocessing::{labeling, preprocessing, LabelingConfig, PreprocessingConfig};
use crate::text::clean;
use crate::tokenization::{tokenizer, Tokenization, TokenizerConfig};
use crate::unicode::{normalize, Normalization};
use crate::utils::{py_invalid_type_error, py_required_key_error};
use crate::windows::{windows, WindowConfig};
use anyhow::anyhow;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::vec::IntoIter;

use self::loading::{
    inference_data_generator_from_file, inference_data_generator_from_python,
    text_data_generator_from_files, ItemSize, TextIterator,
};

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
        let processed = processed.unwrap_or(original.clone());
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
    #[args(processed = "None", language = "None")]
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

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Label {
    Classification(usize),
    SeqClassification(Vec<usize>),
    Seq2Seq(Vec<usize>),
}

impl IntoPy<PyObject> for Label {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let label_type = match self {
            Label::Classification(label) => {
                d.set_item("label", label).unwrap();
                "classification"
            }
            Label::SeqClassification(labels) => {
                d.set_item("labels", labels).unwrap();
                "sequence_classification"
            }
            Label::Seq2Seq(labels) => {
                d.set_item("labels", labels).unwrap();
                "seq2seq"
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
                Label::SeqClassification(labels.extract()?)
            }
            "seq2seq" => {
                let Some(labels) = d.get_item("labels") else {
                    return Err(py_required_key_error(
                        "labels",
                        "seq2seq label",
                    ));
                };
                Label::Seq2Seq(labels.extract()?)
            }
            k => {
                return Err(py_invalid_type_error(k, "label"));
            }
        };
        Ok(label)
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
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
        self.tokenization.token_ids.len()
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

#[derive(Clone, Copy, Debug)]
pub enum InferenceDataFileFormat {
    Text,
    TextPlusDetections,
    TextPlusLanguage,
    TextPlusDetectionsPlusLanguage,
}

impl<'a> FromPyObject<'a> for InferenceDataFileFormat {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let format = match s.as_str() {
            "text" => InferenceDataFileFormat::Text,
            "text_detections" => InferenceDataFileFormat::TextPlusDetections,
            "text_language" => InferenceDataFileFormat::TextPlusLanguage,
            "text_detections_language" => InferenceDataFileFormat::TextPlusDetectionsPlusLanguage,
            k => return Err(py_invalid_type_error(k, "inference data file format")),
        };
        Ok(format)
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[pyclass]
pub struct InferenceData {
    #[pyo3(get)]
    original: String,
    #[pyo3(get)]
    detections: Option<Vec<bool>>,
    #[pyo3(get)]
    language: Option<String>,
}

impl InferenceData {
    pub fn new(original: String, detections: Option<Vec<bool>>, language: Option<String>) -> Self {
        Self {
            original,
            detections,
            language,
        }
    }

    fn parse_detections(str: &str) -> Vec<bool> {
        str.split(char::is_whitespace)
            .map(|s| {
                str::parse::<u8>(s.trim())
                    .expect(format!("failed to parse {s} to integer").as_str())
                    != 0
            })
            .collect()
    }

    pub fn from_str(s: &str, format: &InferenceDataFileFormat) -> Self {
        let (original, detections, language) = match format {
            InferenceDataFileFormat::Text => (s, None, None),
            InferenceDataFileFormat::TextPlusDetections => {
                let splits: Vec<&str> = s.split("\t").collect();
                assert_eq!(splits.len(), 2);
                (splits[0], Some(Self::parse_detections(splits[1])), None)
            }
            InferenceDataFileFormat::TextPlusLanguage => {
                let splits: Vec<&str> = s.split("\t").collect();
                assert_eq!(splits.len(), 2);
                (splits[0], None, Some(splits[1].trim().to_string()))
            }
            InferenceDataFileFormat::TextPlusDetectionsPlusLanguage => {
                let splits: Vec<&str> = s.split("\t").collect();
                assert_eq!(splits.len(), 3);
                (
                    splits[0],
                    Some(Self::parse_detections(splits[1])),
                    Some(splits[2].trim().to_string()),
                )
            }
        };
        Self::new(original.trim().to_string(), detections, language)
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
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
}

impl InferenceItem {
    pub fn new(
        data: InferenceData,
        tokenization: Tokenization,
        item_idx: usize,
        window_idx: usize,
        window: (usize, usize, usize, usize),
    ) -> Self {
        InferenceItem {
            data,
            tokenization,
            item_idx,
            window_idx,
            window,
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

    fn __hash__(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        op.matches(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct Batch<T> {
    items: Vec<T>,
}

impl<T> Batch<T> {
    pub fn new(items: Vec<T>) -> Self {
        Batch { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

#[pyclass]
pub struct ItemBatch {
    batch: Batch<Item>,
    iter: Option<Box<dyn Iterator<Item = Item> + Send>>,
}

#[pymethods]
impl ItemBatch {
    fn __len__(&self) -> usize {
        self.batch.len()
    }

    #[getter]
    fn items(&self) -> Vec<Item> {
        self.batch.items.clone()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.iter = Some(Box::new(slf.batch.items.clone().into_iter()));
        slf
    }

    fn __next__(&mut self) -> Option<Py<Item>> {
        if let Some(item) = self.iter.as_mut().unwrap().next() {
            Some(Python::with_gil(|py| {
                Py::new(py, item).expect("should not fail")
            }))
        } else {
            None
        }
    }
}

#[pyclass]
pub struct InferenceItemBatch {
    batch: Batch<InferenceItem>,
    iter: Option<Box<dyn Iterator<Item = InferenceItem> + Send>>,
}

#[pymethods]
impl InferenceItemBatch {
    fn __len__(&self) -> usize {
        self.batch.len()
    }

    #[getter]
    fn items(&self) -> Vec<InferenceItem> {
        self.batch.items.clone()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.iter = Some(Box::new(slf.batch.items.clone().into_iter()));
        slf
    }

    fn __next__(&mut self) -> Option<Py<InferenceItem>> {
        if let Some(item) = self.iter.as_mut().unwrap().next() {
            Some(Python::with_gil(|py| {
                Py::new(py, item).expect("should not fail")
            }))
        } else {
            None
        }
    }
}

impl<T> IntoIterator for Batch<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PreprocessingPipelineConfig {
    #[pyo3(get)]
    preprocessing: Vec<PreprocessingConfig>,
    #[pyo3(get)]
    labeling: LabelingConfig,
}

impl PreprocessingPipelineConfig {
    pub fn new(preprocessing: Vec<PreprocessingConfig>, labeling: LabelingConfig) -> Self {
        PreprocessingPipelineConfig {
            preprocessing,
            labeling,
        }
    }
}

#[pymethods]
impl PreprocessingPipelineConfig {
    #[new]
    fn py_new(preprocessing: Vec<PreprocessingConfig>, labeling: LabelingConfig) -> PyResult<Self> {
        Ok(Self::new(preprocessing, labeling))
    }
}

pub type ApplyFn<I, O> = dyn Send + Sync + 'static + Fn(I, usize, Option<u64>) -> O;
pub struct Pipeline<I, O> {
    apply_fn: Arc<ApplyFn<I, O>>,
}
impl<I, O> Clone for Pipeline<I, O> {
    fn clone(&self) -> Self {
        Self {
            apply_fn: self.apply_fn.clone(),
        }
    }
}

impl<I, O> Pipeline<I, O> {
    pub fn apply(&self, input: I, idx: usize, seed: Option<u64>) -> O {
        (self.apply_fn)(input, idx, seed)
    }

    pub fn new(apply_fn: Arc<ApplyFn<I, O>>) -> Self {
        Self { apply_fn }
    }
}

pub type TextDataPipeline = Pipeline<TextData, anyhow::Result<Item>>;
impl TextDataPipeline {
    pub fn with_tokenizer(
        pipeline_cfg: PreprocessingPipelineConfig,
        tokenizer_cfg: TokenizerConfig,
    ) -> Self {
        let tok = tokenizer(tokenizer_cfg);
        let preprocess_fn = preprocessing(pipeline_cfg.preprocessing);
        let label_fn = labeling(pipeline_cfg.labeling);
        Pipeline::new(Arc::new(move |data, _, seed| -> anyhow::Result<Item> {
            let data = preprocess_fn(data, seed)?;
            Ok(Item {
                tokenization: tok.tokenize(&data.processed, None, None),
                label: label_fn(&data)?,
                data,
            })
        }))
    }
}

pub type InferencePipeline = Pipeline<InferenceData, anyhow::Result<Vec<InferenceItem>>>;
impl InferencePipeline {
    pub fn with_windows(
        tokenizer_cfg: TokenizerConfig,
        window_cfg: WindowConfig,
        normalization: Option<Normalization>,
        use_graphemes: bool,
    ) -> Self {
        let tok = tokenizer(tokenizer_cfg);
        Pipeline::new(Arc::new(move |data, idx, _| {
            let mut data = InferenceData {
                original: clean(&data.original, use_graphemes),
                ..data
            };
            if normalization.is_some() {
                data.original = normalize(&data.original, normalization.unwrap(), use_graphemes);
            }
            Ok(windows(&data.original, &window_cfg)?
                .iter()
                .enumerate()
                .map(|(w_idx, w)| {
                    let tokenization = tok.tokenize(w.str, None, None);
                    let boundaries = w.boundaries();
                    InferenceItem::new(data.clone(), tokenization, idx, w_idx, boundaries)
                })
                .collect())
        }))
    }
}

#[pyclass]
struct InferenceLoader {
    iter: Box<dyn Iterator<Item = Batch<InferenceItem>> + Send>,
    iter_err: Arc<Mutex<Option<anyhow::Error>>>,
    #[pyo3(get)]
    min_items: usize,
    #[pyo3(get)]
    splits: Vec<usize>,
}

impl InferenceLoader {
    pub fn new(
        generators: Vec<Box<dyn DataGen<Item = anyhow::Result<InferenceData>>>>,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        normalization: Option<Normalization>,
        use_graphemes: bool,
        num_threads: u8,
        mut buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        batch_buffer_size: usize,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        let pipeline = InferencePipeline::with_windows(
            tokenizer_config,
            window_config,
            normalization,
            use_graphemes,
        );
        let splits: Vec<usize> = generators.iter().map(|g| g.min_len()).collect();
        let min_items = splits.iter().sum();
        let prefetch_factor = prefetch_factor.max(1);
        if batch_limit_type == BatchLimitType::BatchSize {
            buffer_size = buffer_size.max(batch_limit * prefetch_factor);
        }
        let text_iter = TextIterator::new(generators, TextIterationStrategy::Sequential, None)?;
        let iter_err = Arc::new(Mutex::new(None));
        let text_iter_err = iter_err.clone();
        let pipe_iter_err = iter_err.clone();
        let iter = text_iter
            .scan((), move |_, d| {
                if d.is_err() {
                    *text_iter_err.lock().unwrap() = Some(d.unwrap_err());
                    None
                } else {
                    d.ok()
                }
            })
            .pipe(&pipeline, num_threads, buffer_size, None)
            .scan((), move |_, i| {
                if i.is_err() {
                    *pipe_iter_err.lock().unwrap() = Some(i.unwrap_err());
                    None
                } else {
                    i.ok()
                }
            })
            .flatten()
            .batched(
                sort,
                false,
                prefetch_factor,
                batch_limit,
                batch_limit_type,
                batch_buffer_size,
                None,
            );
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
    #[staticmethod]
    #[args(
        normalization = "Normalization::NFKC",
        use_graphemes = "true",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "128",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        batch_buffer_size = "8",
        prefetch_factor = "4",
        sort = "false"
    )]
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
        batch_buffer_size: usize,
        prefetch_factor: usize,
        sort: bool,
    ) -> anyhow::Result<Self> {
        let text_gen = Box::new(inference_data_generator_from_python(iterator));
        Self::new(
            vec![text_gen],
            tokenizer_config,
            window_config,
            normalization,
            use_graphemes,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            batch_buffer_size,
            prefetch_factor,
            sort,
        )
    }

    #[staticmethod]
    #[args(
        file_format = "InferenceDataFileFormat::Text",
        normalization = "Normalization::NFKC",
        use_graphemes = "true",
        languages = "None",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "128",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        batch_buffer_size = "8",
        prefetch_factor = "4",
        sort = "false"
    )]
    pub fn from_files(
        files: Vec<String>,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        file_format: InferenceDataFileFormat,
        normalization: Option<Normalization>,
        use_graphemes: bool,
        languages: Option<Vec<String>>,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        batch_buffer_size: usize,
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
            batch_buffer_size,
            prefetch_factor,
            sort,
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> anyhow::Result<Option<Py<InferenceItemBatch>>> {
        if let Some(batch) = self.iter.next() {
            Ok(Some(Python::with_gil(|py| {
                let item_batch = InferenceItemBatch { batch, iter: None };
                Py::new(py, item_batch).expect("should not fail")
            })))
        } else {
            // check if batch is None because iterator is stopped,
            // or because an error was encountered
            match self.iter_err.lock().unwrap().as_ref() {
                Some(e) => Err(anyhow!("error during inference iterator: {e}")),
                None => Ok(None),
            }
        }
    }
}

#[pyclass]
struct DataLoader {
    pipeline: TextDataPipeline,
    files: Vec<String>,
    languages: Option<Vec<String>>,
    strategy: TextIterationStrategy,
    num_threads: u8,
    buffer_size: usize,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    batch_buffer_size: usize,
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
    iter: Option<Box<dyn Iterator<Item = Batch<Item>> + Send>>,
}

impl DataLoader {
    fn new(
        files: Vec<String>,
        languages: Option<Vec<String>>,
        pipeline_config: PreprocessingPipelineConfig,
        tokenizer_config: TokenizerConfig,
        strategy: TextIterationStrategy,
        num_threads: u8,
        mut buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        batch_buffer_size: usize,
        shuffle: bool,
        prefetch_factor: usize,
        sort: bool,
        seed: Option<u64>,
        skip: usize,
        limit: Option<usize>,
        distributed: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        if shuffle && seed.is_none() {
            return Err(PyTypeError::new_err(
                "seed cannot be None if shuffle is true",
            ));
        }
        let prefetch_factor = prefetch_factor.max(1);
        if batch_limit_type == BatchLimitType::BatchSize {
            buffer_size = buffer_size.max(batch_limit * prefetch_factor);
        }
        let pipeline = Pipeline::with_tokenizer(pipeline_config, tokenizer_config);
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
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            batch_buffer_size,
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
}

#[pymethods]
impl DataLoader {
    #[staticmethod]
    #[args(
        languages = "None",
        strategy = "TextIterationStrategy::Sequential",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "128",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        batch_buffer_size = "8",
        shuffle = "false",
        prefetch_factor = "4",
        sort = "false",
        seed = "None",
        skip = "0",
        limit = "None",
        distributed = "None"
    )]
    pub fn from_files(
        files: Vec<String>,
        pipeline_config: PreprocessingPipelineConfig,
        tokenizer_config: TokenizerConfig,
        languages: Option<Vec<String>>,
        strategy: TextIterationStrategy,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        batch_buffer_size: usize,
        shuffle: bool,
        prefetch_factor: usize,
        sort: bool,
        seed: Option<u64>,
        skip: usize,
        limit: Option<usize>,
        distributed: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        if files.is_empty() {
            return Err(PyTypeError::new_err("files is empty"));
        }
        if languages.is_some() && files.len() != languages.as_ref().unwrap().len() {
            return Err(PyTypeError::new_err(format!(
                "there must be one language for every file if specified, but \
                    got {} files and {} languages",
                files.len(),
                languages.as_ref().unwrap().len()
            )));
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
            batch_buffer_size,
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
        let seed = if slf.seed.is_some() {
            Some(slf.seed.unwrap() + slf.epoch as u64)
        } else {
            None
        };
        let mut generators = vec![];
        for (idx, file) in slf.files.iter().enumerate() {
            let lang = if slf.languages.is_some() {
                Some(slf.languages.as_ref().unwrap()[idx].clone())
            } else {
                None
            };
            let generator = text_data_generator_from_files(Path::new(file), None, lang)?;
            generators.push(generator);
        }

        let text_iter = TextIterator::new(generators, slf.strategy, seed)?;
        slf.min_items =
            Some(text_iter.min_len().min(slf.limit).saturating_sub(slf.skip) / slf.world_size);
        let batch_iter = text_iter
            .take(slf.limit)
            .skip(slf.skip + slf.fast_forward + slf.rank)
            .step_by(slf.world_size)
            .filter_map(|d| d.ok())
            .pipe(&slf.pipeline, slf.num_threads, slf.buffer_size, seed)
            .filter_map(|i| i.ok())
            .batched(
                slf.sort,
                slf.shuffle,
                slf.prefetch_factor,
                slf.batch_limit,
                slf.batch_limit_type,
                slf.batch_buffer_size,
                seed,
            );
        slf.iter = Some(Box::new(batch_iter));
        Ok(slf)
    }

    fn __next__(&mut self) -> Option<Py<ItemBatch>> {
        assert!(
            self.iter.is_some(),
            "call iter() on the dataloader before iterating with next()"
        );
        if let Some(batch) = self.iter.as_mut().unwrap().next() {
            Some(Python::with_gil(|py| {
                let item_batch = ItemBatch { batch, iter: None };
                Py::new(py, item_batch).expect("should not fail")
            }))
        } else {
            None
        }
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
/// - loading in memory lists of strings
/// - several loading strategies (sequential, interleaved, weighted)
/// - single or multi-threaded preprocessing
/// - batched loading (limited by a max batch size or a max number of tokens)
/// - distributed loading (distribute work across multiple processes or machines)
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "data")?;
    m.add_class::<DataLoader>()?;
    m.add_class::<InferenceLoader>()?;
    m.add_class::<PreprocessingPipelineConfig>()?;
    m.add_class::<TextData>()?;
    m.add_class::<InferenceData>()?;
    m.add_class::<Item>()?;
    m.add_class::<InferenceItem>()?;
    m.add_class::<ItemBatch>()?;
    m.add_class::<InferenceItemBatch>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
