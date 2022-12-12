use crate::data::loading::{
    BatchLimitType, BatchedIterator, PipelineIterator, TextContainer, TextFile, TextGen,
    TextGenerator, TextIterationStrategy,
};
use crate::data::preprocessing::{
    labeling, preprocessing, LabelingConfig, LabelingFn, PreprocessingConfig, PreprocessingFn,
};
use crate::tokenization::{tokenizer, Tokenization, TokenizerConfig, LANG_UNK};
use crate::utils::{py_invalid_type_error, py_required_key_error};
use crate::windows::{windows, WindowConfig};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::vec::IntoIter;

use self::loading::ItemSize;
use self::preprocessing::ItemFn;

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
    language: String,
}

impl TextData {
    pub fn new(original: String, processed: Option<String>, language: Option<String>) -> Self {
        let processed = processed.unwrap_or(original.clone());
        let language = language.unwrap_or(LANG_UNK.to_string());
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
    label: Option<Label>,
}
impl ItemSize for Item {
    fn size(&self) -> usize {
        self.tokenization.token_ids.len()
    }
}

impl Item {
    pub fn new(data: TextData, tokenization: Tokenization, label: Option<Label>) -> Self {
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

#[derive(Clone, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
#[pyclass]
pub struct InferenceItem {
    #[pyo3(get)]
    item: Item,
    #[pyo3(get)]
    item_idx: usize,
    #[pyo3(get)]
    window_idx: usize,
    #[pyo3(get)]
    window: (usize, usize, usize, usize),
}

impl InferenceItem {
    pub fn new(
        item: Item,
        item_idx: usize,
        window_idx: usize,
        window: (usize, usize, usize, usize),
    ) -> Self {
        InferenceItem {
            item,
            item_idx,
            window_idx,
            window,
        }
    }
}

impl ItemSize for InferenceItem {
    fn size(&self) -> usize {
        self.item.size()
    }
}

#[pymethods]
impl InferenceItem {
    fn __len__(&self) -> usize {
        self.item.size()
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

#[derive(Debug)]
#[pyclass]
pub struct ItemBatch {
    batch: Batch<Item>,
}

#[pymethods]
impl ItemBatch {
    fn __len__(&self) -> usize {
        self.batch.len()
    }
}
#[derive(Debug)]
#[pyclass]
pub struct InferenceItemBatch {
    batch: Batch<InferenceItem>,
}

#[pymethods]
impl InferenceItemBatch {
    fn __len__(&self) -> usize {
        self.batch.len()
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
pub struct PipelineConfig {
    #[pyo3(get)]
    preprocessing: Vec<PreprocessingConfig>,
    #[pyo3(get)]
    labeling: Option<LabelingConfig>,
}

impl PipelineConfig {
    pub fn new(preprocessing: Vec<PreprocessingConfig>, labeling: Option<LabelingConfig>) -> Self {
        PipelineConfig {
            preprocessing,
            labeling,
        }
    }
}

#[pymethods]
impl PipelineConfig {
    #[new]
    #[args(labeling = "None")]
    fn py_new(
        preprocessing: Vec<PreprocessingConfig>,
        labeling: Option<LabelingConfig>,
    ) -> PyResult<Self> {
        Ok(Self::new(preprocessing, labeling))
    }
}

#[derive(Clone)]
pub struct Pipeline<T> {
    preprocessing_fn: Arc<PreprocessingFn>,
    label_fn: Option<Arc<LabelingFn>>,
    item_fn: Arc<ItemFn<T, Arc<LabelingFn>>>,
}

impl<T> Pipeline<T> {
    pub fn apply(&self, item: TextData, idx: usize, seed: Option<u64>) -> T {
        let data = (self.preprocessing_fn)(item, seed);
        (self.item_fn)(data, idx, self.label_fn.as_ref())
    }

    fn from_config_and_item_fn(
        cfg: PipelineConfig,
        item_fn: Arc<ItemFn<T, Arc<LabelingFn>>>,
    ) -> Self {
        Pipeline {
            preprocessing_fn: Arc::new(preprocessing(cfg.preprocessing)),
            label_fn: if cfg.labeling.is_some() {
                Some(Arc::new(labeling(cfg.labeling.unwrap())))
            } else {
                None
            },
            item_fn,
        }
    }
}

impl Pipeline<Item> {
    pub fn with_tokenizer(pipeline_cfg: PipelineConfig, tokenizer_cfg: TokenizerConfig) -> Self {
        let tok = tokenizer(tokenizer_cfg);
        Pipeline::from_config_and_item_fn(
            pipeline_cfg,
            Arc::new(move |data, _, label_fn| -> Item {
                Item {
                    tokenization: tok.tokenize(&data.processed, None, None),
                    label: if label_fn.is_some() {
                        Some(label_fn.unwrap()(&data))
                    } else {
                        None
                    },
                    data,
                }
            }),
        )
    }
}

pub type InferencePipeline = Pipeline<Vec<InferenceItem>>;
impl InferencePipeline {
    pub fn with_windows(
        pipeline_cfg: PipelineConfig,
        tokenizer_cfg: TokenizerConfig,
        window_cfg: WindowConfig,
    ) -> Self {
        let tok = tokenizer(tokenizer_cfg);
        Pipeline::from_config_and_item_fn(
            pipeline_cfg,
            Arc::new(move |data, idx, label_fn| {
                windows(&data.processed, &window_cfg)
                    .iter()
                    .enumerate()
                    .map(|(w_idx, w)| {
                        let tokenization = tok.tokenize(w.str, None, None);
                        let boundaries = w.boundaries();
                        let item = Item {
                            label: if label_fn.is_some() {
                                Some(label_fn.unwrap()(&data))
                            } else {
                                None
                            },
                            data: TextData::new(
                                w.str.to_string(),
                                None,
                                Some(data.language.clone()),
                            ),
                            tokenization,
                        };
                        InferenceItem::new(item, idx, w_idx, boundaries)
                    })
                    .collect()
            }),
        )
    }
}

#[pyclass]
struct DataLoader {
    pipeline: Pipeline<Item>,
    text_gen: TextGenerator,
    strategy: TextIterationStrategy,
    num_threads: u8,
    buffer_size: usize,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    epoch: usize,
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

#[pyclass]
struct InferenceLoader {
    pipeline: Pipeline<Vec<InferenceItem>>,
    text_gen: TextGenerator,
    num_threads: u8,
    buffer_size: usize,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    prefetch_factor: usize,
    sort: bool,
    #[pyo3(get)]
    min_items: Option<usize>,
    iter: Option<Box<dyn Iterator<Item = Batch<InferenceItem>> + Send>>,
    splits: Vec<usize>,
}

impl DataLoader {
    fn new(
        generators: Vec<Box<dyn TextGen>>,
        pipeline_config: PipelineConfig,
        tokenizer_config: TokenizerConfig,
        strategy: TextIterationStrategy,
        num_threads: u8,
        mut buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
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
        let text_gen = TextGenerator::new(generators);
        let limit = limit.unwrap_or(usize::MAX);
        Ok(DataLoader {
            pipeline,
            text_gen,
            strategy,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            iter: None,
            min_items: None,
            epoch: 0,
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

impl InferenceLoader {
    pub fn new(
        generators: Vec<Box<dyn TextGen>>,
        pipeline_config: PipelineConfig,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        num_threads: u8,
        mut buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> PyResult<Self> {
        let pipeline = Pipeline::with_windows(pipeline_config, tokenizer_config, window_config);
        let splits = generators.iter().map(|g| g.min_len()).collect();
        let text_gen = TextGenerator::new(generators);
        let prefetch_factor = prefetch_factor.max(1);
        if batch_limit_type == BatchLimitType::BatchSize {
            buffer_size = buffer_size.max(batch_limit * prefetch_factor);
        }
        Ok(InferenceLoader {
            pipeline,
            text_gen,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            iter: None,
            min_items: None,
            prefetch_factor,
            sort,
            splits,
        })
    }
}

#[pymethods]
impl InferenceLoader {
    #[staticmethod]
    #[args(
        languages = "None",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "32",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        prefetch_factor = "4",
        sort = "false"
    )]
    pub fn from_sequences(
        sequences: Vec<String>,
        pipeline_config: PipelineConfig,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        languages: Option<Vec<String>>,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
    ) -> PyResult<Self> {
        if sequences.is_empty() {
            return Err(PyTypeError::new_err("sequences is empty"));
        }
        if languages.is_some() && sequences.len() == languages.as_ref().unwrap().len() {
            return Err(PyTypeError::new_err(format!(
                "there must be one language for every sequence if specified, but \
                    got {} sequences and {} languages",
                sequences.len(),
                languages.as_ref().unwrap().len()
            )));
        }
        let generators = TextContainer::new_boxed(sequences, None, languages);
        Self::new(
            vec![generators],
            pipeline_config,
            tokenizer_config,
            window_config,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            prefetch_factor,
            sort,
        )
    }

    #[staticmethod]
    #[args(
        languages = "None",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "32",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        prefetch_factor = "4",
        sort = "false"
    )]
    pub fn from_files(
        files: Vec<String>,
        pipeline_config: PipelineConfig,
        tokenizer_config: TokenizerConfig,
        window_config: WindowConfig,
        languages: Option<Vec<String>>,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
        sort: bool,
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
        let generators = files
            .into_iter()
            .enumerate()
            .map(|(idx, file)| {
                let lang = if languages.is_some() {
                    Some(languages.as_ref().unwrap()[idx].clone())
                } else {
                    None
                };
                TextFile::new_boxed(&file.into(), None, lang) as Box<dyn TextGen>
            })
            .collect();
        Self::new(
            generators,
            pipeline_config,
            tokenizer_config,
            window_config,
            num_threads,
            buffer_size,
            batch_limit,
            batch_limit_type,
            prefetch_factor,
            sort,
        )
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let text_iter = slf.text_gen.sequential();
        let batch_iter = text_iter
            .pipe(&slf.pipeline, slf.num_threads, slf.buffer_size, None)
            .flatten()
            .batched(
                slf.sort,
                false,
                slf.prefetch_factor,
                slf.batch_limit,
                slf.batch_limit_type,
                None,
            );
        slf.iter = Some(Box::new(batch_iter));
        slf
    }

    fn __next__(&mut self) -> Option<Py<InferenceItemBatch>> {
        assert!(
            self.iter.is_some(),
            "call iter() on the inferenceloader before iterating with next()"
        );
        if let Some(batch) = self.iter.as_mut().unwrap().next() {
            Some(Python::with_gil(|py| {
                let item_batch = InferenceItemBatch { batch };
                Py::new(py, item_batch).expect("should not fail")
            }))
        } else {
            None
        }
    }

    pub fn splits(&self) -> Vec<usize> {
        self.splits.clone()
    }
}

#[pymethods]
impl DataLoader {
    #[staticmethod]
    #[args(
        languages = "None",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "32",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
        shuffle = "false",
        prefetch_factor = "4",
        sort = "false",
        seed = "None",
        skip = "0",
        limit = "None",
        distributed = "None"
    )]
    pub fn from_sequences(
        sequences: Vec<String>,
        pipeline_config: PipelineConfig,
        tokenizer_config: TokenizerConfig,
        languages: Option<Vec<String>>,
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
    ) -> PyResult<Self> {
        if sequences.is_empty() {
            return Err(PyTypeError::new_err("sequences is empty"));
        }
        if languages.is_some() && sequences.len() == languages.as_ref().unwrap().len() {
            return Err(PyTypeError::new_err(format!(
                "there must be one language for every sequence if specified, but \
                    got {} sequences and {} languages",
                sequences.len(),
                languages.as_ref().unwrap().len()
            )));
        }
        let generators = TextContainer::new_boxed(sequences, None, languages);
        Self::new(
            vec![generators],
            pipeline_config,
            tokenizer_config,
            TextIterationStrategy::Sequential,
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

    #[staticmethod]
    #[args(
        languages = "None",
        strategy = "TextIterationStrategy::Sequential",
        num_threads = "(num_cpus::get() as u8).min(4)",
        buffer_size = "32",
        batch_limit = "16",
        batch_limit_type = "BatchLimitType::BatchSize",
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
        pipeline_config: PipelineConfig,
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
        let generators = files
            .into_iter()
            .enumerate()
            .map(|(idx, file)| {
                let lang = if languages.is_some() {
                    Some(languages.as_ref().unwrap()[idx].clone())
                } else {
                    None
                };
                TextFile::new_boxed(&file.into(), None, lang) as Box<dyn TextGen>
            })
            .collect();
        Self::new(
            generators,
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

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let seed = if slf.seed.is_some() {
            Some(slf.seed.unwrap() + slf.epoch as u64)
        } else {
            None
        };
        let text_iter = slf.text_gen.with_strategy(slf.strategy, seed);
        slf.min_items =
            Some(text_iter.min_len().min(slf.limit).saturating_sub(slf.skip) / slf.world_size);
        let batch_iter = text_iter
            .take(slf.limit)
            .skip(slf.skip + slf.rank)
            .step_by(slf.world_size)
            .pipe(&slf.pipeline, slf.num_threads, slf.buffer_size, seed)
            .batched(
                slf.sort,
                slf.shuffle,
                slf.prefetch_factor,
                slf.batch_limit,
                slf.batch_limit_type,
                seed,
            );
        slf.iter = Some(Box::new(batch_iter));
        slf
    }

    fn __next__(&mut self) -> Option<Py<ItemBatch>> {
        assert!(
            self.iter.is_some(),
            "call iter() on the dataloader before iterating with next()"
        );
        if let Some(batch) = self.iter.as_mut().unwrap().next() {
            Some(Python::with_gil(|py| {
                let item_batch = ItemBatch { batch };
                Py::new(py, item_batch).expect("should not fail")
            }))
        } else {
            None
        }
    }

    fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
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
    m.add_class::<PipelineConfig>()?;
    m.add_class::<TextData>()?;
    m.add_class::<Item>()?;
    m.add_class::<ItemBatch>()?;
    m.add_class::<InferenceItemBatch>()?;
    m.add_class::<TextIterationStrategy>()?;
    m.add_class::<BatchLimitType>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
