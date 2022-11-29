use std::fs::read_to_string;
use std::path::{Path};
use std::vec::IntoIter;
use pyo3::exceptions::{PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict};
use serde::{Deserialize, Serialize};
use crate::data::loading::{BatchLimitType, PipelineIterator, TextContainer, TextFile, TextGen, TextGenerator, TextIterationStrategy};
use crate::tokenization::{LANG_UNK, Tokenization, tokenizer, Tokenizer, TokenizerConfig};
use crate::data::preprocessing::{labeling, LabelingConfig, LabelingFn, preprocessing, PreprocessingConfig, PreprocessingFn};
use crate::utils::{py_invalid_type_error, py_required_key_error};

pub mod preprocessing;
pub mod loading;

#[derive(Clone, Debug, PartialEq)]
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
    #[args(
    processed = "None",
    language = "None"
    )]
    fn new_py(
        original: String,
        processed: Option<String>,
        language: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self::new(original, processed, language))
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
#[pyclass]
pub struct Item {
    #[pyo3(get)]
    data: TextData,
    #[pyo3(get)]
    tokenization: Tokenization,
    #[pyo3(get)]
    label: Option<Label>,
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
    #[new]
    #[args(
    label = "None"
    )]
    fn py_new(data: TextData, tokenization: Tokenization, label: Option<Label>) -> PyResult<Self> {
        Ok(Self::new(data, tokenization, label))
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.tokenization.token_ids.len())
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct Batch {
    #[pyo3(get)]
    items: Vec<Item>,
}

impl Batch {
    pub fn new(items: Vec<Item>) -> Self {
        Batch { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

#[pymethods]
impl Batch {
    #[new]
    fn py_new(items: Vec<Item>) -> PyResult<Self> {
        Ok(Self::new(items))
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.len())
    }
}

impl IntoIterator for Batch {
    type Item = Item;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[pyclass]
pub struct PipelineConfig {
    #[pyo3(get)]
    preprocessing: Vec<PreprocessingConfig>,
    #[pyo3(get)]
    labeling: Option<LabelingConfig>,
    #[pyo3(get)]
    tokenizer: TokenizerConfig,
}

impl PipelineConfig {
    pub fn new(
        preprocessing: Vec<PreprocessingConfig>,
        labeling: Option<LabelingConfig>,
        tokenizer: TokenizerConfig,
    ) -> Self {
        PipelineConfig {
            preprocessing,
            labeling,
            tokenizer,
        }
    }
}

#[pymethods]
impl PipelineConfig {
    #[new]
    #[args(
    labeling = "None"
    )]
    fn py_new(
        preprocessing: Vec<PreprocessingConfig>,
        tokenizer: TokenizerConfig,
        labeling: Option<LabelingConfig>,
    ) -> PyResult<Self> {
        Ok(Self::new(preprocessing, labeling, tokenizer))
    }
}

pub struct Pipeline {
    // Preprocessing a FnMut so we have to wrap it here to be thread safe
    cfg: PipelineConfig,
    preprocessing_fn: PreprocessingFn,
    label_fn: Option<LabelingFn>,
    tokenizer: Tokenizer,
}

impl Clone for Pipeline {
    fn clone(&self) -> Self {
        Pipeline::from_config(self.cfg.clone())
    }
}

impl Pipeline {
    pub fn from_config(cfg: PipelineConfig) -> Self {
        Pipeline {
            cfg: cfg.clone(),
            preprocessing_fn: preprocessing(cfg.preprocessing),
            label_fn: if cfg.labeling.is_some() {
                Some(labeling(cfg.labeling.unwrap()))
            } else {
                None
            },
            tokenizer: tokenizer(cfg.tokenizer),
        }
    }

    pub fn apply(&self, item: TextData, seed: Option<u64>) -> Item {
        let data = (self.preprocessing_fn)(item, seed);
        let label = if self.label_fn.is_some() {
            Some((self.label_fn.as_ref().unwrap())(&data))
        } else {
            None
        };
        let tokenization = self.tokenizer.tokenize(
            &data.processed,
            None,
            None,
        );
        Item {
            data,
            label,
            tokenization,
        }
    }

    pub fn apply_iter(
        self,
        iter: impl Iterator<Item=TextData> + Send + 'static,
    ) -> PipelineIterator {
        PipelineIterator::new(iter, self.clone())
    }

    pub fn apply_iter_threaded(
        self,
        iter: impl Iterator<Item=TextData> + Send + 'static,
        worker_threads: u8,
        buffer_size: usize,
    ) -> PipelineIterator {
        PipelineIterator::new_threaded(
            iter,
            self.clone(),
            worker_threads,
            buffer_size,
        )
    }
}

fn read_yaml(path: &Path) -> String {
    read_to_string(path)
        .expect(&format!("could not read yaml file at {:?}", path))
}

fn parse_yaml<'a, T: Deserialize<'a>>(yaml: &'a str) -> T {
    serde_yaml::from_str(yaml)
        .expect(&format!("could not deserialize from yaml string\n{}", yaml))
}

pub fn pipeline_from_yaml(path: &Path) -> Pipeline {
    pipeline_from_str(&read_yaml(path))
}

pub fn pipeline_from_str(s: &str) -> Pipeline {
    let cfg: PipelineConfig = parse_yaml(s);
    Pipeline::from_config(cfg)
}

pub fn preprocessing_from_yaml(path: &Path) -> PreprocessingFn {
    preprocessing_from_str(&read_yaml(path))
}

pub fn preprocessing_from_str(s: &str) -> PreprocessingFn {
    let fns: Vec<PreprocessingConfig> = serde_yaml::from_str(s)
        .expect(&format!("could not deserialize from yaml string\n{}", s));
    preprocessing(fns)
}

pub fn labeling_from_yaml(path: &Path) -> LabelingFn {
    labeling_from_str(&read_yaml(path))
}

pub fn labeling_from_str(s: &str) -> LabelingFn {
    let cfg: LabelingConfig = serde_yaml::from_str(s)
        .expect(&format!("could not deserialize from yaml string\n{}", s));
    labeling(cfg)
}

#[pyclass]
struct DataLoader {
    iter: Box<dyn Iterator<Item=Batch> + Send + 'static>,
    len: usize,
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
    shuffle_prefetch_factor = "4",
    seed = "None"
    )]
    pub fn from_sequences(
        sequences: Vec<String>,
        pipeline_config: PipelineConfig,
        languages: Option<Vec<String>>,
        num_threads: u8,
        buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        shuffle: bool,
        shuffle_prefetch_factor: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if sequences.is_empty() {
            return Err(PyTypeError::new_err("sequences is empty"));
        }
        if languages.is_some() && sequences.len() == languages.as_ref().unwrap().len() {
            return Err(PyTypeError::new_err(
                format!(
                    "there must be one language for every sequence if specified, but \
                    got {} sequences and {} languages",
                    sequences.len(),
                    languages.as_ref().unwrap().len())));
        }
        if shuffle && seed.is_none() {
            return Err(PyTypeError::new_err("seed cannot be None if shuffle is true"));
        }
        let cont = TextContainer::new_boxed(
            sequences,
            None,
            languages,
        );
        let gen = TextGenerator::new(vec![cont]);
        let pipe = Pipeline::from_config(pipeline_config);
        let text_iter = gen.sequential();
        let len = text_iter.min_len();
        let iter = if num_threads > 0 {
            pipe.apply_iter_threaded(text_iter, num_threads, buffer_size)
        } else {
            pipe.apply_iter(text_iter)
        };
        let batched_iter = iter
            .batched(batch_limit, batch_limit_type, shuffle, shuffle_prefetch_factor, seed);
        Ok(DataLoader {
            iter: Box::new(batched_iter),
            len,
        })
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
    shuffle_prefetch_factor = "4",
    seed = "None"
    )]
    pub fn from_files(
        files: Vec<String>,
        pipeline_config: PipelineConfig,
        languages: Option<Vec<String>>,
        strategy: TextIterationStrategy,
        num_threads: u8,
        mut buffer_size: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        shuffle: bool,
        shuffle_prefetch_factor: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if files.is_empty() {
            return Err(PyTypeError::new_err("files is empty"));
        }
        if languages.is_some() && files.len() != languages.as_ref().unwrap().len() {
            return Err(PyTypeError::new_err(
                format!(
                    "there must be one language for every file if specified, but \
                    got {} files and {} languages",
                    files.len(),
                    languages.as_ref().unwrap().len())));
        }
        if shuffle && seed.is_none() {
            return Err(PyTypeError::new_err("seed cannot be None if shuffle is true"));
        }
        if batch_limit_type == BatchLimitType::BatchSize {
            buffer_size = buffer_size.max(batch_limit);
        }
        let cont = files
            .into_iter()
            .enumerate()
            .map(|(idx, file)| {
                let lang = if languages.is_some() {
                    Some(languages.as_ref().unwrap()[idx].clone())
                } else {
                    None
                };
                TextFile::new_boxed(
                    &file.into(), None, lang,
                ) as Box<dyn TextGen>
            })
            .collect();
        let gen = TextGenerator::new(cont);
        let pipe = Pipeline::from_config(pipeline_config);
        let text_iter = gen.with_strategy(strategy, seed);
        let len = text_iter.min_len();
        let iter = if num_threads > 0 {
            pipe.apply_iter_threaded(
                text_iter,
                num_threads,
                buffer_size,
            )
        } else {
            pipe.apply_iter(text_iter)
        };
        let batched_iter = iter
            .batched(batch_limit, batch_limit_type, shuffle, shuffle_prefetch_factor, seed);
        Ok(DataLoader {
            iter: Box::new(batched_iter),
            len,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Py<Batch>> {
        if let Some(batch) = self.iter.next() {
            Some(Python::with_gil(|py| {
                let item: Py<Batch> = Py::new(py, batch).expect("should not fail");
                item
            }))
        } else {
            None
        }
    }

    fn __len__(&self) -> usize {
        self.len
    }
}

pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "data")?;
    m.add_class::<DataLoader>()?;
    m.add_class::<PipelineConfig>()?;
    m.add_class::<TextData>()?;
    m.add_class::<Item>()?;
    m.add_class::<Batch>()?;
    m.add_class::<TextIterationStrategy>()?;
    m.add_class::<BatchLimitType>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
