use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::{
    prefix::PrefixTreeSearch,
    unicode::{normalize, Normalization},
};

#[derive(Serialize, Deserialize)]
pub struct PrefixVec<V> {
    pub data: Vec<(Vec<u8>, V)>,
}

impl<V> Default for PrefixVec<V> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<V> PrefixVec<V> {
    #[inline]
    fn binary_search(
        &self,
        key: &u8,
        depth: usize,
        mut left: usize,
        mut right: usize,
    ) -> anyhow::Result<usize, usize> {
        // perform binary search over bytes at given depth
        let mut size = right - left;
        while left < right {
            let mid = left + size / 2;

            let cur = &self.data[mid].0;

            let cmp = if depth >= cur.len() {
                Ordering::Greater
            } else {
                cur[depth].cmp(key)
            };

            if cmp == Ordering::Less {
                left = mid + 1;
            } else if cmp == Ordering::Greater {
                right = mid;
            } else {
                return Ok(mid);
            }
            size = right - left;
        }
        Err(left)
    }

    #[inline]
    fn range_search(
        &self,
        key: &u8,
        depth: usize,
        left: usize,
        right: usize,
    ) -> Option<(usize, usize)> {
        let idx = match self.binary_search(key, depth, left, right) {
            Err(_) => return None,
            Ok(idx) => idx,
        };

        // exponential search to overshoot right bound
        let mut right_idx = idx;
        let mut i = 0;
        while right_idx < right
            && depth < self.data[right_idx].0.len()
            && self.data[right_idx].0[depth] <= *key
        {
            right_idx += 2usize.pow(i);
            i += 1;
        }
        assert!(i > 0);

        // search linearly from the previous exponential search position
        // to find right bound
        right_idx = idx + 2usize.pow(i - 1);
        while right_idx < right
            && depth < self.data[right_idx].0.len()
            && self.data[right_idx].0[depth] <= *key
        {
            right_idx += 1;
        }

        // now do the same for the left bound, a little bit
        // different here since left is inclusive
        let mut left_idx = idx;
        i = 0;
        while left_idx > left
            && depth < self.data[left_idx - 1].0.len()
            && self.data[left_idx - 1].0[depth] >= *key
        {
            left_idx = left_idx.saturating_sub(2usize.pow(i));
            i += 1;
        }

        if i > 0 {
            left_idx = idx.saturating_sub(2usize.pow(i - 1));
            while left_idx > left
                && depth < self.data[left_idx - 1].0.len()
                && self.data[left_idx - 1].0[depth] >= *key
            {
                left_idx -= 1;
            }
        }
        Some((left_idx, right_idx))
    }

    #[inline]
    fn find(&self, key: &[u8]) -> (Option<(usize, usize)>, Option<&V>) {
        let mut left = 0;
        let mut right = self.size();
        for (depth, k) in key.iter().enumerate() {
            let Some((new_left, new_right)) = self.range_search(k, depth, left, right) else {
                return (None, None);
            };
            left = new_left;
            right = new_right;
        }
        let indices = Some((left, right));
        if self.data[left].0.len() != key.len() {
            (indices, None)
        } else {
            (indices, Some(&self.data[left].1))
        }
    }
}

impl<V> PrefixTreeSearch<V> for PrefixVec<V> {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn insert(&mut self, key: &str, value: V) {
        let key = normalize(key, Normalization::NFKC, true);
        match self
            .data
            .binary_search_by(|(prefix, _)| prefix.as_slice().cmp(key.as_bytes()))
        {
            Ok(idx) => self.data[idx].1 = value,
            Err(idx) => self.data.insert(idx, (key.as_bytes().to_vec(), value)),
        };
    }

    fn get(&self, prefix: &[u8]) -> Option<&V> {
        let (_, value) = self.find(prefix);
        value
    }

    fn contains(&self, prefix: &[u8]) -> bool {
        let (indices, _) = self.find(prefix);
        indices.is_some()
    }

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (String, &V)> + '_> {
        match self.find(prefix) {
            (None, _) => Box::new(std::iter::empty()),
            (Some((left, right)), _) => Box::new(
                self.data[left..right]
                    .iter()
                    .map(|(k, v)| (String::from_utf8_lossy(k.as_slice()).to_string(), v)),
            ),
        }
    }

    fn contains_continuations(&self, prefix: &[u8], continuations: &[Vec<u8>]) -> Vec<bool> {
        match self.find(prefix) {
            (None, _) => vec![false; continuations.len()],
            (Some((left, right)), _) => continuations
                .iter()
                .map(|cont| {
                    let mut cont_left = left;
                    let mut cont_right = right;
                    for (depth, k) in cont.iter().enumerate() {
                        let Some((new_left, new_right)) =
                            self.range_search(k, prefix.len() + depth, cont_left, cont_right) else {
                            return false;
                        };
                        cont_left = new_left;
                        cont_right = new_right;
                    }
                    self.data[left].0.len() == cont.len()
                })
                .collect(),
        }
    }
}

impl<S, V> FromIterator<(S, V)> for PrefixVec<V>
where
    S: AsRef<str>,
{
    fn from_iter<T: IntoIterator<Item = (S, V)>>(iter: T) -> Self {
        let mut pfx = Self::default();
        for (key, value) in iter {
            if key.as_ref().is_empty() {
                continue;
            }
            let key = normalize(key.as_ref(), Normalization::NFKC, true);
            pfx.data.push((key.as_bytes().to_vec(), value));
        }
        // sort by prefix
        pfx.data.sort_by(|(a, _), (b, _)| a.cmp(b));
        // mark duplicate prefixes
        let mark: Vec<_> = pfx
            .data
            .iter()
            .enumerate()
            .map(|(i, (key, _))| {
                if i == 0 {
                    false
                } else {
                    key == &pfx.data[i - 1].0
                }
            })
            .collect();
        // filter out duplicate prefixes
        pfx.data = pfx
            .data
            .into_iter()
            .zip(mark)
            .filter_map(|((k, v), mark)| if mark { None } else { Some((k, v)) })
            .collect();
        pfx
    }
}

#[cfg(test)]
mod tests {
    use crate::prefix::PrefixTreeSearch;

    use super::PrefixVec;

    #[test]
    fn test_prefix_vec() {
        let pfx = [
            ("a", 1),
            ("ab", 2),
            ("abc", 3),
            ("abcd", 4),
            ("abcde", 5),
            ("abd", 6),
            ("bde", 7),
        ]
        .into_iter()
        .collect::<PrefixVec<_>>();

        assert_eq!(pfx.get("a".as_bytes()), Some(&1));
        assert!(pfx.contains("a".as_bytes()));
        assert!(!pfx.contains("bbd".as_bytes()));
        assert_eq!(pfx.get("ab".as_bytes()), Some(&2));
        assert_eq!(pfx.get("abf".as_bytes()), None);
        assert!(!pfx.contains("abf".as_bytes()));
        assert_eq!(pfx.get("abd".as_bytes()), Some(&6));
    }
}
